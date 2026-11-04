"""Warehouse access layer.

`Warehouse` is the only thing services/tools use to read P2P data. The MVP backs it with SQLite
(zero cost, one file); the same method surface maps 1:1 onto BigQuery/AlloyDB for production —
swap the class, not the callers. Every history query takes an explicit `before` timestamp so the
training pipeline and the online scorer see identical *point-in-time* data (no leakage).

Ground-truth labels live in `LabelStore` (separate class, separate table). Nothing exposed to
agents or MCP tools can reach it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from p2p.common.models import (
    BankChange, Document, Employee, GoodsReceipt, Invoice, PurchaseOrder, Vendor,
)

ISO = "%Y-%m-%dT%H:%M:%S"

SCHEMA = """
CREATE TABLE IF NOT EXISTS vendors (vendor_id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS employees (employee_id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS purchase_orders (po_id TEXT PRIMARY KEY, vendor_id TEXT, issued_at TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS goods_receipts (gr_id TEXT PRIMARY KEY, po_id TEXT, received_at TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY, vendor_id TEXT, invoice_number TEXT, po_id TEXT,
    invoice_date TEXT, cost_center TEXT, total_amount REAL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bank_changes (change_id TEXT PRIMARY KEY, vendor_id TEXT, changed_at TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, doc_type TEXT, vendor_id TEXT, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS labels (invoice_id TEXT PRIMARY KEY, is_fraud INTEGER, fraud_type TEXT);
CREATE INDEX IF NOT EXISTS ix_inv_vendor_date ON invoices(vendor_id, invoice_date);
CREATE INDEX IF NOT EXISTS ix_inv_date ON invoices(invoice_date);
CREATE INDEX IF NOT EXISTS ix_po_vendor ON purchase_orders(vendor_id);
CREATE INDEX IF NOT EXISTS ix_gr_po ON goods_receipts(po_id);
CREATE INDEX IF NOT EXISTS ix_bc_vendor ON bank_changes(vendor_id, changed_at);
CREATE INDEX IF NOT EXISTS ix_doc_vendor ON documents(vendor_id, doc_type);
"""


def iso(dt: datetime) -> str:
    return dt.strftime(ISO)


class _Base:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly

    _tls = threading.local()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:  # one long-lived read-only connection per thread (open/close per query cost ~10x the query)
            cache = self._tls.__dict__.setdefault("conns", {})
            conn = cache.get(str(self.path))
            if conn is None:
                conn = cache[str(self.path)] = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=10)
                conn.row_factory = sqlite3.Row
            yield conn
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


class Warehouse(_Base):
    """Read/write access to the synthetic ERP + master-data warehouse."""

    def init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)

    # ── bulk load (generator only) ──────────────────────────────────────────
    def load(self, *, vendors: Iterable[Vendor], employees: Iterable[Employee], pos: Iterable[PurchaseOrder],
             grs: Iterable[GoodsReceipt], invoices: Iterable[Invoice], bank_changes: Iterable[BankChange],
             documents: Iterable[Document], labels: Iterable[tuple[str, int, str]]) -> None:
        dump = lambda m: m.model_dump_json()  # noqa: E731
        with self._conn() as c:
            c.executemany("INSERT INTO vendors VALUES (?,?)", [(v.vendor_id, dump(v)) for v in vendors])
            c.executemany("INSERT INTO employees VALUES (?,?)", [(e.employee_id, dump(e)) for e in employees])
            c.executemany("INSERT INTO purchase_orders VALUES (?,?,?,?)",
                          [(p.po_id, p.vendor_id, iso(p.issued_at), dump(p)) for p in pos])
            c.executemany("INSERT INTO goods_receipts VALUES (?,?,?,?)",
                          [(g.gr_id, g.po_id, iso(g.received_at), dump(g)) for g in grs])
            c.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)",
                          [(i.invoice_id, i.vendor_id, i.invoice_number, i.po_id, iso(i.invoice_date),
                            i.cost_center, i.total_amount, dump(i)) for i in invoices])
            c.executemany("INSERT INTO bank_changes VALUES (?,?,?,?)",
                          [(b.change_id, b.vendor_id, iso(b.changed_at), dump(b)) for b in bank_changes])
            c.executemany("INSERT INTO documents VALUES (?,?,?,?)",
                          [(d.doc_id, d.doc_type, d.vendor_id, dump(d)) for d in documents])
            c.executemany("INSERT INTO labels VALUES (?,?,?)", list(labels))

    # ── point lookups ───────────────────────────────────────────────────────
    def _one(self, sql: str, args: tuple, model):
        with self._conn() as c:
            row = c.execute(sql, args).fetchone()
        return model.model_validate_json(row["body"]) if row else None

    def _many(self, sql: str, args: tuple, model) -> list:
        with self._conn() as c:
            return [model.model_validate_json(r["body"]) for r in c.execute(sql, args).fetchall()]

    def get_invoice(self, invoice_id: str) -> Invoice | None:
        return self._one("SELECT body FROM invoices WHERE invoice_id=?", (invoice_id,), Invoice)

    def get_vendor(self, vendor_id: str) -> Vendor | None:
        return self._one("SELECT body FROM vendors WHERE vendor_id=?", (vendor_id,), Vendor)

    def get_employee(self, employee_id: str) -> Employee | None:
        return self._one("SELECT body FROM employees WHERE employee_id=?", (employee_id,), Employee)

    def get_po(self, po_id: str) -> PurchaseOrder | None:
        return self._one("SELECT body FROM purchase_orders WHERE po_id=?", (po_id,), PurchaseOrder)

    def get_goods_receipt(self, po_id: str) -> GoodsReceipt | None:
        return self._one("SELECT body FROM goods_receipts WHERE po_id=? ORDER BY received_at LIMIT 1", (po_id,), GoodsReceipt)

    def get_document(self, doc_id: str) -> Document | None:
        return self._one("SELECT body FROM documents WHERE doc_id=?", (doc_id,), Document)

    # ── collections ─────────────────────────────────────────────────────────
    def vendors(self) -> list[Vendor]:
        return self._many("SELECT body FROM vendors ORDER BY vendor_id", (), Vendor)

    def employees(self) -> list[Employee]:
        return self._many("SELECT body FROM employees ORDER BY employee_id", (), Employee)

    def documents(self, doc_type: str | None = None, vendor_id: str | None = None) -> list[Document]:
        sql, args = "SELECT body FROM documents WHERE 1=1", []
        if doc_type:
            sql += " AND doc_type=?"; args.append(doc_type)
        if vendor_id:
            sql += " AND vendor_id=?"; args.append(vendor_id)
        return self._many(sql + " ORDER BY doc_id", tuple(args), Document)

    def invoice_ids(self) -> list[str]:
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT invoice_id FROM invoices ORDER BY invoice_date, invoice_id")]

    def invoices_between(self, start: datetime, end: datetime) -> list[Invoice]:
        return self._many("SELECT body FROM invoices WHERE invoice_date>=? AND invoice_date<? ORDER BY invoice_date",
                          (iso(start), iso(end)), Invoice)

    # ── point-in-time history (all strictly *before* `before`) ──────────────
    def vendor_invoices(self, vendor_id: str, before: datetime, since: datetime | None = None,
                        exclude: str | None = None) -> list[Invoice]:
        sql, args = "SELECT body FROM invoices WHERE vendor_id=? AND invoice_date<?", [vendor_id, iso(before)]
        if since:
            sql += " AND invoice_date>=?"; args.append(iso(since))
        if exclude:
            sql += " AND invoice_id<>?"; args.append(exclude)
        return self._many(sql + " ORDER BY invoice_date", tuple(args), Invoice)

    def window_invoices(self, vendor_id: str, cost_center: str, start: datetime, before: datetime,
                        exclude: str | None = None) -> list[Invoice]:
        return [i for i in self.vendor_invoices(vendor_id, before, since=start, exclude=exclude)
                if i.cost_center == cost_center]

    def same_number_invoices(self, vendor_id: str, invoice_number: str, exclude: str, before: datetime) -> list[Invoice]:
        return self._many(
            "SELECT body FROM invoices WHERE vendor_id=? AND invoice_number=? AND invoice_id<>? AND invoice_date<=?"
            " ORDER BY invoice_date", (vendor_id, invoice_number, exclude, iso(before)), Invoice)

    def vendor_pos(self, vendor_id: str, before: datetime) -> list[PurchaseOrder]:
        return self._many("SELECT body FROM purchase_orders WHERE vendor_id=? AND issued_at<=? ORDER BY issued_at",
                          (vendor_id, iso(before)), PurchaseOrder)

    def bank_changes(self, vendor_id: str, before: datetime | None = None) -> list[BankChange]:
        if before is None:
            return self._many("SELECT body FROM bank_changes WHERE vendor_id=? ORDER BY changed_at", (vendor_id,), BankChange)
        return self._many("SELECT body FROM bank_changes WHERE vendor_id=? AND changed_at<=? ORDER BY changed_at",
                          (vendor_id, iso(before)), BankChange)

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("vendors", "employees", "purchase_orders", "goods_receipts", "invoices",
                              "bank_changes", "documents", "labels")}


class LabelStore(_Base):
    """Ground truth for training/evaluation only. Never imported by agents or MCP servers."""

    def labels(self) -> dict[str, tuple[int, str]]:
        with self._conn() as c:
            return {r["invoice_id"]: (r["is_fraud"], r["fraud_type"]) for r in c.execute("SELECT * FROM labels")}

    def label(self, invoice_id: str) -> tuple[int, str] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM labels WHERE invoice_id=?", (invoice_id,)).fetchone()
        return (r["is_fraud"], r["fraud_type"]) if r else None


def open_warehouse(readonly: bool = True) -> Warehouse:
    from p2p.common.config import get_settings

    return Warehouse(get_settings().warehouse_path, readonly=readonly)


def dumps(o: object) -> str:
    return json.dumps(o, default=str)
