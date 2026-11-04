"""Knowledge graph of vendors, employees and their shared attributes (bank, address, phone, directors).

MVP: NetworkX built in memory from the warehouse (~1k nodes, <10 ms to build). The `KnowledgeGraph`
surface is what `mcp-graph` and the feature builder consume, so Neo4j / Spanner Graph / AlloyDB can
replace it without touching callers. Edges are created only on *exact normalised matches* of a
registered attribute (supplier-name collisions alone never create an edge — spec §9 / edge case 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from p2p.common.models import GraphFindings
from p2p.data.repository import Warehouse

ATTR_TYPES = ("BANK", "ADDR", "PHONE")


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(",", " ").split())


@dataclass
class Link:
    attribute_type: str
    attribute: str
    entity_a: str
    entity_b: str

    @property
    def record_id(self) -> str:
        return f"EDGE-{self.attribute_type}-{self.entity_a}-{self.entity_b}"


@dataclass
class LinkAnalysis:
    vendor_id: str
    links: list[Link] = field(default_factory=list)

    @property
    def vendor_vendor(self) -> list[Link]:
        return [link for link in self.links if link.entity_b.startswith("VEND")]

    @property
    def vendor_employee(self) -> list[Link]:
        return [link for link in self.links if link.entity_b.startswith("EMP")]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.Graph()

    @classmethod
    def from_warehouse(cls, wh: Warehouse) -> "KnowledgeGraph":
        kg = cls()
        for v in wh.vendors():
            kg.g.add_node(v.vendor_id, kind="VENDOR", name=v.legal_name)
            kg._attach(v.vendor_id, "BANK", v.bank_iban)
            for c in wh.bank_changes(v.vendor_id):
                kg._attach(v.vendor_id, "BANK", c.new_iban)
            kg._attach(v.vendor_id, "ADDR", v.address)
            if v.phone:
                kg._attach(v.vendor_id, "PHONE", v.phone)
            # Director names are deliberately NOT graph attributes: a shared name alone is a supplier-name
            # collision, not a link (spec §9). Links need exact bank / address / phone matches.
        for e in wh.employees():
            kg.g.add_node(e.employee_id, kind="EMPLOYEE", name=e.name, role=e.role)
            kg._attach(e.employee_id, "ADDR", e.address)
            if e.phone:
                kg._attach(e.employee_id, "PHONE", e.phone)
        return kg

    def _attach(self, entity: str, attr_type: str, value: str) -> None:
        node = f"{attr_type}:{_norm(value)}"
        self.g.add_node(node, kind=attr_type, value=value)
        self.g.add_edge(entity, node)

    # ── analysis ────────────────────────────────────────────────────────────
    def link_analysis(self, vendor_id: str) -> LinkAnalysis:
        out = LinkAnalysis(vendor_id)
        if vendor_id not in self.g:
            return out
        for attr in self.g.neighbors(vendor_id):
            kind = self.g.nodes[attr]["kind"]
            if kind not in ATTR_TYPES:
                continue
            for other in self.g.neighbors(attr):
                if other == vendor_id or self.g.nodes[other]["kind"] not in ("VENDOR", "EMPLOYEE"):
                    continue
                out.links.append(Link(kind, self.g.nodes[attr]["value"], vendor_id, other))
        out.links.sort(key=lambda link: (link.attribute_type, link.entity_b))
        return out

    def metrics(self, vendor_id: str, staff: tuple[str | None, ...] = ()) -> dict[str, float]:
        """Graph features for Tier 2. `staff` = employees touching this transaction (PO buyer/approver, AP clerk)."""
        la = self.link_analysis(vendor_id)
        staff_ids = {s for s in staff if s}
        v_emp = {link.entity_b for link in la.vendor_employee}
        return {
            "vendor_shared_bank_flag": float(any(link.attribute_type == "BANK" for link in la.vendor_vendor)),
            "vendor_shared_address_flag": float(any(link.attribute_type in ("ADDR", "PHONE") for link in la.vendor_vendor)),
            "vendor_employee_link_flag": float(bool(v_emp)),
            "staff_vendor_link_flag": float(bool(v_emp & staff_ids)),
            "vendor_degree": float(len(la.links)),
        }

    def findings(self, vendor_id: str, staff: tuple[str | None, ...] = ()) -> GraphFindings:
        la = self.link_analysis(vendor_id)
        if not la.links:
            return GraphFindings()
        staff_ids = {s for s in staff if s}
        parts, nodes = [], []
        for link in la.links:
            tag = "approver/transaction staff" if link.entity_b in staff_ids else ("employee" if link.entity_b.startswith("EMP") else "vendor")
            shown = link.attribute if (link.attribute_type == "ADDR" and link.entity_b.startswith("VEND")) else "<redacted>"
            parts.append(f"{link.entity_a} shares {link.attribute_type.lower()} '{shown}' with {tag} {link.entity_b}"
                         f" [{link.record_id}]")
            nodes.append(link.entity_b)
        conflicts = any(link.entity_b.startswith("EMP") or link.attribute_type == "BANK" for link in la.links)
        return GraphFindings(entity_conflicts_detected=conflicts, details="; ".join(parts), collusion_nodes=sorted(set(nodes)))
