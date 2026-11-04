import hashlib
import pytest

from p2p.data.synthetic import build_world
from p2p.features.builder import FEATURE_ORDER, FeatureBuilder
from p2p.features.graph import KnowledgeGraph
from p2p.rules.engine import RuleEngine


def test_generator_is_deterministic():
    sig = lambda w: hashlib.sha256("".join(i.invoice_id + str(i.total_amount) for i in w.invoices).encode()).hexdigest()  # noqa: E731
    assert sig(build_world(seed=3, n_vendors=40)) == sig(build_world(seed=3, n_vendors=40))
    assert sig(build_world(seed=3, n_vendors=40)) != sig(build_world(seed=4, n_vendors=40))


def test_world_covers_every_fraud_type_with_realistic_base_rate(labels):
    kinds = {k for _, k in labels.values()}
    assert {"LEGIT", "BANK_DIVERSION", "SMURFING", "SHELL_VENDOR", "COLLUSION", "PHANTOM_DELIVERY", "CONTRACT_VIOLATION", "DUPLICATE", "SANCTIONED_VENDOR"} <= kinds
    rate = sum(f for f, _ in labels.values()) / len(labels)
    assert 0.02 < rate < 0.12


def test_features_are_point_in_time(wh, by_type):
    """Training and serving share one builder; every history read is strictly earlier than the invoice under evaluation, so
    replaying an old invoice reproduces exactly the vector it had live (no look-ahead leakage)."""
    fb = FeatureBuilder(wh, KnowledgeGraph.from_warehouse(wh))
    for iid in by_type["LEGIT"][100:400:60]:
        inv = wh.get_invoice(iid)
        feats = fb.build(inv).features
        assert list(feats) == FEATURE_ORDER and all(isinstance(v, float) for v in feats.values())
        assert all(h.invoice_date < inv.invoice_date for h in wh.vendor_invoices(inv.vendor_id, inv.invoice_date, exclude=iid))
        assert fb.build(inv).features == feats  # deterministic replay
    # a duplicate is only visible to the *later* of the two invoices
    dup = wh.get_invoice(by_type["DUPLICATE"][0])
    orig = next(h for h in wh.vendor_invoices(dup.vendor_id, dup.invoice_date) if h.total_amount == dup.total_amount)
    assert fb.build(orig).features["near_duplicate_flag"] <= fb.build(dup).features["near_duplicate_flag"] == 1.0


@pytest.mark.parametrize("kind,rule", [("SANCTIONED_VENDOR", "SANCTIONS_OFAC_HIT"), ("DUPLICATE", "EXACT_DUPLICATE_INVOICE"),
                                       ("SMURFING", "THRESHOLD_PROXIMITY_SPLIT_DETECTED"), ("BANK_DIVERSION", "BANK_MODIFIED_WITHIN_72_HOURS")])
def test_tier1_rules_fire_on_their_scenarios(wh, by_type, kind, rule):
    fb, eng = FeatureBuilder(wh, KnowledgeGraph.from_warehouse(wh)), RuleEngine()
    fired = {v for iid in by_type[kind] for v in eng.evaluate(fb.load(iid)).violations}
    assert rule in fired, f"{kind}: expected {rule}, got {fired}"


def test_tier1_is_fast_and_mostly_silent_on_legit(wh, by_type):
    fb, eng = FeatureBuilder(wh, KnowledgeGraph.from_warehouse(wh)), RuleEngine()
    sample = by_type["LEGIT"][::5]
    res = [eng.evaluate(fb.load(i)) for i in sample]
    assert sum(r.is_blocked for r in res) / len(res) < 0.05
    assert max(r.latency_ms for r in res) < 5.0


def test_name_only_collision_creates_no_graph_edge(wh):
    """Two vendors sharing only a director name must not be linked (spec edge case 3)."""
    kg = KnowledgeGraph.from_warehouse(wh)
    by_dir: dict[str, list[str]] = {}
    for v in wh.vendors():
        for d in v.directors:
            by_dir.setdefault(d, []).append(v.vendor_id)
    pair = next(ids for ids in by_dir.values() if len(ids) >= 2)
    a, b = pair[:2]
    assert not any(l.entity_b == b and l.attribute_type in ("PERSON",) for l in kg.link_analysis(a).links)
