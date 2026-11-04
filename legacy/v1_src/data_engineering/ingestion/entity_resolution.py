"""Multi-attribute entity resolution and fuzzy deduplication engine.

Mitigates Edge Case 3: Supplier Name Collisions by requiring multi-field intersection
of Tax ID, bank routing, registration jurisdiction, and physical address.
"""
import re
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import structlog
from src.core.models import Vendor

logger = structlog.get_logger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in range [0.0, 1.0]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    dist = levenshtein_distance(s1.lower(), s2.lower())
    return 1.0 - (dist / max_len)


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Calculate Jaro-Winkler string similarity in range [0.0, 1.0]."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    # Winkler prefix bonus (up to 4 characters)
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return min(1.0, jaro + prefix * p * (1.0 - jaro))


class EntityMatchResult(BaseModel):
    """Result of entity comparison."""
    target_vendor_id: str
    target_vendor_name: str
    composite_similarity: float = Field(..., ge=0.0, le=1.0)
    name_similarity: float
    address_similarity: float
    tax_id_exact_match: bool
    bank_iban_exact_match: bool
    is_name_collision_imposter: bool = False
    is_confirmed_duplicate: bool = False


class EntityResolver:
    """Multi-Attribute Vendor Deduplication & Name Collision Resolver."""

    def __init__(self, existing_vendors: Optional[List[Vendor]] = None):
        self.vendors: Dict[str, Vendor] = {v.vendor_id: v for v in (existing_vendors or [])}

    def register_vendor(self, vendor: Vendor) -> None:
        """Add or update vendor in registry."""
        self.vendors[vendor.vendor_id] = vendor

    def normalize_name(self, name: str) -> str:
        """Standardize corporate suffixes and punctuation."""
        s = name.lower()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\b(inc|llc|corp|ltd|limited|corporation|co|gmbh|sa)\b", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def compare(self, vendor_a: Vendor, vendor_b: Vendor) -> EntityMatchResult:
        """Compare two vendor profiles across multiple orthogonal attributes."""
        norm_a = self.normalize_name(vendor_a.legal_name)
        norm_b = self.normalize_name(vendor_b.legal_name)

        name_jw = jaro_winkler_similarity(norm_a, norm_b)
        name_lev = levenshtein_similarity(norm_a, norm_b)
        name_score = (name_jw * 0.7) + (name_lev * 0.3)

        addr_score = jaro_winkler_similarity(vendor_a.address.lower(), vendor_b.address.lower())
        tax_match = (
            vendor_a.tax_id.strip().upper() == vendor_b.tax_id.strip().upper()
            if vendor_a.tax_id and vendor_b.tax_id else False
        )
        bank_match = (
            vendor_a.bank_iban.strip().upper() == vendor_b.bank_iban.strip().upper()
            if vendor_a.bank_iban and vendor_b.bank_iban else False
        )

        # Composite score
        tax_weight = 0.40 if tax_match else 0.0
        bank_weight = 0.30 if bank_match else 0.0
        name_weight = name_score * 0.20
        addr_weight = addr_score * 0.10
        composite = tax_weight + bank_weight + name_weight + addr_weight

        # Mitigate Edge Case 3: High name similarity with different Tax ID & Bank details is an imposter collision!
        is_collision_imposter = (
            name_score >= 0.88 and not tax_match and not bank_match
        )

        is_duplicate = (
            (tax_match and name_score >= 0.70) or
            (bank_match and name_score >= 0.80) or
            composite >= 0.85
        )

        return EntityMatchResult(
            target_vendor_id=vendor_b.vendor_id,
            target_vendor_name=vendor_b.legal_name,
            composite_similarity=round(composite, 4),
            name_similarity=round(name_score, 4),
            address_similarity=round(addr_score, 4),
            tax_id_exact_match=tax_match,
            bank_iban_exact_match=bank_match,
            is_name_collision_imposter=is_collision_imposter,
            is_confirmed_duplicate=is_duplicate,
        )

    def find_matches(self, candidate: Vendor, threshold: float = 0.60) -> List[EntityMatchResult]:
        """Search registered vendor database for potential matches and collision imposters."""
        results = []
        for vid, v in self.vendors.items():
            if vid == candidate.vendor_id:
                continue
            res = self.compare(candidate, v)
            if res.composite_similarity >= threshold or res.is_name_collision_imposter:
                results.append(res)
        results.sort(key=lambda x: x.composite_similarity, reverse=True)
        return results
