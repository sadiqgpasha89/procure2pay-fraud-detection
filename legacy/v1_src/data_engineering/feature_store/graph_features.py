"""Graph and network collusion feature computation."""
from typing import Any, Dict, List, Optional
import networkx as nx
from src.core.models import GraphFindings


class KnowledgeGraphEngine:
    """Network graph engine maintaining entity relationships (Vendors, Employees, Bank Accounts, Addresses)."""

    def __init__(self):
        self.graph = nx.Graph()

    def add_vendor(self, vendor_id: str, name: str, tax_id: str, bank_iban: str, address: str, directors: List[str]):
        """Add vendor entity and its associated attributes as graph nodes and edges."""
        self.graph.add_node(vendor_id, type="VENDOR", name=name, tax_id=tax_id)

        # Attribute nodes
        iban_node = f"BANK:{bank_iban.strip().upper()}"
        self.graph.add_node(iban_node, type="BANK_ACCOUNT")
        self.graph.add_edge(vendor_id, iban_node, relation="USES_BANK")

        addr_node = f"ADDR:{address.strip().lower()}"
        self.graph.add_node(addr_node, type="ADDRESS")
        self.graph.add_edge(vendor_id, addr_node, relation="LOCATED_AT")

        for d in directors:
            d_node = f"PERSON:{d.strip().lower()}"
            self.graph.add_node(d_node, type="DIRECTOR", name=d)
            self.graph.add_edge(vendor_id, d_node, relation="DIRECTED_BY")

    def add_employee(self, employee_id: str, name: str, role: str, address: str, phone: Optional[str] = None):
        """Add employee / approver entity and contact attributes."""
        self.graph.add_node(employee_id, type="EMPLOYEE", name=name, role=role)

        addr_node = f"ADDR:{address.strip().lower()}"
        self.graph.add_node(addr_node, type="ADDRESS")
        self.graph.add_edge(employee_id, addr_node, relation="LIVES_AT")

        if phone:
            phone_node = f"PHONE:{phone.strip()}"
            self.graph.add_node(phone_node, type="PHONE")
            self.graph.add_edge(employee_id, phone_node, relation="USES_PHONE")

    def link_director_to_address(self, director_name: str, address: str):
        """Link a vendor director to a residential address."""
        d_node = f"PERSON:{director_name.strip().lower()}"
        addr_node = f"ADDR:{address.strip().lower()}"
        self.graph.add_edge(d_node, addr_node, relation="RESIDES_AT")

    def analyze_entity_conflicts(
        self,
        vendor_id: str,
        approver_id: Optional[str] = None
    ) -> GraphFindings:
        """Inspect graph for shared addresses, phone numbers, or cross-vendor bank accounts."""
        if not self.graph.has_node(vendor_id):
            return GraphFindings(entity_conflicts_detected=False, details="Vendor not registered in graph.")

        conflicts = []
        collusion_nodes = []

        # 1. Check for shared bank accounts with other vendors
        for neighbor in self.graph.neighbors(vendor_id):
            if self.graph.nodes[neighbor].get("type") == "BANK_ACCOUNT":
                other_vendors = [
                    n for n in self.graph.neighbors(neighbor)
                    if n != vendor_id and self.graph.nodes[n].get("type") == "VENDOR"
                ]
                if other_vendors:
                    conflicts.append(f"Shared settlement bank account with vendors: {', '.join(other_vendors)}")
                    collusion_nodes.extend(other_vendors)

        # 2. Check for shared address / phone conflicts between vendor directors and approver
        if approver_id and self.graph.has_node(approver_id):
            # Check 2-hop or 3-hop paths between approver and vendor
            try:
                paths = list(nx.all_simple_paths(self.graph, source=approver_id, target=vendor_id, cutoff=3))
                for path in paths:
                    # Look for intermediary ADDRESS or PHONE or PERSON
                    for node in path[1:-1]:
                        node_type = self.graph.nodes[node].get("type")
                        if node_type in ("ADDRESS", "PHONE", "PERSON"):
                            conflicts.append(
                                f"Shared {node_type.lower()} found between Vendor leadership and Approver [{approver_id}]: {node}"
                            )
                            collusion_nodes.append(node)
            except Exception:
                pass

        has_conflicts = len(conflicts) > 0
        details_str = "; ".join(conflicts) if has_conflicts else "No conflicting graph linkages detected."

        return GraphFindings(
            entity_conflicts_detected=has_conflicts,
            details=details_str,
            collusion_nodes=list(set(collusion_nodes))
        )

    def compute_graph_metrics(self, vendor_id: str) -> Dict[str, float]:
        """Calculate graph degree centrality and network anomaly flags."""
        if not self.graph.has_node(vendor_id):
            return {
                "degree_centrality_vendor": 0.0,
                "shared_bank_account_cross_vendor_flag": 0.0,
                "shared_contact_attribute_flag": 0.0,
            }

        centrality = nx.degree_centrality(self.graph)
        vendor_cent = round(float(centrality.get(vendor_id, 0.0)), 4)

        # Cross vendor bank check
        shared_bank = 0.0
        shared_contact = 0.0
        for neighbor in self.graph.neighbors(vendor_id):
            if self.graph.nodes[neighbor].get("type") == "BANK_ACCOUNT":
                if len(list(self.graph.neighbors(neighbor))) > 1:
                    shared_bank = 1.0
            if self.graph.nodes[neighbor].get("type") in ("ADDRESS", "PHONE"):
                if len(list(self.graph.neighbors(neighbor))) > 1:
                    shared_contact = 1.0

        return {
            "degree_centrality_vendor": vendor_cent,
            "shared_bank_account_cross_vendor_flag": shared_bank,
            "shared_contact_attribute_flag": shared_contact,
        }
