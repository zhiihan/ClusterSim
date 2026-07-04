from __future__ import annotations
import random
import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import uuid
import rustworkx as rx
import graphsim
from clustersim.simulator.cluster_state import ClusterState


class FrontendClusterState:
    """
    Adapter and wrapper class that extends ClusterState with 2D layout and frontend metadata,
    such as node 2D positions, compound parent groups, custom styling, persistent node values,
    and snapshot-based Undo/Redo capability.
    """

    def __init__(
        self,
        num_nodes: int = 0,
        cluster_state: Optional[ClusterState] = None,
        positions: Optional[
            Union[List[Dict[str, float]], Dict[int, Dict[str, float]]]
        ] = None,
        groups: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize FrontendClusterState.

        Args:
            num_nodes (int): Number of quantum nodes (if cluster_state is not provided).
            cluster_state (ClusterState, optional): Existing ClusterState instance.
            positions (list or dict, optional): Initial 2D positions for nodes.
            groups (dict, optional): Initial compound parent groups definitions.
        """
        if cluster_state is not None:
            self.cluster_state = cluster_state
        else:
            self.cluster_state = ClusterState(num_nodes)

        n = len(self.cluster_state)

        # Initialize node metadata list (1-to-1 with qubits 0..n-1)
        self.node_metadata: List[Dict[str, Any]] = []

        for i in range(n):
            pos = {"x": 0.0, "y": 0.0}
            if positions is not None:
                if isinstance(positions, list) and i < len(positions):
                    pos = dict(positions[i])
                elif isinstance(positions, dict) and i in positions:
                    pos = dict(positions[i])
                else:
                    pos = {
                        "x": float(random.randint(0, 300)),
                        "y": float(random.randint(0, 300)),
                    }
            else:
                pos = {
                    "x": float(random.randint(0, 300)),
                    "y": float(random.randint(0, 300)),
                }

            self.node_metadata.append(
                {
                    "position": pos,
                    "parent": None,
                    "value": uuid.uuid4().hex[:8],
                    "label": str(i),
                    "classes": "",
                    "style": {},
                }
            )

        self.groups: Dict[str, Dict[str, Any]] = copy.deepcopy(groups) if groups else {}
        self.undo_stack: List[Dict[str, Any]] = []
        self.redo_stack: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.cluster_state)

    @property
    def num_nodes(self) -> int:
        return len(self.cluster_state)

    @property
    def stabilizers(self) -> List[str]:
        return self.cluster_state.stabilizers

    @property
    def vertex_operators(self) -> List[str]:
        return self.cluster_state.vertex_operators

    @property
    def adjacency_list(self) -> List[List[int]]:
        return self.cluster_state.adjacency_list

    def __repr__(self) -> str:
        rep = f"FrontendClusterState({len(self)} qubits, {len(self.groups)} groups)\n"
        rep += repr(self.cluster_state)
        return rep

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrontendClusterState):
            return False
        return (
            self.cluster_state == other.cluster_state
            and self.node_metadata == other.node_metadata
            and self.groups == other.groups
        )

    # ------------------------------------------------------------------
    # Undo / Redo Snapshot Management
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Create a complete serializable snapshot of the current state."""
        return {
            "to_cytoscape": self.to_cytoscape_elements(),
            "node_metadata": copy.deepcopy(self.node_metadata),
            "groups": copy.deepcopy(self.groups),
        }

    def push_state(self) -> None:
        """Push current snapshot onto the undo stack and clear the redo stack."""
        self.undo_stack.append(self.snapshot())
        self.redo_stack.clear()

    def undo(self) -> bool:
        """
        Restore the previous state from the undo stack.

        Returns:
            bool: True if undo was successful, False if undo stack was empty.
        """
        if not self.undo_stack:
            return False

        # Save current state to redo stack
        self.redo_stack.append(self.snapshot())

        # Pop previous state
        prev_state = self.undo_stack.pop()
        self._restore_from_snapshot(prev_state)
        return True

    def redo(self) -> bool:
        """
        Restore the next state from the redo stack.

        Returns:
            bool: True if redo was successful, False if redo stack was empty.
        """
        if not self.redo_stack:
            return False

        # Save current state to undo stack
        self.undo_stack.append(self.snapshot())

        # Pop next state
        next_state = self.redo_stack.pop()
        self._restore_from_snapshot(next_state)
        return True

    def _restore_from_snapshot(self, state_dict: Dict[str, Any]) -> None:
        """Restore instance state from a snapshot dictionary."""
        restored = FrontendClusterState.from_cytoscape_elements(
            state_dict["to_cytoscape"]
        )
        self.cluster_state = restored.cluster_state
        self.node_metadata = copy.deepcopy(state_dict["node_metadata"])
        self.groups = copy.deepcopy(state_dict["groups"])

    # ------------------------------------------------------------------
    # Delegated Quantum Gate Operations
    # ------------------------------------------------------------------

    def measure(self, qubit: Union[int, str], force: int = -1, basis: str = "Z") -> int:
        q = self.resolve_qubit(qubit)
        return self.cluster_state.measure(
            q if q is not None else int(qubit), force=force, basis=basis
        )

    def MX(self, qubit: Union[int, str], force: int = -1) -> int:
        q = self.resolve_qubit(qubit)
        return self.cluster_state.MX(q if q is not None else int(qubit), force=force)

    def MY(self, qubit: Union[int, str], force: int = -1) -> int:
        q = self.resolve_qubit(qubit)
        return self.cluster_state.MY(q if q is not None else int(qubit), force=force)

    def MZ(self, qubit: Union[int, str], force: int = -1) -> int:
        q = self.resolve_qubit(qubit)
        return self.cluster_state.MZ(q if q is not None else int(qubit), force=force)

    def M(self, qubit: Union[int, str], force: int = -1) -> int:
        q = self.resolve_qubit(qubit)
        return self.cluster_state.M(q if q is not None else int(qubit), force=force)

    def I(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.I(q if q is not None else int(qubit))

    def X(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.X(q if q is not None else int(qubit))

    def Y(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.Y(q if q is not None else int(qubit))

    def Z(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.Z(q if q is not None else int(qubit))

    def H(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.H(q if q is not None else int(qubit))

    def S(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.S(q if q is not None else int(qubit))

    def SH(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.SH(q if q is not None else int(qubit))

    def S_DAG(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.S_DAG(q if q is not None else int(qubit))

    def CX(self, control: Union[int, str], qubit: Union[int, str]) -> None:
        c = self.resolve_qubit(control)
        q = self.resolve_qubit(qubit)
        self.cluster_state.CX(
            c if c is not None else int(control), q if q is not None else int(qubit)
        )

    def CZ(self, control: Union[int, str], qubit: Union[int, str]) -> None:
        c = self.resolve_qubit(control)
        q = self.resolve_qubit(qubit)
        self.cluster_state.CZ(
            c if c is not None else int(control), q if q is not None else int(qubit)
        )

    def fusion_gate(
        self,
        qubit1: Union[int, str],
        qubit2: Union[int, str],
        fusion_type: str = "XXZZ",
        mode: str = "success",
        force: int = 0,
    ) -> None:
        q1 = self.resolve_qubit(qubit1)
        q2 = self.resolve_qubit(qubit2)
        self.cluster_state.fusion_gate(
            q1 if q1 is not None else int(qubit1),
            q2 if q2 is not None else int(qubit2),
            fusion_type=fusion_type,
            mode=mode,
            force=force,
        )

    def local_complementation(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.local_complementation(q if q is not None else int(qubit))

    def local_complementation_rewrite(self, qubit: Union[int, str]) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.local_complementation_rewrite(
            q if q is not None else int(qubit)
        )

    def apply_VOP(
        self, qubit: Union[int, str], vop: Union[Tuple[int, int], int, str]
    ) -> None:
        q = self.resolve_qubit(qubit)
        self.cluster_state.apply_VOP(q if q is not None else int(qubit), vop)

    def add_edge(self, qubit1: Union[int, str], qubit2: Union[int, str]) -> None:
        q1 = self.resolve_qubit(qubit1)
        q2 = self.resolve_qubit(qubit2)
        self.cluster_state.add_edge(
            q1 if q1 is not None else int(qubit1), q2 if q2 is not None else int(qubit2)
        )

    def remove_edge(self, qubit1: Union[int, str], qubit2: Union[int, str]) -> None:
        q1 = self.resolve_qubit(qubit1)
        q2 = self.resolve_qubit(qubit2)
        self.cluster_state.remove_edge(
            q1 if q1 is not None else int(qubit1), q2 if q2 is not None else int(qubit2)
        )

    def toggle_edge(self, qubit1: Union[int, str], qubit2: Union[int, str]) -> None:
        q1 = self.resolve_qubit(qubit1)
        q2 = self.resolve_qubit(qubit2)
        self.cluster_state.toggle_edge(
            q1 if q1 is not None else int(qubit1), q2 if q2 is not None else int(qubit2)
        )

    # ------------------------------------------------------------------
    # Node Addition & Removal (with Metadata & Index Re-mapping)
    # ------------------------------------------------------------------

    def add_node(
        self,
        vop: str = "IA",
        position: Optional[Dict[str, float]] = None,
        group: Optional[str] = None,
        value: Optional[Any] = None,
        label: Optional[str] = None,
    ) -> int:
        """
        Add a new quantum node to ClusterState and initialize its frontend metadata.
        """
        idx = self.num_nodes
        self.cluster_state.add_node(vop=vop)

        if position is None:
            position = {
                "x": float(random.randint(0, 300)),
                "y": float(random.randint(0, 300)),
            }

        if value is None:
            value = uuid.uuid4().hex[:8]

        if label is None:
            label = str(idx)

        self.node_metadata.append(
            {
                "position": dict(position),
                "parent": group,
                "value": value,
                "label": label,
                "classes": "",
                "style": {},
            }
        )
        return idx

    def resolve_qubit(self, identifier: Union[int, str]) -> Optional[int]:
        """
        Translate a Cytoscape node ID, qubit index, or persistent value into a valid ClusterState qubit index (0..N-1).
        """
        # 1. If identifier is string digit matching Cytoscape element id str(i)
        if isinstance(identifier, str) and identifier.isdigit():
            q = int(identifier)
            if 0 <= q < len(self):
                return q

        # 2. Check if integer identifier matches persistent value attribute
        for i, meta in enumerate(self.node_metadata):
            if meta.get("value") == identifier or str(meta.get("value")) == str(
                identifier
            ):
                return i

        # 3. Fallback to raw integer index within bounds
        if isinstance(identifier, int) and 0 <= identifier < len(self):
            return identifier

        return None

    def remove_node(self, qubits: Union[int, str, List[Union[int, str]]]) -> List[int]:
        """
        Remove one or more nodes.
        Translates string IDs, qubit indices, or values to valid qubit indices.
        ClusterState automatically relabels remaining nodes contiguously from 0 to N-1.
        FrontendClusterState transfers metadata (position, group, persistent value, style)
        from old node indices to their new contiguous indices.
        """
        if isinstance(qubits, (int, str)):
            raw_list = [qubits]
        else:
            raw_list = list(qubits)

        resolved_qubits = []
        for q in raw_list:
            idx = self.resolve_qubit(q)
            if idx is not None and idx not in resolved_qubits:
                resolved_qubits.append(idx)

        if not resolved_qubits:
            return []

        qubits_to_remove = set(resolved_qubits)

        # Call ClusterState remove_node which relabels remaining nodes
        self.cluster_state.remove_node(list(qubits_to_remove))

        # Re-map node metadata list: keep only non-removed nodes in original relative order
        new_metadata = []
        for i, meta in enumerate(self.node_metadata):
            if i not in qubits_to_remove:
                new_metadata.append(meta)

        self.node_metadata = new_metadata
        return resolved_qubits

    # ------------------------------------------------------------------
    # Frontend Metadata & Group Management
    # ------------------------------------------------------------------

    def get_position(self, qubit: Union[int, str]) -> Dict[str, float]:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        return self.node_metadata[idx]["position"]

    def set_position(self, qubit: Union[int, str], position: Dict[str, float]) -> None:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        self.node_metadata[idx]["position"] = dict(position)

    def get_group(self, qubit: Union[int, str]) -> Optional[str]:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        return self.node_metadata[idx]["parent"]

    def set_group(self, qubit: Union[int, str], group_id: Optional[str]) -> None:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        if group_id is not None and group_id not in self.groups:
            self.add_group(group_id)
        self.node_metadata[idx]["parent"] = group_id

    def get_value(self, qubit: Union[int, str]) -> Any:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        return self.node_metadata[idx]["value"]

    def set_value(self, qubit: Union[int, str], value: Any) -> None:
        q = self.resolve_qubit(qubit)
        idx = q if q is not None else int(qubit)
        self.node_metadata[idx]["value"] = value

    def add_group(
        self,
        group_id: str,
        label: Optional[str] = None,
        style: Optional[Dict[str, Any]] = None,
        value: Optional[str] = None,
    ) -> None:
        """Add or update a compound parent group container."""
        if label is None:
            label = ""
        if style is None:
            style = {}
        if value is None:
            if group_id in self.groups and self.groups[group_id].get("value"):
                value = self.groups[group_id]["value"]
            elif group_id.startswith("group_") and len(group_id) > 6:
                value = group_id[6:]
            else:
                value = uuid.uuid4().hex[:8]

        self.groups[group_id] = {"label": label, "style": style, "value": value}

    def remove_group(self, group_id: str, unassign_children: bool = True) -> None:
        """Remove a compound group container."""
        if group_id in self.groups:
            del self.groups[group_id]
        if unassign_children:
            for meta in self.node_metadata:
                if meta.get("parent") == group_id:
                    meta["parent"] = None

    # ------------------------------------------------------------------
    # Cytoscape Converters
    # ------------------------------------------------------------------

    def to_cytoscape_elements(self) -> List[Dict[str, Any]]:
        """
        Export complete 2D layout and quantum state to Cytoscape elements list.
        """
        elements: List[Dict[str, Any]] = []

        # 1. Compound Parent Group Container Nodes
        for group_id, group_info in self.groups.items():
            val = group_info.get("value")
            if not val:
                val = (
                    group_id[6:]
                    if group_id.startswith("group_") and len(group_id) > 6
                    else uuid.uuid4().hex[:8]
                )
                group_info["value"] = val
            elements.append(
                {
                    "data": {
                        "id": str(group_id),
                        "label": group_info.get("label", ""),
                        "value": val,
                    },
                    "classes": "group container",
                }
            )

        # 2. Qubit Nodes
        for i in range(len(self)):
            meta = self.node_metadata[i]
            node_data = {
                "id": str(i),
                "label": meta.get("label", str(i)),
                "value": meta.get("value", i),
                "vop": self.vertex_operators[i],
            }
            if meta.get("shape"):
                node_data["shape"] = meta["shape"]
            elif meta.get("style", {}).get("shape"):
                node_data["shape"] = meta["style"]["shape"]

            if meta.get("parent"):
                node_data["parent"] = str(meta["parent"])
            else:
                node_data["parent"] = None

            node_elem = {
                "data": node_data,
                "position": meta.get("position", {"x": 0.0, "y": 0.0}),
            }
            if meta.get("classes"):
                node_elem["classes"] = meta["classes"]
            if meta.get("style"):
                node_elem["style"] = meta["style"]

            elements.append(node_elem)

        # 3. Edges between Qubit Nodes
        for u, neighbors in enumerate(self.adjacency_list):
            for v in neighbors:
                if u < v:
                    elements.append(
                        {
                            "data": {
                                "source": str(u),
                                "target": str(v),
                            }
                        }
                    )

        return elements

    @classmethod
    def from_cytoscape_elements(
        cls, elements: List[Dict[str, Any]]
    ) -> FrontendClusterState:
        """
        Reconstruct FrontendClusterState from a list of Cytoscape elements.
        """
        groups: Dict[str, Dict[str, Any]] = {}
        node_elements: List[Dict[str, Any]] = []
        edge_elements: List[Dict[str, Any]] = []

        for item in elements:
            data = item.get("data", {})
            # Distinguish group nodes vs qubit nodes vs edges
            if "source" in data and "target" in data:
                edge_elements.append(item)
            elif "vop" in data or (
                data.get("value") is not None and str(data.get("id", "")).isdigit()
            ):
                node_elements.append(item)
            elif "id" in data:
                # Group container node
                group_id = str(data["id"])
                val = data.get("value")
                if not val:
                    val = (
                        group_id[6:]
                        if group_id.startswith("group_") and len(group_id) > 6
                        else uuid.uuid4().hex[:8]
                    )
                groups[group_id] = {
                    "label": data.get("label", group_id),
                    "style": item.get("style", {}),
                    "value": val,
                }

        # Sort qubit nodes by integer ID
        def get_node_id(elem):
            try:
                return int(elem["data"]["id"])
            except (ValueError, KeyError):
                return 0

        node_elements.sort(key=get_node_id)
        num_nodes = len(node_elements)

        # Build PyGraph for ClusterState reconstruction
        rx_graph = rx.PyGraph()
        for i in range(num_nodes):
            elem = node_elements[i]
            vop_val = elem.get("data", {}).get("vop", "IA")
            rx_graph.add_node({"vop": vop_val})

        for edge in edge_elements:
            u = int(edge["data"]["source"])
            v = int(edge["data"]["target"])
            rx_graph.add_edge(u, v, None)

        base_cluster_state = ClusterState.from_rustworkx(rx_graph)
        frontend_state = cls(
            num_nodes=num_nodes, cluster_state=base_cluster_state, groups=groups
        )

        # Restore node metadata
        for i in range(num_nodes):
            elem = node_elements[i]
            data = elem.get("data", {})
            pos = elem.get("position", {"x": 0.0, "y": 0.0})
            node_shape = data.get("shape") or elem.get("style", {}).get("shape")
            style_dict = dict(elem.get("style", {}))
            if node_shape:
                style_dict["shape"] = node_shape
            frontend_state.node_metadata[i] = {
                "position": dict(pos),
                "parent": data.get("parent"),
                "value": data.get("value", i),
                "label": data.get("label", str(i)),
                "shape": node_shape,
                "classes": elem.get("classes", ""),
                "style": style_dict,
            }

        return frontend_state

    # ------------------------------------------------------------------
    # JSON & Dict Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert FrontendClusterState to a serializable dictionary."""
        return {
            "cluster_state_json": self.cluster_state.to_json(),
            "node_metadata": self.node_metadata,
            "groups": self.groups,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrontendClusterState:
        """Create FrontendClusterState from a dictionary."""
        base_state = ClusterState.from_json(data["cluster_state_json"])
        obj = cls(
            num_nodes=len(base_state),
            cluster_state=base_state,
            groups=data.get("groups", {}),
        )
        if "node_metadata" in data:
            obj.node_metadata = copy.deepcopy(data["node_metadata"])
        return obj

    def to_json(self) -> str:
        """Serialize FrontendClusterState to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> FrontendClusterState:
        """Deserialize FrontendClusterState from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
