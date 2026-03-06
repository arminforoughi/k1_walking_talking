"""
Semantic Scene Graph — computes spatial relationships between tracked objects.

Builds a graph of nodes (objects) and edges (spatial relationships) from
3D positions, bounding boxes, and semantic labels.
"""

import math
import threading
import time
from dataclasses import dataclass, field


@dataclass
class SceneNode:
    id: str
    label: str
    x: float
    y: float  # height
    z: float
    sx: float = 0.3
    sy: float = 0.3
    sz: float = 0.3
    last_seen: float = 0.0


@dataclass
class SceneEdge:
    src: str
    dst: str
    relationship: str


class SceneGraph:
    """Maintains a semantic graph of object spatial relationships."""

    # Vertical tolerance for "same level" relationships
    HEIGHT_TOLERANCE = 0.3
    # Max distance for "next_to" relationship
    PROXIMITY_THRESHOLD = 2.0
    # Vertical offset threshold for "on_top_of"
    ON_TOP_THRESHOLD = 0.15
    # Support surfaces
    SUPPORT_SURFACES = {'dining table', 'desk', 'table', 'counter', 'bed',
                        'couch', 'sofa', 'bench', 'shelf', 'nightstand'}

    def __init__(self):
        self._lock = threading.Lock()
        self.nodes: dict[str, SceneNode] = {}
        self.edges: list[SceneEdge] = []
        self._last_update = 0.0
        self._update_interval = 1.0

    def update_from_objects(self, tracked_objects):
        """Sync nodes from scene_reconstructor's tracked objects list."""
        now = time.time()
        if now - self._last_update < self._update_interval:
            return
        self._last_update = now

        with self._lock:
            current_ids = set()
            for obj in tracked_objects:
                node_id = f"{obj['label']}_{id(obj) % 10000}"
                # Try to match existing node by label + proximity
                matched_id = self._find_matching_node(obj)
                if matched_id:
                    node_id = matched_id
                    node = self.nodes[node_id]
                    alpha = 0.3
                    node.x = node.x * (1 - alpha) + obj['x'] * alpha
                    node.y = node.y * (1 - alpha) + obj['z'] * alpha
                    node.z = node.z * (1 - alpha) + obj['y'] * alpha
                    node.sx = obj.get('sx', 0.3)
                    node.sy = obj.get('sz', 0.3)
                    node.sz = obj.get('sy', 0.3)
                    node.last_seen = now
                else:
                    node_id = f"{obj['label']}_{len(self.nodes)}"
                    self.nodes[node_id] = SceneNode(
                        id=node_id,
                        label=obj['label'],
                        x=obj['x'],
                        y=obj['z'],   # height in Three.js Y
                        z=obj['y'],   # depth in Three.js Z
                        sx=obj.get('sx', 0.3),
                        sy=obj.get('sz', 0.3),
                        sz=obj.get('sy', 0.3),
                        last_seen=now,
                    )
                current_ids.add(node_id)

            # Remove stale nodes
            stale = [nid for nid, n in self.nodes.items()
                     if now - n.last_seen > 120]
            for nid in stale:
                del self.nodes[nid]

            self._compute_relationships()

    def _find_matching_node(self, obj):
        """Find an existing node that matches this object by label and proximity."""
        for nid, node in self.nodes.items():
            if node.label == obj['label']:
                dx = node.x - obj['x']
                dz = node.z - obj['y']
                dist = math.sqrt(dx * dx + dz * dz)
                if dist < 1.5:
                    return nid
        return None

    def _compute_relationships(self):
        """Compute spatial relationships between all pairs of nearby objects."""
        self.edges = []
        node_list = list(self.nodes.values())
        n = len(node_list)

        for i in range(n):
            for j in range(i + 1, n):
                a = node_list[i]
                b = node_list[j]
                rels = self._compute_pairwise(a, b)
                for rel in rels:
                    self.edges.append(SceneEdge(src=a.id, dst=b.id, relationship=rel))

    def _compute_pairwise(self, a: SceneNode, b: SceneNode) -> list[str]:
        """Compute spatial relationships between two nodes."""
        relationships = []

        dx = b.x - a.x
        dy = b.y - a.y   # vertical (height)
        dz = b.z - a.z
        horiz_dist = math.sqrt(dx * dx + dz * dz)

        # Too far apart for any relationship
        if horiz_dist > self.PROXIMITY_THRESHOLD * 2:
            return relationships

        # On top of / below
        a_top = a.y + a.sy / 2
        b_bottom = b.y - b.sy / 2
        b_top = b.y + b.sy / 2
        a_bottom = a.y - a.sy / 2

        if b_bottom > a_top - self.ON_TOP_THRESHOLD and horiz_dist < max(a.sx, a.sz, 0.5):
            if a.label.lower() in self.SUPPORT_SURFACES:
                relationships.append('on_top_of')
            elif dy > self.HEIGHT_TOLERANCE:
                relationships.append('above')

        if a_bottom > b_top - self.ON_TOP_THRESHOLD and horiz_dist < max(b.sx, b.sz, 0.5):
            if b.label.lower() in self.SUPPORT_SURFACES:
                relationships.append('supported_by')
            elif dy < -self.HEIGHT_TOLERANCE:
                relationships.append('below')

        # Next to (at similar height, within proximity)
        if (abs(dy) < self.HEIGHT_TOLERANCE and
                horiz_dist < self.PROXIMITY_THRESHOLD and
                not relationships):
            relationships.append('next_to')

        # In front of / behind (relative to scene, using Z axis)
        if abs(dz) > 0.5 and horiz_dist < self.PROXIMITY_THRESHOLD:
            if dz > 0.5 and abs(dx) < abs(dz) * 0.5:
                if 'next_to' not in relationships:
                    relationships.append('in_front_of')
            elif dz < -0.5 and abs(dx) < abs(dz) * 0.5:
                if 'next_to' not in relationships:
                    relationships.append('behind')

        return relationships

    def get_graph_data(self):
        """Return JSON-serializable graph data for frontend."""
        with self._lock:
            nodes = []
            for nid, node in self.nodes.items():
                nodes.append({
                    'id': node.id,
                    'label': node.label,
                    'x': round(node.x, 2),
                    'y': round(node.y, 2),
                    'z': round(node.z, 2),
                })
            edges = []
            for edge in self.edges:
                edges.append({
                    'src': edge.src,
                    'dst': edge.dst,
                    'relationship': edge.relationship,
                })
            return {'nodes': nodes, 'edges': edges}
