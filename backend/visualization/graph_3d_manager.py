#!/usr/bin/env python3
"""
3D Network Graph Data Structures for Advanced Visualization
Spatial coordinate storage, layout algorithms, and real-time updates
"""

import logging
import math
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


@dataclass
class Node3D:
    """3D positioned node for network visualization"""

    node_id: str
    node_type: str  # user, merchant, device, transaction
    position: Tuple[float, float, float]  # (x, y, z)
    size: float
    color: str
    label: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class Edge3D:
    """3D edge with spatial properties"""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    weight: float
    color: str
    properties: Dict[str, Any]
    created_at: datetime


@dataclass
class Graph3DLayout:
    """3D graph layout configuration"""

    layout_id: str
    layout_type: str  # force_directed, hierarchical, circular, spiral
    nodes: Dict[str, Node3D]
    edges: Dict[str, Edge3D]
    bounds: Tuple[
        Tuple[float, float], Tuple[float, float], Tuple[float, float]
    ]  # ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    center: Tuple[float, float, float]
    scale: float
    parameters: Dict[str, Any]
    created_at: datetime
    version: int


@dataclass
class VisualizationCluster:
    """3D cluster for grouping related nodes"""

    cluster_id: str
    center_position: Tuple[float, float, float]
    radius: float
    node_ids: List[str]
    cluster_type: str  # fraud_ring, merchant_group, geographic_cluster
    confidence_score: float
    properties: Dict[str, Any]
    created_at: datetime


@dataclass
class AnimationFrame:
    """Animation frame for real-time updates"""

    frame_id: str
    timestamp: datetime
    node_updates: Dict[str, Tuple[float, float, float]]  # node_id -> new_position
    edge_updates: Dict[str, Dict[str, Any]]  # edge_id -> properties
    cluster_updates: Dict[str, VisualizationCluster]
    duration_ms: int


class Graph3DManager:
    """
    Advanced 3D graph visualization manager
    Features: Spatial layouts, real-time updates, clustering, animation
    """

    def __init__(
        self, space_bounds: Tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    ):

        self.space_bounds = space_bounds  # (width, height, depth)

        # Graph storage
        self.layouts: Dict[str, Graph3DLayout] = {}
        self.active_layout_id: Optional[str] = None

        # Spatial indexing for performance
        self.spatial_index = {}
        self.grid_size = 100.0  # Grid cell size for spatial partitioning

        # Layout algorithms
        self.layout_algorithms = {
            "force_directed": self._force_directed_layout,
            "hierarchical": self._hierarchical_layout,
            "circular": self._circular_layout,
            "spiral": self._spiral_layout,
            "community_based": self._community_based_layout,
            "temporal": self._temporal_layout,
        }

        # Animation system
        self.animation_queue: deque = deque(maxlen=1000)
        self.is_animating = False
        self.animation_speed = 1.0

        # Clustering system
        self.clusters: Dict[str, VisualizationCluster] = {}
        self.cluster_algorithms = {
            "spatial": self._spatial_clustering,
            "graph_based": self._graph_based_clustering,
            "attribute_based": self._attribute_based_clustering,
        }

        # Real-time update system
        self.update_queue: deque = deque(maxlen=10000)
        self.update_batch_size = 100
        self.update_interval = 0.1  # seconds

        # Performance metrics
        self.metrics = {
            "layouts_created": 0,
            "nodes_positioned": 0,
            "edges_created": 0,
            "clusters_detected": 0,
            "updates_processed": 0,
            "frames_rendered": 0,
        }

        # Thread safety
        self.lock = threading.RLock()

    # =====================================================
    # LAYOUT ALGORITHMS
    # =====================================================

    def create_layout(
        self,
        layout_type: str,
        nodes_data: List[Dict[str, Any]],
        edges_data: List[Dict[str, Any]],
        parameters: Dict[str, Any] = None,
    ) -> str:
        """Create a new 3D graph layout"""

        layout_id = str(uuid.uuid4())

        if parameters is None:
            parameters = {}

        # Create nodes
        nodes = {}
        for node_data in nodes_data:
            node = Node3D(
                node_id=node_data["node_id"],
                node_type=node_data.get("node_type", "unknown"),
                position=(0.0, 0.0, 0.0),  # Will be set by layout algorithm
                size=node_data.get("size", 10.0),
                color=node_data.get("color", "#3498db"),
                label=node_data.get("label", node_data["node_id"]),
                properties=node_data.get("properties", {}),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            nodes[node["node_id"]] = node

        # Create edges
        edges = {}
        for edge_data in edges_data:
            edge = Edge3D(
                edge_id=edge_data.get(
                    "edge_id", f"{edge_data['source']}_{edge_data['target']}"
                ),
                source_node_id=edge_data["source"],
                target_node_id=edge_data["target"],
                edge_type=edge_data.get("edge_type", "connection"),
                weight=edge_data.get("weight", 1.0),
                color=edge_data.get("color", "#95a5a6"),
                properties=edge_data.get("properties", {}),
                created_at=datetime.now(timezone.utc),
            )
            edges[edge.edge_id] = edge

        # Apply layout algorithm
        if layout_type in self.layout_algorithms:
            positioned_nodes = self.layout_algorithms[layout_type](
                nodes, edges, parameters
            )
        else:
            positioned_nodes = self._random_layout(nodes, edges, parameters)

        # Calculate bounds
        bounds = self._calculate_bounds(positioned_nodes)
        center = self._calculate_center(positioned_nodes)

        # Create layout
        layout = Graph3DLayout(
            layout_id=layout_id,
            layout_type=layout_type,
            nodes=positioned_nodes,
            edges=edges,
            bounds=bounds,
            center=center,
            scale=1.0,
            parameters=parameters,
            created_at=datetime.now(timezone.utc),
            version=1,
        )

        with self.lock:
            self.layouts[layout_id] = layout
            self.active_layout_id = layout_id

        # Update spatial index
        self._update_spatial_index(layout_id)

        self.metrics["layouts_created"] += 1
        self.metrics["nodes_positioned"] += len(positioned_nodes)
        self.metrics["edges_created"] += len(edges)

        logger.info(
            f"Created 3D layout {layout_id} with {len(nodes)} nodes and {len(edges)} edges"
        )
        return layout_id

    def _force_directed_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Force-directed 3D layout algorithm"""

        # Parameters
        iterations = parameters.get("iterations", 100)
        k = parameters.get("spring_constant", 1.0)
        repulsion_strength = parameters.get("repulsion_strength", 100.0)
        damping = parameters.get("damping", 0.9)

        # Initialize random positions
        for node in nodes.values():
            x = np.random.uniform(-self.space_bounds[0] / 2, self.space_bounds[0] / 2)
            y = np.random.uniform(-self.space_bounds[1] / 2, self.space_bounds[1] / 2)
            z = np.random.uniform(-self.space_bounds[2] / 2, self.space_bounds[2] / 2)
            node.position = (x, y, z)

        # Build adjacency for faster edge lookup
        adjacency = defaultdict(list)
        edge_weights = {}
        for edge in edges.values():
            adjacency[edge.source_node_id].append(edge.target_node_id)
            adjacency[edge.target_node_id].append(edge.source_node_id)
            edge_weights[(edge.source_node_id, edge.target_node_id)] = edge.weight
            edge_weights[(edge.target_node_id, edge.source_node_id)] = edge.weight

        node_list = list(nodes.keys())
        positions = np.array([nodes[nid].position for nid in node_list])

        # Force-directed iterations
        for iteration in range(iterations):
            forces = np.zeros_like(positions)

            # Repulsive forces (all pairs)
            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    if i != j:
                        diff = positions[i] - positions[j]
                        distance = np.linalg.norm(diff)
                        if distance > 0:
                            force_magnitude = repulsion_strength / (distance**2)
                            force_direction = diff / distance
                            forces[i] += force_magnitude * force_direction

            # Attractive forces (connected nodes)
            for i, node_i in enumerate(node_list):
                for neighbor in adjacency[node_i]:
                    if neighbor in nodes:
                        j = node_list.index(neighbor)
                        diff = positions[j] - positions[i]
                        distance = np.linalg.norm(diff)
                        if distance > 0:
                            weight = edge_weights.get((node_i, neighbor), 1.0)
                            force_magnitude = k * weight * distance
                            force_direction = diff / distance
                            forces[i] += force_magnitude * force_direction

            # Apply forces with damping
            positions += forces * 0.01 * damping

            # Apply bounds
            positions = np.clip(
                positions,
                [
                    -self.space_bounds[0] / 2,
                    -self.space_bounds[1] / 2,
                    -self.space_bounds[2] / 2,
                ],
                [
                    self.space_bounds[0] / 2,
                    self.space_bounds[1] / 2,
                    self.space_bounds[2] / 2,
                ],
            )

        # Update node positions
        for i, node_id in enumerate(node_list):
            nodes[node_id].position = tuple(positions[i])
            nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _hierarchical_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Hierarchical 3D layout based on node types"""

        # Group nodes by type
        node_types = defaultdict(list)
        for node in nodes.values():
            node_types[node.node_type].append(node.node_id)

        # Define layer heights
        layer_height = self.space_bounds[2] / len(node_types)
        layer_radius = min(self.space_bounds[0], self.space_bounds[1]) / 3

        # Position nodes by type layers
        for layer_idx, (node_type, node_ids) in enumerate(node_types.items()):
            z = -self.space_bounds[2] / 2 + (layer_idx + 0.5) * layer_height

            # Circular arrangement within layer
            angle_step = 2 * math.pi / len(node_ids) if len(node_ids) > 1 else 0

            for i, node_id in enumerate(node_ids):
                angle = i * angle_step
                x = layer_radius * math.cos(angle)
                y = layer_radius * math.sin(angle)

                nodes[node_id].position = (x, y, z)
                nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _circular_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Circular 3D layout"""

        radius = parameters.get("radius", min(self.space_bounds) / 3)
        node_ids = list(nodes.keys())

        if len(node_ids) == 1:
            nodes[node_ids[0]].position = (0.0, 0.0, 0.0)
        else:
            angle_step = 2 * math.pi / len(node_ids)

            for i, node_id in enumerate(node_ids):
                angle = i * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0.0  # Flat circle by default

                nodes[node_id].position = (x, y, z)
                nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _spiral_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Spiral 3D layout"""

        spiral_height = parameters.get("height", self.space_bounds[2])
        spiral_radius = parameters.get(
            "radius", min(self.space_bounds[0], self.space_bounds[1]) / 3
        )
        turns = parameters.get("turns", 3)

        node_ids = list(nodes.keys())

        for i, node_id in enumerate(node_ids):
            t = i / len(node_ids) if len(node_ids) > 1 else 0
            angle = t * turns * 2 * math.pi

            x = spiral_radius * math.cos(angle)
            y = spiral_radius * math.sin(angle)
            z = -spiral_height / 2 + t * spiral_height

            nodes[node_id].position = (x, y, z)
            nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _community_based_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Community-based 3D layout"""

        # Detect communities (simplified)
        communities = self._detect_communities(nodes, edges)

        # Position communities in 3D space
        community_positions = self._position_communities(len(communities))

        # Position nodes within communities
        for community_idx, community_nodes in enumerate(communities):
            center = community_positions[community_idx]
            radius = parameters.get("community_radius", 50.0)

            # Circular arrangement within community
            if len(community_nodes) == 1:
                nodes[community_nodes[0]].position = center
            else:
                angle_step = 2 * math.pi / len(community_nodes)

                for i, node_id in enumerate(community_nodes):
                    angle = i * angle_step
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    z = center[2]

                    nodes[node_id].position = (x, y, z)
                    nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _temporal_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Temporal 3D layout based on timestamps"""

        # Extract timestamps from node properties
        node_times = {}
        for node in nodes.values():
            timestamp = node.properties.get("timestamp", node.created_at)
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            node_times[node.node_id] = timestamp

        # Sort by time
        sorted_nodes = sorted(node_times.items(), key=lambda x: x[1])

        # Position along time axis (z-axis)
        min_time = sorted_nodes[0][1]
        max_time = sorted_nodes[-1][1]
        time_range = (max_time - min_time).total_seconds()

        for i, (node_id, timestamp) in enumerate(sorted_nodes):
            # Z position based on time
            if time_range > 0:
                z = (
                    -self.space_bounds[2] / 2
                    + ((timestamp - min_time).total_seconds() / time_range)
                    * self.space_bounds[2]
                )
            else:
                z = 0.0

            # X, Y positions in spiral or circular pattern
            angle = i * 0.5  # Spiral parameter
            radius = 100.0
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)

            nodes[node_id].position = (x, y, z)
            nodes[node_id].updated_at = datetime.now(timezone.utc)

        return nodes

    def _random_layout(
        self,
        nodes: Dict[str, Node3D],
        edges: Dict[str, Edge3D],
        parameters: Dict[str, Any],
    ) -> Dict[str, Node3D]:
        """Random 3D layout"""

        for node in nodes.values():
            x = np.random.uniform(-self.space_bounds[0] / 2, self.space_bounds[0] / 2)
            y = np.random.uniform(-self.space_bounds[1] / 2, self.space_bounds[1] / 2)
            z = np.random.uniform(-self.space_bounds[2] / 2, self.space_bounds[2] / 2)

            node.position = (x, y, z)
            node.updated_at = datetime.now(timezone.utc)

        return nodes

    # =====================================================
    # CLUSTERING ALGORITHMS
    # =====================================================

    def detect_clusters(
        self,
        layout_id: str,
        algorithm: str = "spatial",
        parameters: Dict[str, Any] = None,
    ) -> List[VisualizationCluster]:
        """Detect clusters in 3D layout"""

        if layout_id not in self.layouts:
            logger.warning(f"Layout {layout_id} not found")
            return []

        layout = self.layouts[layout_id]

        if parameters is None:
            parameters = {}

        if algorithm in self.cluster_algorithms:
            clusters = self.cluster_algorithms[algorithm](layout, parameters)
        else:
            clusters = []

        # Store clusters
        for cluster in clusters:
            self.clusters[cluster.cluster_id] = cluster

        self.metrics["clusters_detected"] += len(clusters)
        logger.info(f"Detected {len(clusters)} clusters using {algorithm} algorithm")

        return clusters

    def _spatial_clustering(
        self, layout: Graph3DLayout, parameters: Dict[str, Any]
    ) -> List[VisualizationCluster]:
        """Spatial clustering based on 3D proximity"""

        distance_threshold = parameters.get("distance_threshold", 100.0)
        min_cluster_size = parameters.get("min_cluster_size", 3)

        clusters = []
        visited = set()

        node_positions = {nid: node.position for nid, node in layout.nodes.items()}

        for node_id, position in node_positions.items():
            if node_id in visited:
                continue

            # Find nearby nodes
            cluster_nodes = [node_id]
            visited.add(node_id)

            for other_id, other_position in node_positions.items():
                if other_id not in visited:
                    distance = euclidean(position, other_position)
                    if distance <= distance_threshold:
                        cluster_nodes.append(other_id)
                        visited.add(other_id)

            # Create cluster if large enough
            if len(cluster_nodes) >= min_cluster_size:
                # Calculate cluster center
                positions = [node_positions[nid] for nid in cluster_nodes]
                center = tuple(np.mean(positions, axis=0))

                # Calculate cluster radius
                distances = [euclidean(center, pos) for pos in positions]
                radius = max(distances)

                cluster = VisualizationCluster(
                    cluster_id=str(uuid.uuid4()),
                    center_position=center,
                    radius=radius,
                    node_ids=cluster_nodes,
                    cluster_type="spatial",
                    confidence_score=min(1.0, len(cluster_nodes) / 10.0),
                    properties={
                        "algorithm": "spatial",
                        "threshold": distance_threshold,
                    },
                    created_at=datetime.now(timezone.utc),
                )
                clusters.append(cluster)

        return clusters

    def _graph_based_clustering(
        self, layout: Graph3DLayout, parameters: Dict[str, Any]
    ) -> List[VisualizationCluster]:
        """Graph-based clustering using edge connectivity"""

        min_cluster_size = parameters.get("min_cluster_size", 3)

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in layout.edges.values():
            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)

        clusters = []
        visited = set()

        # DFS to find connected components
        for node_id in layout.nodes.keys():
            if node_id in visited:
                continue

            # DFS traversal
            stack = [node_id]
            component = []

            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)

                    for neighbor in adjacency[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

            # Create cluster if large enough
            if len(component) >= min_cluster_size:
                # Calculate cluster properties
                positions = [layout.nodes[nid].position for nid in component]
                center = tuple(np.mean(positions, axis=0))
                distances = [euclidean(center, pos) for pos in positions]
                radius = max(distances) if distances else 0.0

                cluster = VisualizationCluster(
                    cluster_id=str(uuid.uuid4()),
                    center_position=center,
                    radius=radius,
                    node_ids=component,
                    cluster_type="graph_component",
                    confidence_score=min(1.0, len(component) / 20.0),
                    properties={"algorithm": "graph_based"},
                    created_at=datetime.now(timezone.utc),
                )
                clusters.append(cluster)

        return clusters

    def _attribute_based_clustering(
        self, layout: Graph3DLayout, parameters: Dict[str, Any]
    ) -> List[VisualizationCluster]:
        """Attribute-based clustering using node properties"""

        cluster_attribute = parameters.get("attribute", "node_type")
        min_cluster_size = parameters.get("min_cluster_size", 2)

        # Group by attribute
        attribute_groups = defaultdict(list)
        for node in layout.nodes.values():
            if cluster_attribute == "node_type":
                attr_value = node.node_type
            else:
                attr_value = node.properties.get(cluster_attribute, "unknown")

            attribute_groups[attr_value].append(node.node_id)

        clusters = []

        for attr_value, node_ids in attribute_groups.items():
            if len(node_ids) >= min_cluster_size:
                # Calculate cluster properties
                positions = [layout.nodes[nid].position for nid in node_ids]
                center = tuple(np.mean(positions, axis=0))
                distances = [euclidean(center, pos) for pos in positions]
                radius = max(distances) if distances else 0.0

                cluster = VisualizationCluster(
                    cluster_id=str(uuid.uuid4()),
                    center_position=center,
                    radius=radius,
                    node_ids=node_ids,
                    cluster_type=f"attribute_{cluster_attribute}",
                    confidence_score=1.0,  # High confidence for attribute-based
                    properties={
                        "algorithm": "attribute_based",
                        "attribute": cluster_attribute,
                        "value": attr_value,
                    },
                    created_at=datetime.now(timezone.utc),
                )
                clusters.append(cluster)

        return clusters

    # =====================================================
    # REAL-TIME UPDATES
    # =====================================================

    def update_node_position(
        self,
        layout_id: str,
        node_id: str,
        new_position: Tuple[float, float, float],
        animate: bool = True,
    ):
        """Update node position with optional animation"""

        if layout_id not in self.layouts:
            logger.warning(f"Layout {layout_id} not found")
            return

        layout = self.layouts[layout_id]

        if node_id not in layout.nodes:
            logger.warning(f"Node {node_id} not found in layout")
            return

        old_position = layout.nodes[node_id].position

        if animate:
            # Create animation frame
            frame = AnimationFrame(
                frame_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                node_updates={node_id: new_position},
                edge_updates={},
                cluster_updates={},
                duration_ms=500,  # 500ms animation
            )
            self.animation_queue.append(frame)
        else:
            # Direct update
            layout.nodes[node_id].position = new_position
            layout.nodes[node_id].updated_at = datetime.now(timezone.utc)

        # Update spatial index
        self._update_node_spatial_index(layout_id, node_id, old_position, new_position)

        self.metrics["updates_processed"] += 1

    def update_multiple_nodes(
        self,
        layout_id: str,
        updates: Dict[str, Tuple[float, float, float]],
        animate: bool = True,
    ):
        """Update multiple nodes efficiently"""

        if layout_id not in self.layouts:
            logger.warning(f"Layout {layout_id} not found")
            return

        layout = self.layouts[layout_id]

        if animate:
            # Create batch animation frame
            frame = AnimationFrame(
                frame_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                node_updates=updates,
                edge_updates={},
                cluster_updates={},
                duration_ms=500,
            )
            self.animation_queue.append(frame)
        else:
            # Direct batch update
            for node_id, new_position in updates.items():
                if node_id in layout.nodes:
                    old_position = layout.nodes[node_id].position
                    layout.nodes[node_id].position = new_position
                    layout.nodes[node_id].updated_at = datetime.now(timezone.utc)

                    # Update spatial index
                    self._update_node_spatial_index(
                        layout_id, node_id, old_position, new_position
                    )

        self.metrics["updates_processed"] += len(updates)

    def add_node_realtime(
        self,
        layout_id: str,
        node_data: Dict[str, Any],
        position: Tuple[float, float, float] = None,
    ):
        """Add new node to existing layout"""

        if layout_id not in self.layouts:
            logger.warning(f"Layout {layout_id} not found")
            return

        layout = self.layouts[layout_id]

        # Generate position if not provided
        if position is None:
            position = self._find_optimal_position(layout)

        # Create node
        node = Node3D(
            node_id=node_data["node_id"],
            node_type=node_data.get("node_type", "unknown"),
            position=position,
            size=node_data.get("size", 10.0),
            color=node_data.get("color", "#3498db"),
            label=node_data.get("label", node_data["node_id"]),
            properties=node_data.get("properties", {}),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Add to layout
        layout.nodes[node.node_id] = node
        layout.version += 1

        # Update spatial index
        self._add_node_to_spatial_index(layout_id, node.node_id, position)

        logger.info(f"Added node {node.node_id} to layout {layout_id}")
        self.metrics["nodes_positioned"] += 1

    def remove_node_realtime(self, layout_id: str, node_id: str):
        """Remove node from existing layout"""

        if layout_id not in self.layouts:
            logger.warning(f"Layout {layout_id} not found")
            return

        layout = self.layouts[layout_id]

        if node_id not in layout.nodes:
            logger.warning(f"Node {node_id} not found in layout")
            return

        position = layout.nodes[node_id].position

        # Remove node
        del layout.nodes[node_id]

        # Remove related edges
        edges_to_remove = []
        for edge_id, edge in layout.edges.items():
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                edges_to_remove.append(edge_id)

        for edge_id in edges_to_remove:
            del layout.edges[edge_id]

        # Update spatial index
        self._remove_node_from_spatial_index(layout_id, node_id, position)

        layout.version += 1
        logger.info(f"Removed node {node_id} from layout {layout_id}")

    # =====================================================
    # SPATIAL INDEXING
    # =====================================================

    def _update_spatial_index(self, layout_id: str):
        """Update spatial index for layout"""

        if layout_id not in self.layouts:
            return

        layout = self.layouts[layout_id]
        self.spatial_index[layout_id] = defaultdict(list)

        for node_id, node in layout.nodes.items():
            grid_coords = self._get_grid_coordinates(node.position)
            self.spatial_index[layout_id][grid_coords].append(node_id)

    def _get_grid_coordinates(
        self, position: Tuple[float, float, float]
    ) -> Tuple[int, int, int]:
        """Get grid coordinates for spatial indexing"""
        x, y, z = position
        grid_x = int(x // self.grid_size)
        grid_y = int(y // self.grid_size)
        grid_z = int(z // self.grid_size)
        return (grid_x, grid_y, grid_z)

    def _update_node_spatial_index(
        self,
        layout_id: str,
        node_id: str,
        old_position: Tuple[float, float, float],
        new_position: Tuple[float, float, float],
    ):
        """Update spatial index for single node"""

        if layout_id not in self.spatial_index:
            return

        # Remove from old grid cell
        old_coords = self._get_grid_coordinates(old_position)
        if node_id in self.spatial_index[layout_id][old_coords]:
            self.spatial_index[layout_id][old_coords].remove(node_id)

        # Add to new grid cell
        new_coords = self._get_grid_coordinates(new_position)
        self.spatial_index[layout_id][new_coords].append(node_id)

    def _add_node_to_spatial_index(
        self, layout_id: str, node_id: str, position: Tuple[float, float, float]
    ):
        """Add node to spatial index"""

        if layout_id not in self.spatial_index:
            self.spatial_index[layout_id] = defaultdict(list)

        coords = self._get_grid_coordinates(position)
        self.spatial_index[layout_id][coords].append(node_id)

    def _remove_node_from_spatial_index(
        self, layout_id: str, node_id: str, position: Tuple[float, float, float]
    ):
        """Remove node from spatial index"""

        if layout_id not in self.spatial_index:
            return

        coords = self._get_grid_coordinates(position)
        if node_id in self.spatial_index[layout_id][coords]:
            self.spatial_index[layout_id][coords].remove(node_id)

    def find_nearby_nodes(
        self, layout_id: str, position: Tuple[float, float, float], radius: float
    ) -> List[str]:
        """Find nodes within radius using spatial index"""

        if layout_id not in self.spatial_index or layout_id not in self.layouts:
            return []

        layout = self.layouts[layout_id]
        nearby_nodes = []

        # Calculate grid range
        grid_radius = int(radius // self.grid_size) + 1
        center_coords = self._get_grid_coordinates(position)

        # Check surrounding grid cells
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-grid_radius, grid_radius + 1):
                    check_coords = (
                        center_coords[0] + dx,
                        center_coords[1] + dy,
                        center_coords[2] + dz,
                    )

                    # Check nodes in this grid cell
                    for node_id in self.spatial_index[layout_id][check_coords]:
                        if node_id in layout.nodes:
                            node_position = layout.nodes[node_id].position
                            distance = euclidean(position, node_position)
                            if distance <= radius:
                                nearby_nodes.append(node_id)

        return nearby_nodes

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    def _calculate_bounds(
        self, nodes: Dict[str, Node3D]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Calculate bounding box for nodes"""

        if not nodes:
            return ((0, 0), (0, 0), (0, 0))

        positions = [node.position for node in nodes.values()]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        z_coords = [pos[2] for pos in positions]

        return (
            (min(x_coords), max(x_coords)),
            (min(y_coords), max(y_coords)),
            (min(z_coords), max(z_coords)),
        )

    def _calculate_center(self, nodes: Dict[str, Node3D]) -> Tuple[float, float, float]:
        """Calculate center point of nodes"""

        if not nodes:
            return (0.0, 0.0, 0.0)

        positions = [node.position for node in nodes.values()]
        x_mean = np.mean([pos[0] for pos in positions])
        y_mean = np.mean([pos[1] for pos in positions])
        z_mean = np.mean([pos[2] for pos in positions])

        return (float(x_mean), float(y_mean), float(z_mean))

    def _detect_communities(
        self, nodes: Dict[str, Node3D], edges: Dict[str, Edge3D]
    ) -> List[List[str]]:
        """Simple community detection"""

        # Build adjacency list
        adjacency = defaultdict(set)
        for edge in edges.values():
            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)

        # Find connected components
        visited = set()
        communities = []

        for node_id in nodes.keys():
            if node_id not in visited:
                community = []
                stack = [node_id]

                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        community.append(current)

                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)

                communities.append(community)

        return communities

    def _position_communities(
        self, num_communities: int
    ) -> List[Tuple[float, float, float]]:
        """Position communities in 3D space"""

        positions = []

        if num_communities == 1:
            positions.append((0.0, 0.0, 0.0))
        elif num_communities <= 8:
            # Cube corners
            for i in range(min(num_communities, 8)):
                x = -200.0 if i & 1 else 200.0
                y = -200.0 if i & 2 else 200.0
                z = -200.0 if i & 4 else 200.0
                positions.append((x, y, z))
        else:
            # Spherical distribution
            for i in range(num_communities):
                phi = math.acos(1 - 2 * i / num_communities)
                theta = math.pi * (1 + 5**0.5) * i

                radius = 300.0
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)

                positions.append((x, y, z))

        return positions

    def _find_optimal_position(
        self, layout: Graph3DLayout
    ) -> Tuple[float, float, float]:
        """Find optimal position for new node"""

        if not layout.nodes:
            return (0.0, 0.0, 0.0)

        # Find least crowded area
        center = layout.center
        radius = 100.0

        # Try positions around center
        best_position = center
        max_distance = 0

        for _ in range(20):  # Try 20 random positions
            angle1 = np.random.uniform(0, 2 * math.pi)
            angle2 = np.random.uniform(0, math.pi)

            x = center[0] + radius * math.sin(angle2) * math.cos(angle1)
            y = center[1] + radius * math.sin(angle2) * math.sin(angle1)
            z = center[2] + radius * math.cos(angle2)

            position = (x, y, z)

            # Find minimum distance to existing nodes
            min_distance = float("inf")
            for node in layout.nodes.values():
                distance = euclidean(position, node.position)
                min_distance = min(min_distance, distance)

            if min_distance > max_distance:
                max_distance = min_distance
                best_position = position

        return best_position

    def get_layout(self, layout_id: str) -> Optional[Graph3DLayout]:
        """Get layout by ID"""
        return self.layouts.get(layout_id)

    def get_active_layout(self) -> Optional[Graph3DLayout]:
        """Get currently active layout"""
        if self.active_layout_id:
            return self.layouts.get(self.active_layout_id)
        return None

    def list_layouts(self) -> List[str]:
        """List all layout IDs"""
        return list(self.layouts.keys())

    def get_clusters(self, layout_id: str = None) -> List[VisualizationCluster]:
        """Get clusters for layout"""
        if layout_id:
            return [
                cluster
                for cluster in self.clusters.values()
                if any(
                    node_id in self.layouts[layout_id].nodes
                    for node_id in cluster.node_ids
                )
            ]
        return list(self.clusters.values())

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    def export_layout_data(self, layout_id: str) -> Optional[Dict[str, Any]]:
        """Export layout data for visualization"""

        if layout_id not in self.layouts:
            return None

        layout = self.layouts[layout_id]

        # Convert to serializable format
        export_data = {
            "layout_id": layout.layout_id,
            "layout_type": layout.layout_type,
            "bounds": layout.bounds,
            "center": layout.center,
            "scale": layout.scale,
            "version": layout.version,
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "position": node.position,
                    "size": node.size,
                    "color": node.color,
                    "label": node.label,
                    "properties": node.properties,
                }
                for node in layout.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "color": edge.color,
                    "properties": edge.properties,
                }
                for edge in layout.edges.values()
            ],
            "clusters": [
                {
                    "id": cluster.cluster_id,
                    "center": cluster.center_position,
                    "radius": cluster.radius,
                    "nodes": cluster.node_ids,
                    "type": cluster.cluster_type,
                    "confidence": cluster.confidence_score,
                    "properties": cluster.properties,
                }
                for cluster in self.get_clusters(layout_id)
            ],
        }

        return export_data


# =====================================================
# USAGE EXAMPLE
# =====================================================


def main():
    """Example usage of 3D graph manager"""

    # Initialize manager
    manager = Graph3DManager(space_bounds=(800, 600, 400))

    # Sample data
    nodes_data = [
        {"node_id": "user_1", "node_type": "user", "color": "#e74c3c", "size": 15},
        {"node_id": "user_2", "node_type": "user", "color": "#e74c3c", "size": 12},
        {
            "node_id": "merchant_1",
            "node_type": "merchant",
            "color": "#2ecc71",
            "size": 20,
        },
        {"node_id": "device_1", "node_type": "device", "color": "#f39c12", "size": 8},
    ]

    edges_data = [
        {
            "source": "user_1",
            "target": "merchant_1",
            "edge_type": "transaction",
            "weight": 2.0,
        },
        {
            "source": "user_2",
            "target": "merchant_1",
            "edge_type": "transaction",
            "weight": 1.5,
        },
        {
            "source": "user_1",
            "target": "device_1",
            "edge_type": "uses_device",
            "weight": 1.0,
        },
    ]

    # Create force-directed layout
    layout_id = manager.create_layout("force_directed", nodes_data, edges_data)
    print(f"Created layout: {layout_id}")

    # Detect spatial clusters
    clusters = manager.detect_clusters(
        layout_id, "spatial", {"distance_threshold": 150.0}
    )
    print(f"Detected {len(clusters)} clusters")

    # Update node position
    manager.update_node_position(layout_id, "user_1", (100.0, 50.0, 25.0))
    print("Updated node position")

    # Add new node
    manager.add_node_realtime(
        layout_id,
        {"node_id": "user_3", "node_type": "user", "color": "#e74c3c", "size": 14},
    )
    print("Added new node")

    # Export layout data
    export_data = manager.export_layout_data(layout_id)
    if export_data:
        print(f"Exported layout with {len(export_data['nodes'])} nodes")

    # Get metrics
    metrics = manager.get_metrics()
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
