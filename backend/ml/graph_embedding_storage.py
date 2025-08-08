#!/usr/bin/env python3
"""
Graph Neural Network (GNN) Support Infrastructure
Graph embedding storage, sampling algorithms, and feature extraction
"""

import asyncio
import base64
import hashlib
import json
import logging
import pickle
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
import redis.asyncio as redis
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

logger = logging.getLogger(__name__)


@dataclass
class NodeEmbedding:
    """Node embedding representation"""

    node_id: str
    node_type: str  # user, merchant, device, location
    embedding_vector: np.ndarray
    embedding_dimension: int
    model_version: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EdgeEmbedding:
    """Edge embedding representation"""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str  # transaction, device_used, location_visited
    embedding_vector: np.ndarray
    embedding_dimension: int
    model_version: str
    weight: float
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphSample:
    """Graph sample for GNN training"""

    sample_id: str
    center_node_id: str
    subgraph_nodes: List[str]
    subgraph_edges: List[Tuple[str, str, str]]  # (source, target, edge_type)
    node_features: Dict[str, np.ndarray]
    edge_features: Dict[str, np.ndarray]
    labels: Dict[str, float]  # fraud labels for nodes
    sample_strategy: str
    hop_count: int
    created_at: datetime


@dataclass
class GNNFeatureVector:
    """GNN-extracted feature vector"""

    node_id: str
    feature_vector: np.ndarray
    feature_names: List[str]
    model_version: str
    extraction_method: str
    confidence_score: float
    created_at: datetime


class GraphEmbeddingStorage:
    """
    Vector storage system for graph embeddings with PostgreSQL and Redis
    Supports high-dimensional vectors, similarity search, and versioning
    """

    def __init__(
        self,
        postgres_config: Dict[str, str],
        redis_config: Dict[str, str],
        embedding_dimension: int = 128,
    ):

        self.postgres_config = postgres_config
        self.redis_config = redis_config
        self.embedding_dimension = embedding_dimension

        # Database connections
        self.pg_conn = None
        self.redis_client = None

        # Vector storage configuration
        self.batch_size = 1000
        self.index_type = "ivfflat"  # For pgvector

        # Caching configuration
        self.cache_ttl = 3600  # 1 hour
        self.cache_prefix = "graph_embedding:"

        # Metrics
        self.metrics = {
            "embeddings_stored": 0,
            "embeddings_retrieved": 0,
            "similarity_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def initialize(self):
        """Initialize database connections and vector storage"""
        try:
            # PostgreSQL connection with pgvector support
            self.pg_conn = psycopg2.connect(**self.postgres_config)

            # Redis connection
            self.redis_client = redis.Redis(**self.redis_config)
            await self.redis_client.ping()

            # Create vector storage tables
            await self._create_embedding_tables()

            logger.info("Graph embedding storage initialized")

        except Exception as e:
            logger.error(f"Failed to initialize embedding storage: {e}")
            raise

    async def _create_embedding_tables(self):
        """Create tables for storing embeddings with vector support"""

        cursor = self.pg_conn.cursor()

        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Node embeddings table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS node_embeddings (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(255) NOT NULL,
                    node_type VARCHAR(50) NOT NULL,
                    embedding_vector vector({self.embedding_dimension}) NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(node_id, model_version)
                );
            """
            )

            # Edge embeddings table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS edge_embeddings (
                    id SERIAL PRIMARY KEY,
                    edge_id VARCHAR(255) NOT NULL,
                    source_node_id VARCHAR(255) NOT NULL,
                    target_node_id VARCHAR(255) NOT NULL,
                    edge_type VARCHAR(50) NOT NULL,
                    embedding_vector vector({self.embedding_dimension}) NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    weight FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB,
                    UNIQUE(edge_id, model_version)
                );
            """
            )

            # Graph samples table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_samples (
                    id SERIAL PRIMARY KEY,
                    sample_id VARCHAR(255) NOT NULL UNIQUE,
                    center_node_id VARCHAR(255) NOT NULL,
                    subgraph_data JSONB NOT NULL,
                    sample_strategy VARCHAR(50) NOT NULL,
                    hop_count INTEGER NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            )

            # GNN features table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS gnn_features (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(255) NOT NULL,
                    feature_vector vector({self.embedding_dimension}) NOT NULL,
                    feature_names JSONB NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    extraction_method VARCHAR(50) NOT NULL,
                    confidence_score FLOAT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(node_id, model_version, extraction_method)
                );
            """
            )

            # Create indexes for vector similarity search
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS node_embeddings_vector_idx 
                ON node_embeddings USING ivfflat (embedding_vector vector_cosine_ops);
            """
            )

            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS edge_embeddings_vector_idx 
                ON edge_embeddings USING ivfflat (embedding_vector vector_cosine_ops);
            """
            )

            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS gnn_features_vector_idx 
                ON gnn_features USING ivfflat (feature_vector vector_cosine_ops);
            """
            )

            # Create other useful indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS node_embeddings_node_id_idx ON node_embeddings(node_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS node_embeddings_type_idx ON node_embeddings(node_type);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS edge_embeddings_source_idx ON edge_embeddings(source_node_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS edge_embeddings_target_idx ON edge_embeddings(target_node_id);"
            )

            self.pg_conn.commit()
            logger.info("Embedding storage tables created")

        except Exception as e:
            self.pg_conn.rollback()
            logger.error(f"Failed to create embedding tables: {e}")
            raise
        finally:
            cursor.close()

    # =====================================================
    # NODE EMBEDDING OPERATIONS
    # =====================================================

    async def store_node_embedding(self, embedding: NodeEmbedding) -> bool:
        """Store a single node embedding"""
        try:
            cursor = self.pg_conn.cursor()

            # Convert numpy array to list for PostgreSQL
            vector_list = embedding.embedding_vector.tolist()

            cursor.execute(
                """
                INSERT INTO node_embeddings 
                (node_id, node_type, embedding_vector, embedding_dimension, model_version, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id, model_version) 
                DO UPDATE SET 
                    embedding_vector = EXCLUDED.embedding_vector,
                    created_at = NOW(),
                    metadata = EXCLUDED.metadata;
            """,
                (
                    embedding.node_id,
                    embedding.node_type,
                    vector_list,
                    embedding.embedding_dimension,
                    embedding.model_version,
                    json.dumps(embedding.metadata) if embedding.metadata else None,
                ),
            )

            self.pg_conn.commit()
            cursor.close()

            # Cache in Redis
            await self._cache_node_embedding(embedding)

            self.metrics["embeddings_stored"] += 1
            logger.debug(f"Stored node embedding for {embedding.node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store node embedding {embedding.node_id}: {e}")
            self.pg_conn.rollback()
            return False

    async def bulk_store_node_embeddings(self, embeddings: List[NodeEmbedding]) -> int:
        """Bulk store multiple node embeddings"""
        try:
            cursor = self.pg_conn.cursor()

            # Prepare batch data
            batch_data = []
            for embedding in embeddings:
                vector_list = embedding.embedding_vector.tolist()
                batch_data.append(
                    (
                        embedding.node_id,
                        embedding.node_type,
                        vector_list,
                        embedding.embedding_dimension,
                        embedding.model_version,
                        json.dumps(embedding.metadata) if embedding.metadata else None,
                    )
                )

            # Execute batch insert
            cursor.executemany(
                """
                INSERT INTO node_embeddings 
                (node_id, node_type, embedding_vector, embedding_dimension, model_version, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id, model_version) 
                DO UPDATE SET 
                    embedding_vector = EXCLUDED.embedding_vector,
                    created_at = NOW(),
                    metadata = EXCLUDED.metadata;
            """,
                batch_data,
            )

            self.pg_conn.commit()
            cursor.close()

            # Cache in Redis (sample for performance)
            if len(embeddings) <= 100:
                for embedding in embeddings:
                    await self._cache_node_embedding(embedding)

            self.metrics["embeddings_stored"] += len(embeddings)
            logger.info(f"Bulk stored {len(embeddings)} node embeddings")
            return len(embeddings)

        except Exception as e:
            logger.error(f"Failed to bulk store node embeddings: {e}")
            self.pg_conn.rollback()
            return 0

    async def get_node_embedding(
        self, node_id: str, model_version: str
    ) -> Optional[NodeEmbedding]:
        """Retrieve a node embedding"""
        try:
            # Try cache first
            cached = await self._get_cached_node_embedding(node_id, model_version)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached

            self.metrics["cache_misses"] += 1

            # Query database
            cursor = self.pg_conn.cursor()
            cursor.execute(
                """
                SELECT node_id, node_type, embedding_vector, embedding_dimension, 
                       model_version, created_at, metadata
                FROM node_embeddings 
                WHERE node_id = %s AND model_version = %s;
            """,
                (node_id, model_version),
            )

            row = cursor.fetchone()
            cursor.close()

            if row:
                embedding = NodeEmbedding(
                    node_id=row[0],
                    node_type=row[1],
                    embedding_vector=np.array(row[2]),
                    embedding_dimension=row[3],
                    model_version=row[4],
                    created_at=row[5],
                    metadata=json.loads(row[6]) if row[6] else None,
                )

                # Cache result
                await self._cache_node_embedding(embedding)

                self.metrics["embeddings_retrieved"] += 1
                return embedding

            return None

        except Exception as e:
            logger.error(f"Failed to get node embedding {node_id}: {e}")
            return None

    async def find_similar_nodes(
        self,
        query_vector: np.ndarray,
        node_type: str = None,
        model_version: str = None,
        limit: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find similar nodes using vector similarity search"""
        try:
            cursor = self.pg_conn.cursor()

            # Build query conditions
            conditions = []
            params = [query_vector.tolist(), limit]

            if node_type:
                conditions.append("node_type = %s")
                params.insert(-1, node_type)

            if model_version:
                conditions.append("model_version = %s")
                params.insert(-1, model_version)

            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

            # Cosine similarity search
            cursor.execute(
                f"""
                SELECT node_id, (1 - (embedding_vector <=> %s)) as similarity
                FROM node_embeddings 
                {where_clause}
                ORDER BY embedding_vector <=> %s
                LIMIT %s;
            """,
                params,
            )

            results = cursor.fetchall()
            cursor.close()

            self.metrics["similarity_searches"] += 1
            return [(node_id, float(similarity)) for node_id, similarity in results]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    # =====================================================
    # EDGE EMBEDDING OPERATIONS
    # =====================================================

    async def store_edge_embedding(self, embedding: EdgeEmbedding) -> bool:
        """Store a single edge embedding"""
        try:
            cursor = self.pg_conn.cursor()

            vector_list = embedding.embedding_vector.tolist()

            cursor.execute(
                """
                INSERT INTO edge_embeddings 
                (edge_id, source_node_id, target_node_id, edge_type, embedding_vector, 
                 embedding_dimension, model_version, weight, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (edge_id, model_version) 
                DO UPDATE SET 
                    embedding_vector = EXCLUDED.embedding_vector,
                    weight = EXCLUDED.weight,
                    created_at = NOW(),
                    metadata = EXCLUDED.metadata;
            """,
                (
                    embedding.edge_id,
                    embedding.source_node_id,
                    embedding.target_node_id,
                    embedding.edge_type,
                    vector_list,
                    embedding.embedding_dimension,
                    embedding.model_version,
                    embedding.weight,
                    json.dumps(embedding.metadata) if embedding.metadata else None,
                ),
            )

            self.pg_conn.commit()
            cursor.close()

            self.metrics["embeddings_stored"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to store edge embedding {embedding.edge_id}: {e}")
            self.pg_conn.rollback()
            return False

    # =====================================================
    # GRAPH SAMPLING ALGORITHMS
    # =====================================================

    async def random_walk_sampling(
        self, start_node_id: str, walk_length: int = 10, num_walks: int = 100
    ) -> List[List[str]]:
        """Perform random walk sampling from a start node"""
        try:
            # Get graph structure from database
            graph = await self._build_networkx_graph()

            if start_node_id not in graph:
                logger.warning(f"Start node {start_node_id} not found in graph")
                return []

            walks = []
            for _ in range(num_walks):
                walk = [start_node_id]
                current_node = start_node_id

                for step in range(walk_length - 1):
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break

                    # Weighted random selection
                    weights = [
                        graph[current_node][neighbor].get("weight", 1.0)
                        for neighbor in neighbors
                    ]
                    total_weight = sum(weights)

                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        next_node = np.random.choice(neighbors, p=weights)
                        walk.append(next_node)
                        current_node = next_node
                    else:
                        break

                walks.append(walk)

            logger.info(f"Generated {len(walks)} random walks from {start_node_id}")
            return walks

        except Exception as e:
            logger.error(f"Random walk sampling failed: {e}")
            return []

    async def neighborhood_sampling(
        self, center_node_id: str, num_hops: int = 2, max_nodes_per_hop: int = 10
    ) -> GraphSample:
        """Sample neighborhood around a center node"""
        try:
            # Get graph structure
            graph = await self._build_networkx_graph()

            if center_node_id not in graph:
                logger.warning(f"Center node {center_node_id} not found in graph")
                return None

            # BFS sampling with hop limits
            sampled_nodes = {center_node_id}
            current_level = {center_node_id}
            sampled_edges = []

            for hop in range(num_hops):
                next_level = set()

                for node in current_level:
                    neighbors = list(graph.neighbors(node))

                    # Limit neighbors per hop
                    if len(neighbors) > max_nodes_per_hop:
                        # Prioritize by edge weight
                        neighbor_weights = [
                            (n, graph[node][n].get("weight", 1.0)) for n in neighbors
                        ]
                        neighbor_weights.sort(key=lambda x: x[1], reverse=True)
                        neighbors = [n for n, w in neighbor_weights[:max_nodes_per_hop]]

                    for neighbor in neighbors:
                        if neighbor not in sampled_nodes:
                            next_level.add(neighbor)
                            sampled_nodes.add(neighbor)

                        # Add edge
                        edge_data = graph[node][neighbor]
                        edge_type = edge_data.get("type", "unknown")
                        sampled_edges.append((node, neighbor, edge_type))

                current_level = next_level
                if not current_level:
                    break

            # Extract node and edge features
            node_features = await self._extract_node_features(list(sampled_nodes))
            edge_features = await self._extract_edge_features(sampled_edges)

            # Get fraud labels for nodes
            labels = await self._get_fraud_labels(list(sampled_nodes))

            sample = GraphSample(
                sample_id=str(uuid.uuid4()),
                center_node_id=center_node_id,
                subgraph_nodes=list(sampled_nodes),
                subgraph_edges=sampled_edges,
                node_features=node_features,
                edge_features=edge_features,
                labels=labels,
                sample_strategy="neighborhood",
                hop_count=num_hops,
                created_at=datetime.now(timezone.utc),
            )

            # Store sample
            await self._store_graph_sample(sample)

            logger.info(
                f"Created neighborhood sample: {len(sampled_nodes)} nodes, {len(sampled_edges)} edges"
            )
            return sample

        except Exception as e:
            logger.error(f"Neighborhood sampling failed: {e}")
            return None

    async def stratified_sampling(
        self, node_types: List[str], samples_per_type: int = 100
    ) -> List[GraphSample]:
        """Stratified sampling across different node types"""
        try:
            samples = []

            for node_type in node_types:
                # Get nodes of this type
                cursor = self.pg_conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT node_id FROM node_embeddings 
                    WHERE node_type = %s 
                    ORDER BY RANDOM() 
                    LIMIT %s;
                """,
                    (node_type, samples_per_type),
                )

                nodes = [row[0] for row in cursor.fetchall()]
                cursor.close()

                # Create neighborhood samples for each node
                for node_id in nodes:
                    sample = await self.neighborhood_sampling(node_id, num_hops=1)
                    if sample:
                        samples.append(sample)

            logger.info(f"Created {len(samples)} stratified samples")
            return samples

        except Exception as e:
            logger.error(f"Stratified sampling failed: {e}")
            return []

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    async def extract_graph_features(
        self, node_id: str, feature_methods: List[str] = None
    ) -> GNNFeatureVector:
        """Extract graph-based features for a node"""
        try:
            if feature_methods is None:
                feature_methods = ["degree", "clustering", "pagerank", "betweenness"]

            # Build graph
            graph = await self._build_networkx_graph()

            if node_id not in graph:
                logger.warning(f"Node {node_id} not found in graph")
                return None

            features = {}
            feature_names = []

            # Degree-based features
            if "degree" in feature_methods:
                degree = graph.degree(node_id)
                in_degree = graph.in_degree(node_id) if graph.is_directed() else degree
                out_degree = (
                    graph.out_degree(node_id) if graph.is_directed() else degree
                )

                features.update(
                    {"degree": degree, "in_degree": in_degree, "out_degree": out_degree}
                )
                feature_names.extend(["degree", "in_degree", "out_degree"])

            # Clustering coefficient
            if "clustering" in feature_methods:
                clustering = nx.clustering(graph, node_id)
                features["clustering_coefficient"] = clustering
                feature_names.append("clustering_coefficient")

            # PageRank
            if "pagerank" in feature_methods:
                pagerank = nx.pagerank(graph)
                features["pagerank"] = pagerank.get(node_id, 0.0)
                feature_names.append("pagerank")

            # Betweenness centrality (expensive for large graphs)
            if "betweenness" in feature_methods and len(graph) < 10000:
                betweenness = nx.betweenness_centrality(graph)
                features["betweenness_centrality"] = betweenness.get(node_id, 0.0)
                feature_names.append("betweenness_centrality")

            # Local features
            if "local" in feature_methods:
                neighbors = list(graph.neighbors(node_id))
                neighbor_degrees = [graph.degree(n) for n in neighbors]

                features.update(
                    {
                        "avg_neighbor_degree": (
                            np.mean(neighbor_degrees) if neighbor_degrees else 0
                        ),
                        "max_neighbor_degree": (
                            max(neighbor_degrees) if neighbor_degrees else 0
                        ),
                        "min_neighbor_degree": (
                            min(neighbor_degrees) if neighbor_degrees else 0
                        ),
                    }
                )
                feature_names.extend(
                    [
                        "avg_neighbor_degree",
                        "max_neighbor_degree",
                        "min_neighbor_degree",
                    ]
                )

            # Convert to numpy array
            feature_vector = np.array([features[name] for name in feature_names])

            # Normalize features
            if len(feature_vector) > 0:
                scaler = StandardScaler()
                feature_vector = scaler.fit_transform(
                    feature_vector.reshape(-1, 1)
                ).flatten()

            gnn_features = GNNFeatureVector(
                node_id=node_id,
                feature_vector=feature_vector,
                feature_names=feature_names,
                model_version="graph_features_v1",
                extraction_method="networkx",
                confidence_score=1.0,  # Rule-based features have high confidence
                created_at=datetime.now(timezone.utc),
            )

            # Store features
            await self._store_gnn_features(gnn_features)

            return gnn_features

        except Exception as e:
            logger.error(f"Feature extraction failed for {node_id}: {e}")
            return None

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    async def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph from database"""
        try:
            cursor = self.pg_conn.cursor()

            # Get all edges
            cursor.execute(
                """
                SELECT source_node_id, target_node_id, edge_type, weight
                FROM edge_embeddings;
            """
            )

            edges = cursor.fetchall()
            cursor.close()

            # Create graph
            graph = nx.Graph()  # Use DiGraph() for directed graphs

            for source, target, edge_type, weight in edges:
                graph.add_edge(source, target, type=edge_type, weight=weight)

            return graph

        except Exception as e:
            logger.error(f"Failed to build NetworkX graph: {e}")
            return nx.Graph()

    async def _extract_node_features(
        self, node_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract features for nodes"""
        features = {}
        for node_id in node_ids:
            # Simple node features (can be enhanced)
            features[node_id] = np.random.random(10)  # Placeholder
        return features

    async def _extract_edge_features(
        self, edges: List[Tuple[str, str, str]]
    ) -> Dict[str, np.ndarray]:
        """Extract features for edges"""
        features = {}
        for i, (source, target, edge_type) in enumerate(edges):
            edge_key = f"{source}_{target}_{edge_type}"
            features[edge_key] = np.random.random(5)  # Placeholder
        return features

    async def _get_fraud_labels(self, node_ids: List[str]) -> Dict[str, float]:
        """Get fraud labels for nodes"""
        # Placeholder - would query actual fraud labels
        return {node_id: np.random.random() for node_id in node_ids}

    async def _cache_node_embedding(self, embedding: NodeEmbedding):
        """Cache node embedding in Redis"""
        try:
            cache_key = (
                f"{self.cache_prefix}node:{embedding.node_id}:{embedding.model_version}"
            )
            embedding_data = {
                "vector": embedding.embedding_vector.tolist(),
                "type": embedding.node_type,
                "metadata": embedding.metadata,
            }

            await self.redis_client.setex(
                cache_key, self.cache_ttl, json.dumps(embedding_data)
            )

        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    async def _get_cached_node_embedding(
        self, node_id: str, model_version: str
    ) -> Optional[NodeEmbedding]:
        """Retrieve cached node embedding"""
        try:
            cache_key = f"{self.cache_prefix}node:{node_id}:{model_version}"
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data)
                return NodeEmbedding(
                    node_id=node_id,
                    node_type=data["type"],
                    embedding_vector=np.array(data["vector"]),
                    embedding_dimension=len(data["vector"]),
                    model_version=model_version,
                    created_at=datetime.now(timezone.utc),
                    metadata=data.get("metadata"),
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
            return None

    async def _store_graph_sample(self, sample: GraphSample):
        """Store graph sample in database"""
        try:
            cursor = self.pg_conn.cursor()

            subgraph_data = {
                "nodes": sample.subgraph_nodes,
                "edges": sample.subgraph_edges,
                "node_features": {
                    k: v.tolist() for k, v in sample.node_features.items()
                },
                "edge_features": {
                    k: v.tolist() for k, v in sample.edge_features.items()
                },
                "labels": sample.labels,
            }

            cursor.execute(
                """
                INSERT INTO graph_samples 
                (sample_id, center_node_id, subgraph_data, sample_strategy, hop_count)
                VALUES (%s, %s, %s, %s, %s);
            """,
                (
                    sample.sample_id,
                    sample.center_node_id,
                    json.dumps(subgraph_data),
                    sample.sample_strategy,
                    sample.hop_count,
                ),
            )

            self.pg_conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to store graph sample: {e}")
            self.pg_conn.rollback()

    async def _store_gnn_features(self, features: GNNFeatureVector):
        """Store GNN features in database"""
        try:
            cursor = self.pg_conn.cursor()

            cursor.execute(
                """
                INSERT INTO gnn_features 
                (node_id, feature_vector, feature_names, model_version, 
                 extraction_method, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (node_id, model_version, extraction_method)
                DO UPDATE SET 
                    feature_vector = EXCLUDED.feature_vector,
                    feature_names = EXCLUDED.feature_names,
                    confidence_score = EXCLUDED.confidence_score,
                    created_at = NOW();
            """,
                (
                    features.node_id,
                    features.feature_vector.tolist(),
                    json.dumps(features.feature_names),
                    features.model_version,
                    features.extraction_method,
                    features.confidence_score,
                ),
            )

            self.pg_conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to store GNN features: {e}")
            self.pg_conn.rollback()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    async def close(self):
        """Close database connections"""
        try:
            if self.pg_conn:
                self.pg_conn.close()

            if self.redis_client:
                await self.redis_client.close()

            logger.info("Graph embedding storage connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage of graph embedding storage"""

    # Configuration
    postgres_config = {
        "host": "localhost",
        "port": 5432,
        "database": "fraud_detection",
        "user": "postgres",
        "password": "password",
    }

    redis_config = {"host": "localhost", "port": 6379, "password": None}

    # Initialize storage
    storage = GraphEmbeddingStorage(postgres_config, redis_config)
    await storage.initialize()

    try:
        # Create sample node embedding
        embedding = NodeEmbedding(
            node_id="user_12345",
            node_type="user",
            embedding_vector=np.random.random(128),
            embedding_dimension=128,
            model_version="node2vec_v1",
            created_at=datetime.now(timezone.utc),
            metadata={"source": "transaction_graph"},
        )

        # Store embedding
        success = await storage.store_node_embedding(embedding)
        print(f"Embedding stored: {success}")

        # Retrieve embedding
        retrieved = await storage.get_node_embedding("user_12345", "node2vec_v1")
        print(f"Embedding retrieved: {retrieved is not None}")

        # Find similar nodes
        similar = await storage.find_similar_nodes(embedding.embedding_vector, "user")
        print(f"Found {len(similar)} similar nodes")

        # Perform neighborhood sampling
        sample = await storage.neighborhood_sampling("user_12345", num_hops=2)
        if sample:
            print(f"Created sample with {len(sample.subgraph_nodes)} nodes")

        # Extract graph features
        features = await storage.extract_graph_features("user_12345")
        if features:
            print(f"Extracted {len(features.feature_names)} features")

    finally:
        await storage.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
