#!/usr/bin/env python3
"""
Neo4j Graph Database Manager for Fraud Detection
Handles graph operations, fraud ring detection, and relationship analysis
"""

from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Represents a node in the fraud detection graph"""
    node_id: str
    node_type: str  # 'User', 'Merchant', 'Transaction', 'Device', 'Location', 'IPAddress'
    properties: Dict[str, Any]
    labels: List[str]

@dataclass
class GraphRelationship:
    """Represents a relationship in the fraud detection graph"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    strength: float
    created_at: datetime

@dataclass
class FraudRing:
    """Represents a detected fraud ring"""
    ring_id: str
    members: List[GraphNode]
    relationships: List[GraphRelationship]
    risk_score: float
    detection_algorithm: str
    confidence: float
    detected_at: datetime

class Neo4jManager:
    """Neo4j database manager for fraud detection graph operations"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        """Initialize Neo4j connection"""
        self.uri = uri or os.getenv('NEO4J_URL', 'bolt://localhost:7687')
        self.username = username or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'FraudGraph2024!')
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.username, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            logger.info(f"Connected to Neo4j at {self.uri}")
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("Neo4j connection test successful")
                    
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =====================================================
    # NODE OPERATIONS
    # =====================================================
    
    def create_user_node(self, user_data: Dict[str, Any]) -> bool:
        """Create or update a user node"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (u:User {user_id: $user_id})
                SET u += $properties
                SET u.updated_at = datetime()
                RETURN u.user_id as user_id
                """
                
                result = session.run(query, {
                    'user_id': user_data['user_id'],
                    'properties': user_data
                })
                
                user_id = result.single()["user_id"]
                logger.info(f"Created/updated user node: {user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create user node: {e}")
            return False
    
    def create_merchant_node(self, merchant_data: Dict[str, Any]) -> bool:
        """Create or update a merchant node"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (m:Merchant {merchant_id: $merchant_id})
                SET m += $properties
                SET m.updated_at = datetime()
                RETURN m.merchant_id as merchant_id
                """
                
                result = session.run(query, {
                    'merchant_id': merchant_data['merchant_id'],
                    'properties': merchant_data
                })
                
                merchant_id = result.single()["merchant_id"]
                logger.info(f"Created/updated merchant node: {merchant_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create merchant node: {e}")
            return False
    
    def create_transaction_node(self, transaction_data: Dict[str, Any]) -> bool:
        """Create a transaction node with relationships"""
        try:
            with self.driver.session() as session:
                # Create transaction node
                query = """
                CREATE (t:Transaction {
                    transaction_id: $transaction_id,
                    amount: $amount,
                    currency: $currency,
                    timestamp: datetime($timestamp),
                    fraud_score: $fraud_score,
                    risk_level: $risk_level,
                    decision: $decision,
                    processing_time_ms: $processing_time_ms
                })
                
                // Create relationships
                WITH t
                MATCH (u:User {user_id: $user_id})
                CREATE (u)-[:MADE_TRANSACTION {
                    timestamp: t.timestamp,
                    amount: t.amount
                }]->(t)
                
                WITH t
                MATCH (m:Merchant {merchant_id: $merchant_id})
                CREATE (t)-[:PAID_TO {
                    amount: t.amount,
                    timestamp: t.timestamp
                }]->(m)
                
                RETURN t.transaction_id as transaction_id
                """
                
                result = session.run(query, transaction_data)
                transaction_id = result.single()["transaction_id"]
                logger.info(f"Created transaction node with relationships: {transaction_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create transaction node: {e}")
            return False
    
    def create_device_relationship(self, user_id: str, device_data: Dict[str, Any]) -> bool:
        """Create device node and relationship with user"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (d:Device {device_id: $device_id})
                SET d += $device_properties
                SET d.updated_at = datetime()
                
                WITH d
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[r:USED_DEVICE]->(d)
                SET r.last_used = datetime(),
                    r.frequency = COALESCE(r.frequency, 0) + 1
                
                RETURN d.device_id as device_id
                """
                
                result = session.run(query, {
                    'user_id': user_id,
                    'device_id': device_data['device_id'],
                    'device_properties': device_data
                })
                
                device_id = result.single()["device_id"]
                logger.info(f"Created/updated device relationship: {user_id} -> {device_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create device relationship: {e}")
            return False
    
    # =====================================================
    # FRAUD RING DETECTION
    # =====================================================
    
    def detect_device_sharing_rings(self, min_users: int = 2, min_risk_score: float = 0.5) -> List[FraudRing]:
        """Detect fraud rings based on device sharing"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (u1:User)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(u2:User)
                WHERE u1.user_id <> u2.user_id 
                  AND u1.risk_score >= $min_risk_score 
                  AND u2.risk_score >= $min_risk_score
                
                WITH d, COLLECT(DISTINCT u1) + COLLECT(DISTINCT u2) as users
                WHERE SIZE(users) >= $min_users
                
                RETURN d.device_id as device_id,
                       users,
                       AVG([u IN users | u.risk_score]) as avg_risk_score,
                       SIZE(users) as user_count
                ORDER BY avg_risk_score DESC, user_count DESC
                LIMIT 50
                """
                
                result = session.run(query, {
                    'min_users': min_users,
                    'min_risk_score': min_risk_score
                })
                
                fraud_rings = []
                for record in result:
                    # Create fraud ring object
                    ring = FraudRing(
                        ring_id=f"device_sharing_{record['device_id']}",
                        members=[GraphNode(
                            node_id=user['user_id'],
                            node_type='User',
                            properties=dict(user),
                            labels=['User']
                        ) for user in record['users']],
                        relationships=[],  # Would need additional query to get relationships
                        risk_score=record['avg_risk_score'],
                        detection_algorithm='device_sharing',
                        confidence=min(record['avg_risk_score'] * record['user_count'] / 10, 1.0),
                        detected_at=datetime.now()
                    )
                    fraud_rings.append(ring)
                
                logger.info(f"Detected {len(fraud_rings)} device sharing fraud rings")
                return fraud_rings
                
        except Exception as e:
            logger.error(f"Failed to detect device sharing rings: {e}")
            return []
    
    def detect_velocity_rings(self, time_window_minutes: int = 60, min_transactions: int = 5) -> List[FraudRing]:
        """Detect fraud rings based on transaction velocity"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (u:User)-[:MADE_TRANSACTION]->(t:Transaction)-[:PAID_TO]->(m:Merchant)
                WHERE t.timestamp > datetime() - duration({minutes: $time_window})
                
                WITH u, m, COUNT(t) as tx_count, AVG(t.fraud_score) as avg_fraud_score
                WHERE tx_count >= $min_transactions AND avg_fraud_score > 0.6
                
                // Find users who transacted with the same high-risk merchants
                MATCH (u2:User)-[:MADE_TRANSACTION]->(t2:Transaction)-[:PAID_TO]->(m)
                WHERE u2.user_id <> u.user_id 
                  AND t2.timestamp > datetime() - duration({minutes: $time_window})
                
                WITH m, COLLECT(DISTINCT u) + COLLECT(DISTINCT u2) as users, 
                     AVG(avg_fraud_score) as merchant_avg_fraud_score
                WHERE SIZE(users) >= 2
                
                RETURN m.merchant_id as merchant_id,
                       m.business_name as merchant_name,
                       users,
                       merchant_avg_fraud_score,
                       SIZE(users) as user_count
                ORDER BY merchant_avg_fraud_score DESC, user_count DESC
                LIMIT 20
                """
                
                result = session.run(query, {
                    'time_window': time_window_minutes,
                    'min_transactions': min_transactions
                })
                
                fraud_rings = []
                for record in result:
                    ring = FraudRing(
                        ring_id=f"velocity_{record['merchant_id']}",
                        members=[GraphNode(
                            node_id=user['user_id'],
                            node_type='User',
                            properties=dict(user),
                            labels=['User']
                        ) for user in record['users']],
                        relationships=[],
                        risk_score=record['merchant_avg_fraud_score'],
                        detection_algorithm='velocity_analysis',
                        confidence=min(record['merchant_avg_fraud_score'] * record['user_count'] / 5, 1.0),
                        detected_at=datetime.now()
                    )
                    fraud_rings.append(ring)
                
                logger.info(f"Detected {len(fraud_rings)} velocity-based fraud rings")
                return fraud_rings
                
        except Exception as e:
            logger.error(f"Failed to detect velocity rings: {e}")
            return []
    
    def detect_community_rings(self, algorithm: str = 'louvain') -> List[FraudRing]:
        """Detect fraud rings using graph community detection algorithms"""
        try:
            with self.driver.session() as session:
                # First, create a graph projection
                projection_query = """
                CALL gds.graph.project(
                    'fraud-network',
                    ['User', 'Merchant', 'Device'],
                    {
                        MADE_TRANSACTION: {orientation: 'UNDIRECTED'},
                        USED_DEVICE: {orientation: 'UNDIRECTED'},
                        PAID_TO: {orientation: 'UNDIRECTED'}
                    }
                )
                """
                
                try:
                    session.run(projection_query)
                except Exception:
                    # Graph might already exist, drop and recreate
                    session.run("CALL gds.graph.drop('fraud-network', false)")
                    session.run(projection_query)
                
                # Run community detection
                if algorithm == 'louvain':
                    community_query = """
                    CALL gds.louvain.stream('fraud-network')
                    YIELD nodeId, communityId
                    WITH gds.util.asNode(nodeId) as node, communityId
                    WHERE node:User AND node.risk_score > 0.5
                    RETURN communityId, COLLECT(node) as community_members, 
                           AVG(node.risk_score) as avg_risk_score,
                           COUNT(node) as member_count
                    HAVING member_count >= 2 AND avg_risk_score > 0.6
                    ORDER BY avg_risk_score DESC, member_count DESC
                    LIMIT 10
                    """
                else:
                    # Default to label propagation
                    community_query = """
                    CALL gds.labelPropagation.stream('fraud-network')
                    YIELD nodeId, communityId
                    WITH gds.util.asNode(nodeId) as node, communityId
                    WHERE node:User AND node.risk_score > 0.5
                    RETURN communityId, COLLECT(node) as community_members,
                           AVG(node.risk_score) as avg_risk_score,
                           COUNT(node) as member_count
                    HAVING member_count >= 2 AND avg_risk_score > 0.6
                    ORDER BY avg_risk_score DESC, member_count DESC
                    LIMIT 10
                    """
                
                result = session.run(community_query)
                
                fraud_rings = []
                for record in result:
                    ring = FraudRing(
                        ring_id=f"community_{algorithm}_{record['communityId']}",
                        members=[GraphNode(
                            node_id=user['user_id'],
                            node_type='User',
                            properties=dict(user),
                            labels=['User']
                        ) for user in record['community_members']],
                        relationships=[],
                        risk_score=record['avg_risk_score'],
                        detection_algorithm=f'community_{algorithm}',
                        confidence=min(record['avg_risk_score'] * record['member_count'] / 8, 1.0),
                        detected_at=datetime.now()
                    )
                    fraud_rings.append(ring)
                
                # Clean up graph projection
                session.run("CALL gds.graph.drop('fraud-network', false)")
                
                logger.info(f"Detected {len(fraud_rings)} community-based fraud rings using {algorithm}")
                return fraud_rings
                
        except Exception as e:
            logger.error(f"Failed to detect community rings: {e}")
            return []
    
    # =====================================================
    # RELATIONSHIP ANALYSIS
    # =====================================================
    
    def calculate_relationship_strength(self, source_id: str, target_id: str, 
                                      relationship_type: str) -> float:
        """Calculate the strength of a relationship between two entities"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a)-[r]->(b)
                WHERE (a.user_id = $source_id OR a.merchant_id = $source_id OR a.device_id = $source_id)
                  AND (b.user_id = $target_id OR b.merchant_id = $target_id OR b.device_id = $target_id)
                  AND TYPE(r) = $relationship_type
                
                RETURN 
                    COUNT(r) as relationship_count,
                    AVG(CASE WHEN r.amount IS NOT NULL THEN r.amount ELSE 1 END) as avg_amount,
                    MAX(r.frequency) as max_frequency,
                    duration.between(MIN(r.timestamp), MAX(r.timestamp)).days as duration_days
                """
                
                result = session.run(query, {
                    'source_id': source_id,
                    'target_id': target_id,
                    'relationship_type': relationship_type
                })
                
                record = result.single()
                if record:
                    # Calculate strength based on multiple factors
                    count_factor = min(record['relationship_count'] / 10.0, 1.0)
                    amount_factor = min((record['avg_amount'] or 0) / 1000.0, 1.0)
                    frequency_factor = min((record['max_frequency'] or 0) / 50.0, 1.0)
                    duration_factor = min((record['duration_days'] or 0) / 365.0, 1.0)
                    
                    strength = (count_factor + amount_factor + frequency_factor + duration_factor) / 4.0
                    return min(strength, 1.0)
                
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to calculate relationship strength: {e}")
            return 0.0
    
    def get_entity_network(self, entity_id: str, max_depth: int = 2, 
                          min_relationship_strength: float = 0.3) -> Dict[str, Any]:
        """Get the network of relationships around an entity"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = (start)-[*1..$max_depth]-(connected)
                WHERE (start.user_id = $entity_id OR start.merchant_id = $entity_id OR start.device_id = $entity_id)
                
                WITH path, relationships(path) as rels, nodes(path) as path_nodes
                UNWIND rels as rel
                UNWIND path_nodes as node
                
                RETURN DISTINCT
                    CASE 
                        WHEN node.user_id IS NOT NULL THEN node.user_id
                        WHEN node.merchant_id IS NOT NULL THEN node.merchant_id
                        WHEN node.device_id IS NOT NULL THEN node.device_id
                        ELSE toString(id(node))
                    END as node_id,
                    labels(node) as node_labels,
                    properties(node) as node_properties,
                    TYPE(rel) as relationship_type,
                    properties(rel) as relationship_properties
                LIMIT 1000
                """
                
                result = session.run(query, {
                    'entity_id': entity_id,
                    'max_depth': max_depth
                })
                
                nodes = {}
                relationships = []
                
                for record in result:
                    node_id = record['node_id']
                    if node_id not in nodes:
                        nodes[node_id] = {
                            'id': node_id,
                            'labels': record['node_labels'],
                            'properties': record['node_properties']
                        }
                    
                    if record['relationship_type']:
                        relationships.append({
                            'type': record['relationship_type'],
                            'properties': record['relationship_properties']
                        })
                
                return {
                    'center_entity': entity_id,
                    'nodes': nodes,
                    'relationships': relationships,
                    'node_count': len(nodes),
                    'relationship_count': len(relationships)
                }
                
        except Exception as e:
            logger.error(f"Failed to get entity network: {e}")
            return {}
    
    # =====================================================
    # ANALYTICS AND REPORTING
    # =====================================================
    
    def get_fraud_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get fraud detection statistics from the graph"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (t:Transaction)
                WHERE t.timestamp > datetime() - duration({hours: $time_window})
                
                WITH t
                OPTIONAL MATCH (u:User)-[:MADE_TRANSACTION]->(t)
                OPTIONAL MATCH (t)-[:PAID_TO]->(m:Merchant)
                
                RETURN 
                    COUNT(t) as total_transactions,
                    COUNT(CASE WHEN t.fraud_score > 0.7 THEN 1 END) as high_risk_transactions,
                    AVG(t.fraud_score) as avg_fraud_score,
                    COUNT(DISTINCT u.user_id) as unique_users,
                    COUNT(DISTINCT m.merchant_id) as unique_merchants,
                    SUM(t.amount) as total_amount,
                    AVG(t.processing_time_ms) as avg_processing_time
                """
                
                result = session.run(query, {'time_window': time_window_hours})
                record = result.single()
                
                if record:
                    return {
                        'total_transactions': record['total_transactions'],
                        'high_risk_transactions': record['high_risk_transactions'],
                        'fraud_rate': record['high_risk_transactions'] / max(record['total_transactions'], 1),
                        'avg_fraud_score': round(record['avg_fraud_score'] or 0, 4),
                        'unique_users': record['unique_users'],
                        'unique_merchants': record['unique_merchants'],
                        'total_amount': record['total_amount'] or 0,
                        'avg_processing_time': round(record['avg_processing_time'] or 0, 2),
                        'time_window_hours': time_window_hours
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get fraud statistics: {e}")
            return {}
    
    def get_top_risky_entities(self, entity_type: str = 'User', limit: int = 10) -> List[Dict[str, Any]]:
        """Get top risky entities (users, merchants, devices)"""
        try:
            with self.driver.session() as session:
                if entity_type == 'User':
                    query = """
                    MATCH (u:User)
                    OPTIONAL MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)
                    WHERE t.timestamp > datetime() - duration({days: 30})
                    
                    WITH u, COUNT(t) as recent_transactions, AVG(t.fraud_score) as avg_fraud_score
                    RETURN 
                        u.user_id as entity_id,
                        u.risk_score as risk_score,
                        recent_transactions,
                        avg_fraud_score,
                        u.total_transactions as total_transactions,
                        u.total_amount as total_amount
                    ORDER BY u.risk_score DESC, avg_fraud_score DESC
                    LIMIT $limit
                    """
                elif entity_type == 'Merchant':
                    query = """
                    MATCH (m:Merchant)
                    OPTIONAL MATCH (t:Transaction)-[:PAID_TO]->(m)
                    WHERE t.timestamp > datetime() - duration({days: 30})
                    
                    WITH m, COUNT(t) as recent_transactions, AVG(t.fraud_score) as avg_fraud_score
                    RETURN 
                        m.merchant_id as entity_id,
                        m.risk_score as risk_score,
                        recent_transactions,
                        avg_fraud_score,
                        m.total_transactions as total_transactions,
                        m.total_amount as total_amount
                    ORDER BY m.risk_score DESC, avg_fraud_score DESC
                    LIMIT $limit
                    """
                else:  # Device
                    query = """
                    MATCH (d:Device)
                    OPTIONAL MATCH (u:User)-[:USED_DEVICE]->(d)
                    OPTIONAL MATCH (u)-[:MADE_TRANSACTION]->(t:Transaction)
                    WHERE t.timestamp > datetime() - duration({days: 30})
                    
                    WITH d, COUNT(DISTINCT u) as user_count, AVG(t.fraud_score) as avg_fraud_score
                    RETURN 
                        d.device_id as entity_id,
                        d.risk_score as risk_score,
                        user_count,
                        avg_fraud_score,
                        d.usage_count as usage_count
                    ORDER BY d.risk_score DESC, user_count DESC
                    LIMIT $limit
                    """
                
                result = session.run(query, {'limit': limit})
                
                entities = []
                for record in result:
                    entities.append(dict(record))
                
                logger.info(f"Retrieved {len(entities)} top risky {entity_type.lower()}s")
                return entities
                
        except Exception as e:
            logger.error(f"Failed to get top risky entities: {e}")
            return []

# Example usage and testing
if __name__ == "__main__":
    # Test Neo4j connection and operations
    try:
        with Neo4jManager() as neo4j:
            # Test connection
            stats = neo4j.get_fraud_statistics(24)
            print(f"Fraud statistics (24h): {stats}")
            
            # Test fraud ring detection
            device_rings = neo4j.detect_device_sharing_rings(min_users=2, min_risk_score=0.5)
            print(f"Detected {len(device_rings)} device sharing rings")
            
            velocity_rings = neo4j.detect_velocity_rings(time_window_minutes=60, min_transactions=3)
            print(f"Detected {len(velocity_rings)} velocity-based rings")
            
            # Test entity analysis
            top_users = neo4j.get_top_risky_entities('User', limit=5)
            print(f"Top 5 risky users: {[u['entity_id'] for u in top_users]}")
            
            print("✅ Neo4j Manager test completed successfully!")
            
    except Exception as e:
        print(f"❌ Neo4j Manager test failed: {e}")