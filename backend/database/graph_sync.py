#!/usr/bin/env python3
"""
Graph Data Synchronization Pipeline
Syncs data between PostgreSQL and Neo4j for fraud detection
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncpg
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass
class SyncEvent:
    table_name: str
    operation: str  # 'INSERT', 'UPDATE', 'DELETE'
    old_data: Optional[Dict[str, Any]]
    new_data: Optional[Dict[str, Any]]
    timestamp: datetime


class GraphSyncManager:
    """Manages synchronization between PostgreSQL and Neo4j"""

    def __init__(self):
        # PostgreSQL connection
        self.pg_dsn = os.getenv(
            "POSTGRES_URL",
            "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection",
        )

        # Neo4j connection
        self.neo4j_uri = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "FraudGraph2024!")

        self.neo4j_driver = None
        self.pg_pool = None

    async def initialize(self):
        """Initialize database connections"""
        # Initialize PostgreSQL connection pool
        self.pg_pool = await asyncpg.create_pool(
            self.pg_dsn, min_size=5, max_size=20, command_timeout=60
        )

        # Initialize Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
        )

        # Set up PostgreSQL triggers for change tracking
        await self.setup_change_tracking()

        logger.info("Graph sync manager initialized")

    async def setup_change_tracking(self):
        """Set up PostgreSQL triggers to track changes"""
        async with self.pg_pool.acquire() as conn:
            # Create change log table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_sync_log (
                    id BIGSERIAL PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    old_data JSONB,
                    new_data JSONB,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    synced BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Create trigger function
            await conn.execute(
                """
                CREATE OR REPLACE FUNCTION log_graph_changes()
                RETURNS TRIGGER AS $$
                BEGIN
                    IF TG_OP = 'DELETE' THEN
                        INSERT INTO graph_sync_log (table_name, operation, old_data)
                        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD));
                        RETURN OLD;
                    ELSIF TG_OP = 'UPDATE' THEN
                        INSERT INTO graph_sync_log (table_name, operation, old_data, new_data)
                        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(OLD), row_to_json(NEW));
                        RETURN NEW;
                    ELSIF TG_OP = 'INSERT' THEN
                        INSERT INTO graph_sync_log (table_name, operation, new_data)
                        VALUES (TG_TABLE_NAME, TG_OP, row_to_json(NEW));
                        RETURN NEW;
                    END IF;
                    RETURN NULL;
                END;
                $$ LANGUAGE plpgsql;
            """
            )

            # Create triggers on relevant tables
            tables = ["users", "merchants", "transactions"]
            for table in tables:
                await conn.execute(
                    f"""
                    DROP TRIGGER IF EXISTS {table}_graph_sync ON {table};
                    CREATE TRIGGER {table}_graph_sync
                    AFTER INSERT OR UPDATE OR DELETE ON {table}
                    FOR EACH ROW EXECUTE FUNCTION log_graph_changes();
                """
                )

    async def sync_user_to_graph(self, user_data: Dict[str, Any], operation: str):
        """Sync user data to Neo4j"""
        with self.neo4j_driver.session() as session:
            if operation == "DELETE":
                query = "MATCH (u:User {user_id: $user_id}) DETACH DELETE u"
                session.run(query, {"user_id": user_data["user_id"]})
            else:
                query = """
                MERGE (u:User {user_id: $user_id})
                SET u += $properties
                SET u.updated_at = datetime()
                """
                session.run(
                    query, {"user_id": user_data["user_id"], "properties": user_data}
                )

    async def sync_merchant_to_graph(
        self, merchant_data: Dict[str, Any], operation: str
    ):
        """Sync merchant data to Neo4j"""
        with self.neo4j_driver.session() as session:
            if operation == "DELETE":
                query = "MATCH (m:Merchant {merchant_id: $merchant_id}) DETACH DELETE m"
                session.run(query, {"merchant_id": merchant_data["merchant_id"]})
            else:
                query = """
                MERGE (m:Merchant {merchant_id: $merchant_id})
                SET m += $properties
                SET m.updated_at = datetime()
                """
                session.run(
                    query,
                    {
                        "merchant_id": merchant_data["merchant_id"],
                        "properties": merchant_data,
                    },
                )

    async def sync_transaction_to_graph(self, tx_data: Dict[str, Any], operation: str):
        """Sync transaction data to Neo4j with relationships"""
        with self.neo4j_driver.session() as session:
            if operation == "DELETE":
                query = "MATCH (t:Transaction {transaction_id: $transaction_id}) DETACH DELETE t"
                session.run(query, {"transaction_id": tx_data["transaction_id"]})
            else:
                query = """
                // Create transaction node
                MERGE (t:Transaction {transaction_id: $transaction_id})
                SET t += $properties
                SET t.updated_at = datetime()
                
                // Create user relationship
                WITH t
                MATCH (u:User {user_id: $user_id})
                MERGE (u)-[r1:MADE_TRANSACTION]->(t)
                SET r1.timestamp = datetime($timestamp),
                    r1.amount = $amount
                
                // Create merchant relationship
                WITH t
                MATCH (m:Merchant {merchant_id: $merchant_id})
                MERGE (t)-[r2:PAID_TO]->(m)
                SET r2.timestamp = datetime($timestamp),
                    r2.amount = $amount
                """

                session.run(
                    query,
                    {
                        "transaction_id": tx_data["transaction_id"],
                        "properties": tx_data,
                        "user_id": tx_data["user_id"],
                        "merchant_id": tx_data["merchant_id"],
                        "timestamp": tx_data["transaction_timestamp"].isoformat(),
                        "amount": tx_data["amount"],
                    },
                )

    async def process_sync_events(self):
        """Process pending sync events"""
        async with self.pg_pool.acquire() as conn:
            # Get unsynced events
            events = await conn.fetch(
                """
                SELECT id, table_name, operation, old_data, new_data, timestamp
                FROM graph_sync_log
                WHERE synced = FALSE
                ORDER BY timestamp
                LIMIT 100
            """
            )

            for event in events:
                try:
                    # Process based on table
                    if event["table_name"] == "users":
                        data = event["new_data"] or event["old_data"]
                        await self.sync_user_to_graph(data, event["operation"])

                    elif event["table_name"] == "merchants":
                        data = event["new_data"] or event["old_data"]
                        await self.sync_merchant_to_graph(data, event["operation"])

                    elif event["table_name"] == "transactions":
                        data = event["new_data"] or event["old_data"]
                        await self.sync_transaction_to_graph(data, event["operation"])

                    # Mark as synced
                    await conn.execute(
                        "UPDATE graph_sync_log SET synced = TRUE WHERE id = $1",
                        event["id"],
                    )

                except Exception as e:
                    logger.error(f"Failed to sync event {event['id']}: {e}")

    async def run_sync_loop(self):
        """Run continuous sync loop"""
        while True:
            try:
                await self.process_sync_events()
                await asyncio.sleep(5)  # Process every 5 seconds
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(10)

    async def close(self):
        """Close connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()


# Standalone sync service
async def main():
    sync_manager = GraphSyncManager()
    await sync_manager.initialize()

    try:
        await sync_manager.run_sync_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down sync service")
    finally:
        await sync_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
