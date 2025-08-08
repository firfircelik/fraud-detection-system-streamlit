#!/usr/bin/env python3
"""
Kafka Streaming Manager for Real-time Fraud Detection
Handles real-time transaction streaming and processing
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aiokafka
import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


@dataclass
class StreamEvent:
    event_id: str
    event_type: str
    timestamp: datetime
    payload: Dict[str, Any]
    source: str
    partition_key: str


@dataclass
class TransactionEvent:
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    device_id: Optional[str]
    ip_address: Optional[str]
    location: Optional[Dict[str, float]]


@dataclass
class FraudAlert:
    alert_id: str
    transaction_id: str
    fraud_score: float
    risk_level: str
    decision: str
    model_name: str
    timestamp: datetime
    risk_factors: List[str]


class KafkaManager:
    """Manages Kafka streaming for real-time fraud detection"""

    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
        self.pg_dsn = os.getenv(
            "POSTGRES_URL",
            "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection",
        )

        # Topic configuration
        self.topics = {
            "transactions": "fraud.transactions",
            "fraud_alerts": "fraud.alerts",
            "model_predictions": "fraud.predictions",
            "user_events": "fraud.user_events",
            "system_events": "fraud.system_events",
        }

        self.producer = None
        self.consumers = {}
        self.pg_pool = None

    async def initialize(self):
        """Initialize Kafka producer and database connection"""
        # Initialize Kafka producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            compression_type="gzip",
            batch_size=16384,
            linger_ms=10,
            acks="all",
            retries=3,
            max_in_flight_requests_per_connection=1,
        )

        await self.producer.start()

        # Initialize PostgreSQL connection
        self.pg_pool = await asyncpg.create_pool(
            self.pg_dsn, min_size=5, max_size=20, command_timeout=60
        )

        # Create stream processing state table
        await self.create_stream_tables()

        logger.info("Kafka manager initialized")

    async def create_stream_tables(self):
        """Create tables for stream processing state"""
        async with self.pg_pool.acquire() as conn:
            # Stream processing state table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_processing_state (
                    id BIGSERIAL PRIMARY KEY,
                    stream_name VARCHAR(100) NOT NULL,
                    partition_id INTEGER NOT NULL,
                    offset_position BIGINT NOT NULL,
                    checkpoint_data JSONB DEFAULT '{}',
                    last_processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    processing_lag_ms INTEGER DEFAULT 0,
                    UNIQUE(stream_name, partition_id)
                )
            """
            )

            # Real-time fraud events table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS real_time_fraud_events (
                    id BIGSERIAL PRIMARY KEY,
                    event_id VARCHAR(100) UNIQUE NOT NULL,
                    transaction_id VARCHAR(100),
                    event_type VARCHAR(50) NOT NULL,
                    fraud_score DECIMAL(5,4),
                    risk_level VARCHAR(20),
                    decision VARCHAR(20),
                    model_name VARCHAR(100),
                    processing_time_ms INTEGER,
                    event_data JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

    # =====================================================
    # PRODUCER METHODS
    # =====================================================

    async def publish_transaction_event(self, transaction: TransactionEvent) -> bool:
        """Publish transaction event to Kafka"""
        try:
            event = StreamEvent(
                event_id=f"tx_{transaction.transaction_id}_{int(datetime.now().timestamp())}",
                event_type="transaction",
                timestamp=transaction.timestamp,
                payload=asdict(transaction),
                source="fraud_api",
                partition_key=transaction.user_id,
            )

            await self.producer.send(
                self.topics["transactions"],
                value=asdict(event),
                key=event.partition_key,
            )

            logger.debug(f"Published transaction event: {transaction.transaction_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish transaction event: {e}")
            return False

    async def publish_fraud_alert(self, alert: FraudAlert) -> bool:
        """Publish fraud alert to Kafka"""
        try:
            event = StreamEvent(
                event_id=alert.alert_id,
                event_type="fraud_alert",
                timestamp=alert.timestamp,
                payload=asdict(alert),
                source="fraud_detector",
                partition_key=alert.transaction_id,
            )

            await self.producer.send(
                self.topics["fraud_alerts"],
                value=asdict(event),
                key=event.partition_key,
            )

            logger.info(f"Published fraud alert: {alert.alert_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish fraud alert: {e}")
            return False

    async def publish_model_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Publish model prediction result"""
        try:
            event = StreamEvent(
                event_id=f"pred_{prediction_data['transaction_id']}_{int(datetime.now().timestamp())}",
                event_type="model_prediction",
                timestamp=datetime.now(timezone.utc),
                payload=prediction_data,
                source="ml_engine",
                partition_key=prediction_data["transaction_id"],
            )

            await self.producer.send(
                self.topics["model_predictions"],
                value=asdict(event),
                key=event.partition_key,
            )

            logger.debug(
                f"Published model prediction: {prediction_data['transaction_id']}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish model prediction: {e}")
            return False

    async def publish_user_event(
        self, user_id: str, event_type: str, event_data: Dict[str, Any]
    ) -> bool:
        """Publish user behavior event"""
        try:
            event = StreamEvent(
                event_id=f"user_{user_id}_{event_type}_{int(datetime.now().timestamp())}",
                event_type=event_type,
                timestamp=datetime.now(timezone.utc),
                payload={
                    "user_id": user_id,
                    "event_type": event_type,
                    "event_data": event_data,
                },
                source="user_tracker",
                partition_key=user_id,
            )

            await self.producer.send(
                self.topics["user_events"], value=asdict(event), key=event.partition_key
            )

            logger.debug(f"Published user event: {user_id} - {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish user event: {e}")
            return False

    # =====================================================
    # CONSUMER METHODS
    # =====================================================

    async def create_consumer(
        self, topic: str, group_id: str, auto_offset_reset: str = "latest"
    ) -> AIOKafkaConsumer:
        """Create Kafka consumer"""
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            enable_auto_commit=False,
            max_poll_records=100,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        await consumer.start()
        self.consumers[f"{topic}_{group_id}"] = consumer
        return consumer

    async def consume_transactions(self, callback: Callable[[TransactionEvent], None]):
        """Consume transaction events"""
        consumer = await self.create_consumer(
            self.topics["transactions"], "fraud_processor", "earliest"
        )

        try:
            async for message in consumer:
                try:
                    event_data = message.value
                    transaction_data = event_data["payload"]

                    transaction = TransactionEvent(**transaction_data)

                    # Process transaction
                    await callback(transaction)

                    # Commit offset
                    await consumer.commit()

                    # Update processing state
                    await self.update_processing_state(
                        "transactions", message.partition, message.offset
                    )

                except Exception as e:
                    logger.error(f"Error processing transaction message: {e}")

        except Exception as e:
            logger.error(f"Error in transaction consumer: {e}")
        finally:
            await consumer.stop()

    async def consume_fraud_alerts(self, callback: Callable[[FraudAlert], None]):
        """Consume fraud alert events"""
        consumer = await self.create_consumer(
            self.topics["fraud_alerts"], "alert_processor", "earliest"
        )

        try:
            async for message in consumer:
                try:
                    event_data = message.value
                    alert_data = event_data["payload"]

                    alert = FraudAlert(**alert_data)

                    # Process alert
                    await callback(alert)

                    # Store in database
                    await self.store_fraud_event(alert)

                    # Commit offset
                    await consumer.commit()

                except Exception as e:
                    logger.error(f"Error processing fraud alert: {e}")

        except Exception as e:
            logger.error(f"Error in fraud alert consumer: {e}")
        finally:
            await consumer.stop()

    # =====================================================
    # STREAM PROCESSING
    # =====================================================

    async def process_transaction_stream(self):
        """Process transaction stream for real-time fraud detection"""

        async def process_transaction(transaction: TransactionEvent):
            try:
                # Simulate fraud detection processing
                fraud_score = await self.calculate_fraud_score(transaction)

                # Determine risk level and decision
                if fraud_score >= 0.8:
                    risk_level = "CRITICAL"
                    decision = "DECLINED"
                elif fraud_score >= 0.6:
                    risk_level = "HIGH"
                    decision = "REVIEW"
                elif fraud_score >= 0.4:
                    risk_level = "MEDIUM"
                    decision = "APPROVED"
                else:
                    risk_level = "LOW"
                    decision = "APPROVED"

                # Create fraud alert if high risk
                if fraud_score >= 0.6:
                    alert = FraudAlert(
                        alert_id=f"alert_{transaction.transaction_id}_{int(datetime.now().timestamp())}",
                        transaction_id=transaction.transaction_id,
                        fraud_score=fraud_score,
                        risk_level=risk_level,
                        decision=decision,
                        model_name="real_time_detector",
                        timestamp=datetime.now(timezone.utc),
                        risk_factors=["high_fraud_score", "unusual_pattern"],
                    )

                    await self.publish_fraud_alert(alert)

                # Publish prediction result
                prediction_data = {
                    "transaction_id": transaction.transaction_id,
                    "fraud_score": fraud_score,
                    "risk_level": risk_level,
                    "decision": decision,
                    "model_name": "real_time_detector",
                    "processing_time_ms": 50,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                await self.publish_model_prediction(prediction_data)

                logger.info(
                    f"Processed transaction {transaction.transaction_id}: {fraud_score:.3f} ({risk_level})"
                )

            except Exception as e:
                logger.error(
                    f"Error processing transaction {transaction.transaction_id}: {e}"
                )

        # Start consuming transactions
        await self.consume_transactions(process_transaction)

    async def calculate_fraud_score(self, transaction: TransactionEvent) -> float:
        """Calculate fraud score for transaction (simplified)"""
        try:
            score = 0.0

            # Amount-based scoring
            if transaction.amount > 10000:
                score += 0.4
            elif transaction.amount > 5000:
                score += 0.2
            elif transaction.amount < 1:
                score += 0.3

            # Time-based scoring
            hour = transaction.timestamp.hour
            if hour < 6 or hour > 22:
                score += 0.2

            # User velocity check (simplified)
            user_velocity = await self.get_user_velocity(transaction.user_id)
            if user_velocity > 10:  # More than 10 transactions in last hour
                score += 0.3

            # Merchant risk check
            merchant_risk = await self.get_merchant_risk(transaction.merchant_id)
            score += merchant_risk * 0.2

            # Add some randomness for demo
            import random

            score += random.uniform(-0.1, 0.1)

            return min(max(score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Error calculating fraud score: {e}")
            return 0.5  # Default score

    async def get_user_velocity(self, user_id: str) -> int:
        """Get user transaction velocity (transactions per hour)"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT COUNT(*) as tx_count
                    FROM transactions
                    WHERE user_id = $1 AND transaction_timestamp > NOW() - INTERVAL '1 hour'
                """,
                    user_id,
                )

                return result["tx_count"] if result else 0

        except Exception as e:
            logger.error(f"Error getting user velocity: {e}")
            return 0

    async def get_merchant_risk(self, merchant_id: str) -> float:
        """Get merchant risk score"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT risk_score FROM merchants WHERE merchant_id = $1
                """,
                    merchant_id,
                )

                return float(result["risk_score"]) if result else 0.5

        except Exception as e:
            logger.error(f"Error getting merchant risk: {e}")
            return 0.5

    # =====================================================
    # STATE MANAGEMENT
    # =====================================================

    async def update_processing_state(
        self, stream_name: str, partition_id: int, offset: int
    ):
        """Update stream processing state"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO stream_processing_state
                    (stream_name, partition_id, offset_position, last_processed_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (stream_name, partition_id)
                    DO UPDATE SET
                        offset_position = EXCLUDED.offset_position,
                        last_processed_at = EXCLUDED.last_processed_at
                """,
                    stream_name,
                    partition_id,
                    offset,
                )

        except Exception as e:
            logger.error(f"Error updating processing state: {e}")

    async def get_processing_state(
        self, stream_name: str, partition_id: int
    ) -> Optional[int]:
        """Get last processed offset for stream partition"""
        try:
            async with self.pg_pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT offset_position FROM stream_processing_state
                    WHERE stream_name = $1 AND partition_id = $2
                """,
                    stream_name,
                    partition_id,
                )

                return result["offset_position"] if result else None

        except Exception as e:
            logger.error(f"Error getting processing state: {e}")
            return None

    async def store_fraud_event(self, alert: FraudAlert):
        """Store fraud event in database"""
        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO real_time_fraud_events
                    (event_id, transaction_id, event_type, fraud_score, risk_level,
                     decision, model_name, event_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    (
                        alert.alert_id,
                        alert.transaction_id,
                        "fraud_alert",
                        alert.fraud_score,
                        alert.risk_level,
                        alert.decision,
                        alert.model_name,
                        {"risk_factors": alert.risk_factors},
                    ),
                )

        except Exception as e:
            logger.error(f"Error storing fraud event: {e}")

    # =====================================================
    # MONITORING AND HEALTH
    # =====================================================

    async def get_stream_health(self) -> Dict[str, Any]:
        """Get stream processing health metrics"""
        try:
            async with self.pg_pool.acquire() as conn:
                # Get processing lag for each stream
                lag_results = await conn.fetch(
                    """
                    SELECT stream_name, partition_id, processing_lag_ms, last_processed_at
                    FROM stream_processing_state
                    ORDER BY stream_name, partition_id
                """
                )

                # Get recent event counts
                event_counts = await conn.fetch(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM real_time_fraud_events
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    GROUP BY event_type
                """
                )

                return {
                    "processing_lag": [dict(row) for row in lag_results],
                    "event_counts": {
                        row["event_type"]: row["count"] for row in event_counts
                    },
                    "total_consumers": len(self.consumers),
                    "producer_status": "healthy" if self.producer else "disconnected",
                }

        except Exception as e:
            logger.error(f"Error getting stream health: {e}")
            return {}

    async def close(self):
        """Close Kafka connections"""
        if self.producer:
            await self.producer.stop()

        for consumer in self.consumers.values():
            await consumer.stop()

        if self.pg_pool:
            await self.pg_pool.close()


# Example usage and testing
async def main():
    kafka_manager = KafkaManager()
    await kafka_manager.initialize()

    try:
        # Test transaction publishing
        transaction = TransactionEvent(
            transaction_id="test_tx_001",
            user_id="user_001",
            merchant_id="merchant_001",
            amount=1500.00,
            currency="USD",
            timestamp=datetime.now(timezone.utc),
            device_id="device_001",
            ip_address="192.168.1.100",
            location={"lat": 40.7128, "lon": -74.0060},
        )

        await kafka_manager.publish_transaction_event(transaction)
        print(f"Published transaction: {transaction.transaction_id}")

        # Test fraud alert publishing
        alert = FraudAlert(
            alert_id="alert_001",
            transaction_id="test_tx_001",
            fraud_score=0.85,
            risk_level="HIGH",
            decision="DECLINED",
            model_name="test_model",
            timestamp=datetime.now(timezone.utc),
            risk_factors=["high_amount", "unusual_time"],
        )

        await kafka_manager.publish_fraud_alert(alert)
        print(f"Published fraud alert: {alert.alert_id}")

        # Get stream health
        health = await kafka_manager.get_stream_health()
        print(f"Stream health: {health}")

        print("✅ Kafka manager test completed successfully!")

        # Start stream processing (uncomment to test)
        # await kafka_manager.process_transaction_stream()

    except Exception as e:
        print(f"❌ Kafka manager test failed: {e}")
    finally:
        await kafka_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
