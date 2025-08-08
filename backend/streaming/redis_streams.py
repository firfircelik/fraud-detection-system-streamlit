#!/usr/bin/env python3
"""
Redis Streams Manager for Real-time Event Processing
Handles fraud event streams, consumer groups, and monitoring
"""

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


@dataclass
class FraudEvent:
    """Fraud event structure for streams"""

    event_id: str
    event_type: str  # 'fraud_alert', 'transaction_processed', 'model_prediction'
    transaction_id: str
    user_id: str
    merchant_id: str
    fraud_score: float
    risk_level: str
    decision: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class StreamConsumerInfo:
    """Consumer group information"""

    stream_name: str
    group_name: str
    consumer_name: str
    last_id: str
    pending_count: int
    idle_time_ms: int


class RedisStreamsManager:
    """Redis Streams manager for real-time fraud event processing"""

    def __init__(self):
        self.redis_url = os.getenv(
            "REDIS_URL", "redis://:RedisStack2024!@localhost:6379"
        )
        self.redis_client = None

        # Stream configuration
        self.streams = {
            "fraud_events": "fraud:events",
            "transaction_events": "fraud:transactions",
            "model_predictions": "fraud:predictions",
            "alerts": "fraud:alerts",
        }

        self.consumer_groups = {
            "fraud_processors": "fraud_processing_group",
            "alert_handlers": "alert_handling_group",
            "analytics": "analytics_group",
        }

        # Processing configuration
        self.batch_size = 10
        self.block_time = 5000  # 5 seconds
        self.max_retries = 3

        self.running = False
        self.consumers = {}

    async def initialize(self):
        """Initialize Redis Streams connection and setup"""
        try:
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )

            # Test connection
            await self.redis_client.ping()

            # Create streams and consumer groups
            await self._setup_streams()

            logger.info("Redis Streams manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Redis Streams: {e}")
            raise

    async def _setup_streams(self):
        """Setup streams and consumer groups"""
        for stream_name in self.streams.values():
            try:
                # Create stream with initial dummy message if not exists
                await self.redis_client.xadd(stream_name, {"init": "true"}, id="0-1")

                # Create consumer groups
                for group_name in self.consumer_groups.values():
                    try:
                        await self.redis_client.xgroup_create(
                            stream_name, group_name, id="0", mkstream=True
                        )
                        logger.info(
                            f"Created consumer group {group_name} for stream {stream_name}"
                        )
                    except Exception:
                        # Group already exists
                        pass

            except Exception as e:
                logger.error(f"Failed to setup stream {stream_name}: {e}")

    # =====================================================
    # EVENT PUBLISHING
    # =====================================================

    async def publish_fraud_event(self, event: FraudEvent) -> str:
        """Publish fraud event to stream"""
        try:
            event_data = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "transaction_id": event.transaction_id,
                "user_id": event.user_id,
                "merchant_id": event.merchant_id,
                "fraud_score": str(event.fraud_score),
                "risk_level": event.risk_level,
                "decision": event.decision,
                "timestamp": event.timestamp.isoformat(),
                "metadata": json.dumps(event.metadata),
            }

            stream_id = await self.redis_client.xadd(
                self.streams["fraud_events"], event_data
            )

            logger.debug(
                f"Published fraud event {event.event_id} to stream: {stream_id}"
            )
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish fraud event: {e}")
            raise

    async def publish_transaction_event(self, transaction_data: Dict[str, Any]) -> str:
        """Publish transaction event to stream"""
        try:
            stream_id = await self.redis_client.xadd(
                self.streams["transaction_events"], transaction_data
            )

            logger.debug(f"Published transaction event to stream: {stream_id}")
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish transaction event: {e}")
            raise

    async def publish_model_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Publish model prediction to stream"""
        try:
            stream_id = await self.redis_client.xadd(
                self.streams["model_predictions"], prediction_data
            )

            logger.debug(f"Published model prediction to stream: {stream_id}")
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish model prediction: {e}")
            raise

    async def publish_alert(self, alert_data: Dict[str, Any]) -> str:
        """Publish alert to stream"""
        try:
            stream_id = await self.redis_client.xadd(self.streams["alerts"], alert_data)

            logger.info(f"Published alert to stream: {stream_id}")
            return stream_id

        except Exception as e:
            logger.error(f"Failed to publish alert: {e}")
            raise

    # =====================================================
    # CONSUMER GROUPS
    # =====================================================

    async def start_consumer(
        self, stream_name: str, group_name: str, consumer_name: str, handler: Callable
    ) -> None:
        """Start a consumer for processing events"""
        try:
            self.running = True
            consumer_key = f"{group_name}:{consumer_name}"
            self.consumers[consumer_key] = True

            logger.info(f"Starting consumer {consumer_name} for group {group_name}")

            while self.running and self.consumers.get(consumer_key, False):
                try:
                    # Read from stream
                    messages = await self.redis_client.xreadgroup(
                        group_name,
                        consumer_name,
                        {stream_name: ">"},
                        count=self.batch_size,
                        block=self.block_time,
                    )

                    if messages:
                        for stream, msgs in messages:
                            for msg_id, fields in msgs:
                                try:
                                    # Process message
                                    await handler(msg_id, fields)

                                    # Acknowledge message
                                    await self.redis_client.xack(
                                        stream_name, group_name, msg_id
                                    )

                                except Exception as e:
                                    logger.error(
                                        f"Error processing message {msg_id}: {e}"
                                    )
                                    # Could implement retry logic here

                except Exception as e:
                    logger.error(f"Error in consumer {consumer_name}: {e}")
                    await asyncio.sleep(1)  # Brief pause before retry

        except Exception as e:
            logger.error(f"Consumer {consumer_name} failed: {e}")
        finally:
            if consumer_key in self.consumers:
                del self.consumers[consumer_key]
            logger.info(f"Consumer {consumer_name} stopped")

    async def stop_consumer(self, group_name: str, consumer_name: str):
        """Stop a specific consumer"""
        consumer_key = f"{group_name}:{consumer_name}"
        if consumer_key in self.consumers:
            self.consumers[consumer_key] = False
            logger.info(f"Stopping consumer {consumer_name}")

    async def stop_all_consumers(self):
        """Stop all consumers"""
        self.running = False
        for consumer_key in list(self.consumers.keys()):
            self.consumers[consumer_key] = False
        logger.info("Stopping all consumers")

    # =====================================================
    # EVENT HANDLERS
    # =====================================================

    async def fraud_event_handler(self, message_id: str, fields: Dict[str, str]):
        """Handle fraud events"""
        try:
            # Parse fraud event
            fraud_score = float(fields.get("fraud_score", 0))
            risk_level = fields.get("risk_level", "LOW")

            logger.info(
                f"Processing fraud event {message_id}: score={fraud_score}, risk={risk_level}"
            )

            # High-risk fraud processing
            if fraud_score >= 0.8:
                await self._handle_high_risk_fraud(fields)
            elif fraud_score >= 0.6:
                await self._handle_medium_risk_fraud(fields)

            # Update metrics
            await self._update_fraud_metrics(fields)

        except Exception as e:
            logger.error(f"Error handling fraud event {message_id}: {e}")
            raise

    async def alert_handler(self, message_id: str, fields: Dict[str, str]):
        """Handle alert events"""
        try:
            alert_type = fields.get("alert_type", "unknown")
            severity = fields.get("severity", "low")

            logger.info(
                f"Processing alert {message_id}: type={alert_type}, severity={severity}"
            )

            # Route alert based on severity
            if severity in ["critical", "high"]:
                await self._send_immediate_alert(fields)
            else:
                await self._queue_alert_for_review(fields)

        except Exception as e:
            logger.error(f"Error handling alert {message_id}: {e}")
            raise

    async def analytics_handler(self, message_id: str, fields: Dict[str, str]):
        """Handle analytics events"""
        try:
            # Process for analytics and reporting
            await self._update_analytics_data(fields)

        except Exception as e:
            logger.error(f"Error handling analytics event {message_id}: {e}")
            raise

    # =====================================================
    # MONITORING AND MANAGEMENT
    # =====================================================

    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get stream information"""
        try:
            info = await self.redis_client.xinfo_stream(stream_name)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": info.get("groups", 0),
            }
        except Exception as e:
            logger.error(f"Error getting stream info for {stream_name}: {e}")
            return {}

    async def get_consumer_group_info(
        self, stream_name: str
    ) -> List[StreamConsumerInfo]:
        """Get consumer group information"""
        try:
            groups = await self.redis_client.xinfo_groups(stream_name)

            group_info = []
            for group in groups:
                consumers = await self.redis_client.xinfo_consumers(
                    stream_name, group["name"]
                )

                for consumer in consumers:
                    group_info.append(
                        StreamConsumerInfo(
                            stream_name=stream_name,
                            group_name=group["name"],
                            consumer_name=consumer["name"],
                            last_id=group.get("last-delivered-id", "0-0"),
                            pending_count=consumer.get("pending", 0),
                            idle_time_ms=consumer.get("idle", 0),
                        )
                    )

            return group_info

        except Exception as e:
            logger.error(f"Error getting consumer group info: {e}")
            return []

    async def get_pending_messages(
        self, stream_name: str, group_name: str
    ) -> List[Dict]:
        """Get pending messages for a consumer group"""
        try:
            pending = await self.redis_client.xpending(stream_name, group_name)
            return pending
        except Exception as e:
            logger.error(f"Error getting pending messages: {e}")
            return []

    async def trim_stream(self, stream_name: str, max_length: int = 10000):
        """Trim stream to prevent memory issues"""
        try:
            trimmed = await self.redis_client.xtrim(stream_name, maxlen=max_length)
            logger.info(f"Trimmed {trimmed} messages from stream {stream_name}")
            return trimmed
        except Exception as e:
            logger.error(f"Error trimming stream {stream_name}: {e}")
            return 0

    # =====================================================
    # HELPER METHODS
    # =====================================================

    async def _handle_high_risk_fraud(self, fields: Dict[str, str]):
        """Handle high-risk fraud events"""
        # Immediate alert
        alert_data = {
            "alert_type": "high_risk_fraud",
            "severity": "critical",
            "transaction_id": fields.get("transaction_id"),
            "fraud_score": fields.get("fraud_score"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.publish_alert(alert_data)

    async def _handle_medium_risk_fraud(self, fields: Dict[str, str]):
        """Handle medium-risk fraud events"""
        # Queue for review
        alert_data = {
            "alert_type": "medium_risk_fraud",
            "severity": "medium",
            "transaction_id": fields.get("transaction_id"),
            "fraud_score": fields.get("fraud_score"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.publish_alert(alert_data)

    async def _update_fraud_metrics(self, fields: Dict[str, str]):
        """Update fraud metrics"""
        # This could update metrics in TimescaleDB or other systems
        pass

    async def _send_immediate_alert(self, fields: Dict[str, str]):
        """Send immediate alert for critical issues"""
        # Implementation for immediate alerting (email, SMS, etc.)
        logger.critical(f"IMMEDIATE ALERT: {fields}")

    async def _queue_alert_for_review(self, fields: Dict[str, str]):
        """Queue alert for manual review"""
        # Implementation for queuing alerts
        logger.warning(f"QUEUED ALERT: {fields}")

    async def _update_analytics_data(self, fields: Dict[str, str]):
        """Update analytics data"""
        # Implementation for analytics updates
        pass

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis Streams connection closed")


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage of Redis Streams Manager"""
    streams_manager = RedisStreamsManager()

    try:
        await streams_manager.initialize()

        # Start consumers
        await asyncio.gather(
            streams_manager.start_consumer(
                streams_manager.streams["fraud_events"],
                streams_manager.consumer_groups["fraud_processors"],
                "processor_1",
                streams_manager.fraud_event_handler,
            ),
            streams_manager.start_consumer(
                streams_manager.streams["alerts"],
                streams_manager.consumer_groups["alert_handlers"],
                "alert_handler_1",
                streams_manager.alert_handler,
            ),
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await streams_manager.stop_all_consumers()
        await streams_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
