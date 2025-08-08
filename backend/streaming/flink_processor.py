#!/usr/bin/env python3
"""
Apache Flink Stream Processing for Real-time Fraud Detection
Real-time feature computation, windowing, and exactly-once semantics
"""

import asyncio
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import redis.asyncio as redis
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


@dataclass
class TransactionEvent:
    """Transaction event for stream processing"""

    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    location: Optional[Dict[str, Any]] = None
    device_info: Optional[Dict[str, Any]] = None
    transaction_type: str = "purchase"
    channel: str = "online"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FeatureVector:
    """Real-time computed feature vector"""

    transaction_id: str
    user_id: str
    features: Dict[str, float]
    computed_at: datetime
    feature_version: str = "1.0"


@dataclass
class FraudAlert:
    """Real-time fraud alert"""

    alert_id: str
    transaction_id: str
    user_id: str
    risk_score: float
    risk_level: str
    features_used: List[str]
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class WindowState:
    """State for windowed computations"""

    window_id: str
    window_type: str  # tumbling, sliding, session
    start_time: datetime
    end_time: Optional[datetime]
    events: List[TransactionEvent]
    aggregates: Dict[str, Any]
    state: Dict[str, Any]


class FlinkStreamProcessor:
    """
    Python implementation of Flink-style stream processing
    Features: Windowing, State Management, Exactly-once Processing
    """

    def __init__(
        self,
        kafka_config: Dict[str, Any],
        redis_config: Dict[str, Any],
        checkpoint_interval_ms: int = 30000,
    ):

        # Configuration
        self.kafka_config = kafka_config
        self.redis_config = redis_config
        self.checkpoint_interval_ms = checkpoint_interval_ms

        # Kafka clients
        self.producer = None
        self.consumer = None

        # Redis for state management
        self.redis_client = None

        # Stream processing state
        self.operators = {}
        self.watermarks = {}
        self.checkpoints = {}
        self.state_backend = {}

        # Windows management
        self.tumbling_windows = {}
        self.sliding_windows = {}
        self.session_windows = {}

        # Exactly-once processing
        self.processed_offsets = set()
        self.pending_transactions = {}
        self.transaction_timeout = timedelta(minutes=5)

        # Performance monitoring
        self.metrics = {
            "events_processed": 0,
            "features_computed": 0,
            "alerts_generated": 0,
            "processing_latency_ms": deque(maxlen=1000),
            "throughput_events_per_sec": deque(maxlen=100),
        }

        # Thread management
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Window configurations
        self.window_configs = {
            "user_activity_1min": {
                "type": "tumbling",
                "size_ms": 60000,
                "allowed_lateness_ms": 5000,
            },
            "user_activity_5min": {
                "type": "tumbling",
                "size_ms": 300000,
                "allowed_lateness_ms": 10000,
            },
            "velocity_sliding_10min": {
                "type": "sliding",
                "size_ms": 600000,
                "slide_ms": 60000,
                "allowed_lateness_ms": 30000,
            },
            "merchant_session": {
                "type": "session",
                "gap_ms": 1800000,  # 30 minutes
                "allowed_lateness_ms": 60000,
            },
        }

    async def initialize(self):
        """Initialize Kafka and Redis connections"""
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",  # Wait for all replicas
                retries=3,
                enable_idempotence=True,  # Exactly-once semantics
                max_in_flight_requests_per_connection=1,
            )

            # Initialize Kafka consumer
            self.consumer = KafkaConsumer(
                "fraud-transactions",
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                group_id="flink-fraud-processor",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                enable_auto_commit=False,  # Manual commit for exactly-once
                auto_offset_reset="earliest",
                isolation_level="read_committed",
            )

            # Initialize Redis
            self.redis_client = redis.Redis(
                host=self.redis_config["host"],
                port=self.redis_config["port"],
                password=self.redis_config.get("password"),
                decode_responses=True,
            )

            await self.redis_client.ping()
            logger.info("Flink stream processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize stream processor: {e}")
            raise

    # =====================================================
    # STREAM PROCESSING OPERATORS
    # =====================================================

    def register_operator(self, name: str, operator_func, parallelism: int = 1):
        """Register a stream processing operator"""
        self.operators[name] = {
            "function": operator_func,
            "parallelism": parallelism,
            "state": {},
            "watermark": 0,
        }

    async def process_stream(self):
        """Main stream processing loop"""
        self.running = True
        last_checkpoint = time.time()
        throughput_window = deque(maxlen=60)  # 1 minute window

        try:
            while self.running:
                batch_start = time.time()
                events_processed = 0

                # Process message batch
                message_batch = self.consumer.poll(timeout_ms=1000, max_records=100)

                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Check for duplicate processing
                            if self._is_duplicate(message):
                                continue

                            # Parse transaction event
                            event = self._parse_transaction_event(message.value)

                            # Process through operators
                            await self._process_event_through_operators(event, message)

                            events_processed += 1
                            self.metrics["events_processed"] += 1

                        except Exception as e:
                            logger.error(f"Error processing message: {e}")

                # Commit offsets for exactly-once processing
                if events_processed > 0:
                    self.consumer.commit()

                # Update throughput metrics
                batch_time = time.time() - batch_start
                if batch_time > 0:
                    throughput = events_processed / batch_time
                    throughput_window.append(throughput)

                    if len(throughput_window) >= 10:
                        avg_throughput = sum(throughput_window) / len(throughput_window)
                        self.metrics["throughput_events_per_sec"].append(avg_throughput)

                # Periodic checkpoint
                if time.time() - last_checkpoint > (self.checkpoint_interval_ms / 1000):
                    await self._create_checkpoint()
                    last_checkpoint = time.time()

                # Process window triggers
                await self._process_window_triggers()

                # Clean up expired state
                await self._cleanup_expired_state()

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        finally:
            self.running = False

    async def _process_event_through_operators(self, event: TransactionEvent, message):
        """Process event through all registered operators"""

        # Update watermarks
        event_time = int(event.timestamp.timestamp() * 1000)
        self._update_watermarks(event_time)

        # Assign to windows
        await self._assign_to_windows(event)

        # Process through operators
        for op_name, operator in self.operators.items():
            try:
                start_time = time.time()

                # Execute operator function
                result = await operator["function"](event, operator["state"])

                # Track latency
                latency_ms = (time.time() - start_time) * 1000
                self.metrics["processing_latency_ms"].append(latency_ms)

                # Handle operator output
                if result:
                    await self._handle_operator_output(op_name, result)

            except Exception as e:
                logger.error(f"Error in operator {op_name}: {e}")

    # =====================================================
    # WINDOWING SYSTEM
    # =====================================================

    async def _assign_to_windows(self, event: TransactionEvent):
        """Assign event to appropriate windows"""
        event_time = int(event.timestamp.timestamp() * 1000)

        for window_name, config in self.window_configs.items():
            window_type = config["type"]

            if window_type == "tumbling":
                await self._assign_to_tumbling_window(
                    event, window_name, config, event_time
                )
            elif window_type == "sliding":
                await self._assign_to_sliding_window(
                    event, window_name, config, event_time
                )
            elif window_type == "session":
                await self._assign_to_session_window(
                    event, window_name, config, event_time
                )

    async def _assign_to_tumbling_window(
        self, event: TransactionEvent, window_name: str, config: Dict, event_time: int
    ):
        """Assign event to tumbling window"""
        window_size = config["size_ms"]
        window_start = (event_time // window_size) * window_size
        window_end = window_start + window_size

        window_id = f"{window_name}_{window_start}_{window_end}"

        if window_id not in self.tumbling_windows:
            self.tumbling_windows[window_id] = WindowState(
                window_id=window_id,
                window_type="tumbling",
                start_time=datetime.fromtimestamp(window_start / 1000, timezone.utc),
                end_time=datetime.fromtimestamp(window_end / 1000, timezone.utc),
                events=[],
                aggregates={},
                state={},
            )

        self.tumbling_windows[window_id].events.append(event)

    async def _assign_to_sliding_window(
        self, event: TransactionEvent, window_name: str, config: Dict, event_time: int
    ):
        """Assign event to sliding windows"""
        window_size = config["size_ms"]
        slide_size = config["slide_ms"]

        # Calculate all windows this event belongs to
        latest_window_start = (event_time // slide_size) * slide_size
        earliest_window_start = latest_window_start - window_size + slide_size

        window_start = earliest_window_start
        while window_start <= latest_window_start:
            window_end = window_start + window_size

            # Check if event falls within this window
            if window_start <= event_time < window_end:
                window_id = f"{window_name}_{window_start}_{window_end}"

                if window_id not in self.sliding_windows:
                    self.sliding_windows[window_id] = WindowState(
                        window_id=window_id,
                        window_type="sliding",
                        start_time=datetime.fromtimestamp(
                            window_start / 1000, timezone.utc
                        ),
                        end_time=datetime.fromtimestamp(
                            window_end / 1000, timezone.utc
                        ),
                        events=[],
                        aggregates={},
                        state={},
                    )

                self.sliding_windows[window_id].events.append(event)

            window_start += slide_size

    async def _assign_to_session_window(
        self, event: TransactionEvent, window_name: str, config: Dict, event_time: int
    ):
        """Assign event to session window (gap-based)"""
        gap_ms = config["gap_ms"]
        session_key = f"{event.user_id}_{window_name}"

        # Find existing session window
        existing_window = None
        for window_id, window in self.session_windows.items():
            if window_id.startswith(session_key):
                # Check if event is within gap of last event
                last_event_time = max(
                    int(e.timestamp.timestamp() * 1000) for e in window.events
                )
                if event_time - last_event_time <= gap_ms:
                    existing_window = window
                    break

        if existing_window:
            # Add to existing session
            existing_window.events.append(event)
            existing_window.end_time = event.timestamp + timedelta(milliseconds=gap_ms)
        else:
            # Create new session window
            window_id = f"{session_key}_{event_time}"
            self.session_windows[window_id] = WindowState(
                window_id=window_id,
                window_type="session",
                start_time=event.timestamp,
                end_time=event.timestamp + timedelta(milliseconds=gap_ms),
                events=[event],
                aggregates={},
                state={},
            )

    async def _process_window_triggers(self):
        """Process window triggers and compute aggregates"""
        current_time = int(time.time() * 1000)

        # Process tumbling windows
        for window_id, window in list(self.tumbling_windows.items()):
            window_end = int(window.end_time.timestamp() * 1000)
            if current_time >= window_end:
                await self._trigger_window_computation(window)
                del self.tumbling_windows[window_id]

        # Process sliding windows
        for window_id, window in list(self.sliding_windows.items()):
            window_end = int(window.end_time.timestamp() * 1000)
            if current_time >= window_end:
                await self._trigger_window_computation(window)
                del self.sliding_windows[window_id]

        # Process session windows (based on inactivity)
        for window_id, window in list(self.session_windows.items()):
            if window.end_time and current_time >= int(
                window.end_time.timestamp() * 1000
            ):
                await self._trigger_window_computation(window)
                del self.session_windows[window_id]

    async def _trigger_window_computation(self, window: WindowState):
        """Trigger computation for a completed window"""
        try:
            if not window.events:
                return

            # Compute window aggregates
            aggregates = await self._compute_window_aggregates(window)
            window.aggregates = aggregates

            # Generate features from window
            features = await self._extract_window_features(window)

            # Check for fraud patterns
            if features:
                await self._evaluate_fraud_risk(window, features)

            logger.debug(
                f"Processed window {window.window_id} with {len(window.events)} events"
            )

        except Exception as e:
            logger.error(f"Error processing window {window.window_id}: {e}")

    # =====================================================
    # FEATURE COMPUTATION
    # =====================================================

    async def _compute_window_aggregates(self, window: WindowState) -> Dict[str, Any]:
        """Compute aggregates for a window"""
        events = window.events
        aggregates = {}

        if not events:
            return aggregates

        # Basic aggregates
        amounts = [e.amount for e in events]
        aggregates.update(
            {
                "event_count": len(events),
                "total_amount": sum(amounts),
                "avg_amount": np.mean(amounts),
                "min_amount": min(amounts),
                "max_amount": max(amounts),
                "std_amount": np.std(amounts) if len(amounts) > 1 else 0,
                "unique_users": len(set(e.user_id for e in events)),
                "unique_merchants": len(set(e.merchant_id for e in events)),
            }
        )

        # Time-based aggregates
        timestamps = [e.timestamp for e in events]
        if len(timestamps) > 1:
            time_diffs = [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]
            aggregates.update(
                {
                    "avg_time_between_events": np.mean(time_diffs),
                    "min_time_between_events": min(time_diffs),
                    "max_time_between_events": max(time_diffs),
                }
            )

        # Channel and type distributions
        channels = [e.channel for e in events]
        transaction_types = [e.transaction_type for e in events]

        aggregates.update(
            {
                "channel_distribution": {
                    channel: channels.count(channel) for channel in set(channels)
                },
                "type_distribution": {
                    ttype: transaction_types.count(ttype)
                    for ttype in set(transaction_types)
                },
            }
        )

        return aggregates

    async def _extract_window_features(
        self, window: WindowState
    ) -> Optional[FeatureVector]:
        """Extract features from window aggregates"""
        if not window.events:
            return None

        aggregates = window.aggregates
        features = {}

        # Basic statistical features
        features.update(
            {
                "window_event_count": aggregates.get("event_count", 0),
                "window_total_amount": aggregates.get("total_amount", 0),
                "window_avg_amount": aggregates.get("avg_amount", 0),
                "window_amount_std": aggregates.get("std_amount", 0),
                "window_unique_users": aggregates.get("unique_users", 0),
                "window_unique_merchants": aggregates.get("unique_merchants", 0),
            }
        )

        # Velocity features
        window_duration_minutes = (
            window.end_time - window.start_time
        ).total_seconds() / 60
        if window_duration_minutes > 0:
            features.update(
                {
                    "transactions_per_minute": aggregates.get("event_count", 0)
                    / window_duration_minutes,
                    "amount_per_minute": aggregates.get("total_amount", 0)
                    / window_duration_minutes,
                }
            )

        # Risk indicators
        avg_amount = aggregates.get("avg_amount", 0)
        max_amount = aggregates.get("max_amount", 0)
        if avg_amount > 0:
            features["amount_deviation_ratio"] = max_amount / avg_amount

        # Channel diversity
        channel_dist = aggregates.get("channel_distribution", {})
        features["channel_diversity"] = len(channel_dist)
        if len(channel_dist) > 1:
            # Calculate entropy for channel distribution
            total_txns = sum(channel_dist.values())
            entropy = -sum(
                (count / total_txns) * np.log2(count / total_txns)
                for count in channel_dist.values()
                if count > 0
            )
            features["channel_entropy"] = entropy

        # Time pattern features
        if "avg_time_between_events" in aggregates:
            features.update(
                {
                    "avg_time_between_events": aggregates["avg_time_between_events"],
                    "time_regularity": 1.0 / (1.0 + aggregates.get("std_amount", 1.0)),
                }
            )

        # Use the first event's user_id as representative
        representative_event = window.events[0]

        return FeatureVector(
            transaction_id=f"window_{window.window_id}",
            user_id=representative_event.user_id,
            features=features,
            computed_at=datetime.now(timezone.utc),
            feature_version="window_1.0",
        )

    # =====================================================
    # FRAUD DETECTION & ALERTING
    # =====================================================

    async def _evaluate_fraud_risk(self, window: WindowState, features: FeatureVector):
        """Evaluate fraud risk for window features"""
        try:
            risk_score = await self._calculate_risk_score(features)

            # Generate alert if risk is high
            if risk_score > 0.7:  # High risk threshold
                alert = FraudAlert(
                    alert_id=str(uuid.uuid4()),
                    transaction_id=f"window_{window.window_id}",
                    user_id=features.user_id,
                    risk_score=risk_score,
                    risk_level=self._determine_risk_level(risk_score),
                    features_used=list(features.features.keys()),
                    model_version="window_risk_1.0",
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "window_type": window.window_type,
                        "window_events": len(window.events),
                        "window_duration_minutes": (
                            window.end_time - window.start_time
                        ).total_seconds()
                        / 60,
                    },
                )

                # Send alert
                await self._send_fraud_alert(alert)
                self.metrics["alerts_generated"] += 1

            # Store features
            await self._store_features(features)
            self.metrics["features_computed"] += 1

        except Exception as e:
            logger.error(f"Error evaluating fraud risk: {e}")

    async def _calculate_risk_score(self, features: FeatureVector) -> float:
        """Calculate risk score based on features"""
        # Simplified rule-based risk scoring
        score = 0.0
        feature_dict = features.features

        # High transaction velocity
        if feature_dict.get("transactions_per_minute", 0) > 10:
            score += 0.3

        # High amount velocity
        if feature_dict.get("amount_per_minute", 0) > 10000:
            score += 0.2

        # Unusual amount patterns
        if feature_dict.get("amount_deviation_ratio", 1.0) > 5:
            score += 0.2

        # High channel diversity (potential testing)
        if feature_dict.get("channel_diversity", 1) > 3:
            score += 0.15

        # Rapid consecutive transactions
        if feature_dict.get("avg_time_between_events", 60) < 5:
            score += 0.15

        return min(score, 1.0)

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        elif risk_score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    # =====================================================
    # EXACTLY-ONCE PROCESSING
    # =====================================================

    def _is_duplicate(self, message) -> bool:
        """Check if message has already been processed"""
        message_id = f"{message.topic}_{message.partition}_{message.offset}"

        if message_id in self.processed_offsets:
            logger.debug(f"Duplicate message detected: {message_id}")
            return True

        self.processed_offsets.add(message_id)
        return False

    async def _create_checkpoint(self):
        """Create checkpoint for exactly-once processing"""
        try:
            checkpoint_id = str(uuid.uuid4())
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processed_offsets": list(self.processed_offsets),
                "operator_states": {
                    name: op["state"] for name, op in self.operators.items()
                },
                "watermarks": self.watermarks,
                "metrics": dict(self.metrics),
            }

            # Store checkpoint in Redis
            await self.redis_client.setex(
                f"checkpoint:{checkpoint_id}",
                3600,  # 1 hour TTL
                json.dumps(checkpoint_data, default=str),
            )

            self.checkpoints[checkpoint_id] = checkpoint_data
            logger.debug(f"Created checkpoint {checkpoint_id}")

        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")

    async def _restore_from_checkpoint(self, checkpoint_id: str):
        """Restore state from checkpoint"""
        try:
            checkpoint_data = await self.redis_client.get(f"checkpoint:{checkpoint_id}")
            if not checkpoint_data:
                raise ValueError(f"Checkpoint {checkpoint_id} not found")

            data = json.loads(checkpoint_data)

            # Restore processed offsets
            self.processed_offsets = set(data["processed_offsets"])

            # Restore operator states
            for op_name, state in data["operator_states"].items():
                if op_name in self.operators:
                    self.operators[op_name]["state"] = state

            # Restore watermarks
            self.watermarks = data["watermarks"]

            logger.info(f"Restored from checkpoint {checkpoint_id}")

        except Exception as e:
            logger.error(f"Error restoring checkpoint: {e}")
            raise

    # =====================================================
    # HELPER METHODS
    # =====================================================

    def _parse_transaction_event(self, message_data: Dict) -> TransactionEvent:
        """Parse Kafka message into TransactionEvent"""
        return TransactionEvent(
            transaction_id=message_data["transaction_id"],
            user_id=message_data["user_id"],
            merchant_id=message_data["merchant_id"],
            amount=float(message_data["amount"]),
            currency=message_data.get("currency", "USD"),
            timestamp=datetime.fromisoformat(message_data["timestamp"]),
            location=message_data.get("location"),
            device_info=message_data.get("device_info"),
            transaction_type=message_data.get("transaction_type", "purchase"),
            channel=message_data.get("channel", "online"),
            metadata=message_data.get("metadata"),
        )

    def _update_watermarks(self, event_time: int):
        """Update watermarks for event time processing"""
        for op_name in self.operators:
            current_watermark = self.watermarks.get(op_name, 0)
            self.watermarks[op_name] = max(current_watermark, event_time)

    async def _handle_operator_output(self, operator_name: str, output: Any):
        """Handle output from operator"""
        # In a real implementation, this would route output to next operator
        # or emit to output streams
        logger.debug(f"Operator {operator_name} output: {output}")

    async def _send_fraud_alert(self, alert: FraudAlert):
        """Send fraud alert to Kafka"""
        try:
            alert_data = asdict(alert)

            # Send to alerts topic
            self.producer.send("fraud-alerts", key=alert.user_id, value=alert_data)

            # Store in Redis for real-time access
            await self.redis_client.setex(
                f"alert:{alert.alert_id}",
                3600,  # 1 hour TTL
                json.dumps(alert_data, default=str),
            )

            logger.info(f"Sent fraud alert {alert.alert_id} for user {alert.user_id}")

        except Exception as e:
            logger.error(f"Error sending fraud alert: {e}")

    async def _store_features(self, features: FeatureVector):
        """Store computed features"""
        try:
            feature_data = asdict(features)

            # Store in Redis with TTL
            await self.redis_client.setex(
                f"features:{features.transaction_id}",
                1800,  # 30 minutes TTL
                json.dumps(feature_data, default=str),
            )

        except Exception as e:
            logger.error(f"Error storing features: {e}")

    async def _cleanup_expired_state(self):
        """Clean up expired state and windows"""
        current_time = datetime.now(timezone.utc)

        # Clean up old processed offsets (keep last 10000)
        if len(self.processed_offsets) > 10000:
            self.processed_offsets = set(list(self.processed_offsets)[-5000:])

        # Clean up old checkpoints (keep last 10)
        if len(self.checkpoints) > 10:
            sorted_checkpoints = sorted(
                self.checkpoints.items(), key=lambda x: x[1]["timestamp"]
            )
            self.checkpoints = dict(sorted_checkpoints[-5:])

    # =====================================================
    # MONITORING & METRICS
    # =====================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        latencies = list(self.metrics["processing_latency_ms"])
        throughputs = list(self.metrics["throughput_events_per_sec"])

        return {
            "events_processed": self.metrics["events_processed"],
            "features_computed": self.metrics["features_computed"],
            "alerts_generated": self.metrics["alerts_generated"],
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "avg_throughput_eps": np.mean(throughputs) if throughputs else 0,
            "active_windows": {
                "tumbling": len(self.tumbling_windows),
                "sliding": len(self.sliding_windows),
                "session": len(self.session_windows),
            },
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.Process().cpu_percent(),
        }

    async def stop(self):
        """Stop stream processing"""
        self.running = False

        if self.consumer:
            self.consumer.close()

        if self.producer:
            self.producer.close()

        if self.redis_client:
            await self.redis_client.close()

        self.executor.shutdown(wait=True)
        logger.info("Stream processor stopped")


# =====================================================
# OPERATOR FUNCTIONS
# =====================================================


async def user_velocity_operator(
    event: TransactionEvent, state: Dict
) -> Optional[Dict]:
    """Operator to track user transaction velocity"""
    user_id = event.user_id
    current_time = int(event.timestamp.timestamp())

    # Initialize user state
    if user_id not in state:
        state[user_id] = {
            "transactions": deque(maxlen=100),
            "last_update": current_time,
        }

    user_state = state[user_id]

    # Add current transaction
    user_state["transactions"].append(
        {
            "amount": event.amount,
            "timestamp": current_time,
            "merchant_id": event.merchant_id,
        }
    )

    # Calculate velocity metrics
    recent_transactions = [
        t
        for t in user_state["transactions"]
        if current_time - t["timestamp"] <= 300  # Last 5 minutes
    ]

    if len(recent_transactions) >= 5:  # Potential velocity fraud
        total_amount = sum(t["amount"] for t in recent_transactions)
        unique_merchants = len(set(t["merchant_id"] for t in recent_transactions))

        return {
            "user_id": user_id,
            "velocity_alert": True,
            "transaction_count_5min": len(recent_transactions),
            "total_amount_5min": total_amount,
            "unique_merchants_5min": unique_merchants,
            "timestamp": current_time,
        }

    return None


async def merchant_anomaly_operator(
    event: TransactionEvent, state: Dict
) -> Optional[Dict]:
    """Operator to detect merchant-level anomalies"""
    merchant_id = event.merchant_id
    current_time = int(event.timestamp.timestamp())

    # Initialize merchant state
    if merchant_id not in state:
        state[merchant_id] = {
            "hourly_stats": defaultdict(lambda: {"count": 0, "total_amount": 0}),
            "baseline": {"avg_count": 0, "avg_amount": 0},
        }

    merchant_state = state[merchant_id]

    # Update hourly stats
    hour_key = current_time // 3600  # Hour bucket
    hourly_stats = merchant_state["hourly_stats"][hour_key]
    hourly_stats["count"] += 1
    hourly_stats["total_amount"] += event.amount

    # Calculate baseline from historical data
    if len(merchant_state["hourly_stats"]) > 24:  # At least 24 hours of data
        recent_hours = sorted(merchant_state["hourly_stats"].items())[-24:]
        avg_count = np.mean([stats["count"] for _, stats in recent_hours])
        avg_amount = np.mean([stats["total_amount"] for _, stats in recent_hours])

        merchant_state["baseline"]["avg_count"] = avg_count
        merchant_state["baseline"]["avg_amount"] = avg_amount

        # Check for anomalies
        current_count = hourly_stats["count"]
        current_amount = hourly_stats["total_amount"]

        if (
            current_count > avg_count * 3  # 3x normal transaction count
            or current_amount > avg_amount * 5
        ):  # 5x normal amount

            return {
                "merchant_id": merchant_id,
                "anomaly_alert": True,
                "current_count": current_count,
                "baseline_count": avg_count,
                "current_amount": current_amount,
                "baseline_amount": avg_amount,
                "timestamp": current_time,
            }

    return None


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage of Flink stream processor"""

    # Configuration
    kafka_config = {"bootstrap_servers": ["localhost:9092"]}

    redis_config = {"host": "localhost", "port": 6379}

    # Initialize processor
    processor = FlinkStreamProcessor(kafka_config, redis_config)
    await processor.initialize()

    # Register operators
    processor.register_operator("user_velocity", user_velocity_operator)
    processor.register_operator("merchant_anomaly", merchant_anomaly_operator)

    try:
        # Start processing
        await processor.process_stream()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await processor.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
