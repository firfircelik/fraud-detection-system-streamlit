#!/usr/bin/env python3
"""
Exactly-Once Processing Semantics for Fraud Detection
Transactional processing, idempotency, and data consistency guarantees
"""

import asyncio
import base64
import hashlib
import json
import logging
import pickle
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TransactionState(Enum):
    """Transaction processing states"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMMITTED = "committed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class ProcessingTransaction:
    """Represents a processing transaction for exactly-once semantics"""

    transaction_id: str
    coordinator_id: str
    participant_ids: List[str]
    state: TransactionState
    start_time: datetime
    timeout_duration: timedelta
    data_checksum: str
    retry_count: int = 0
    max_retries: int = 3
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IdempotencyKey:
    """Idempotency key for duplicate detection"""

    key: str
    request_hash: str
    response_data: Optional[Dict[str, Any]]
    created_at: datetime
    expires_at: datetime
    processing_node: str


@dataclass
class TransactionLog:
    """Transaction log entry for recovery"""

    log_id: str
    transaction_id: str
    operation: str
    data: Dict[str, Any]
    timestamp: datetime
    checksum: str
    node_id: str


@dataclass
class CheckpointBarrier:
    """Checkpoint barrier for coordinated snapshots"""

    barrier_id: str
    checkpoint_id: str
    source_operator: str
    timestamp: datetime
    alignment_timeout: timedelta


class ExactlyOnceProcessor:
    """
    Exactly-once processing coordinator with transactional guarantees
    Implements two-phase commit protocol and idempotency checks
    """

    def __init__(self, node_id: str, coordinator_timeout: int = 30):
        self.node_id = node_id
        self.coordinator_timeout = timedelta(seconds=coordinator_timeout)

        # Transaction coordination
        self.active_transactions: Dict[str, ProcessingTransaction] = {}
        self.completed_transactions: Dict[str, ProcessingTransaction] = {}
        self.transaction_logs: List[TransactionLog] = []

        # Idempotency management
        self.idempotency_cache: Dict[str, IdempotencyKey] = {}
        self.duplicate_requests: Set[str] = set()

        # Checkpoint management
        self.checkpoint_barriers: Dict[str, CheckpointBarrier] = {}
        self.aligned_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.checkpoint_interval = timedelta(minutes=5)
        self.last_checkpoint = datetime.now(timezone.utc)

        # State management
        self.operator_states: Dict[str, Dict[str, Any]] = {}
        self.persistent_state: Dict[str, Any] = {}

        # Recovery and consistency
        self.write_ahead_log: List[Dict[str, Any]] = []
        self.committed_offsets: Dict[str, int] = {}
        self.pending_commits: Dict[str, Dict[str, Any]] = {}

        # Monitoring
        self.metrics = {
            "transactions_started": 0,
            "transactions_committed": 0,
            "transactions_aborted": 0,
            "duplicate_requests_blocked": 0,
            "recovery_operations": 0,
            "checkpoint_operations": 0,
        }

        # Thread safety
        self.lock = threading.RLock()

        # Background tasks
        self.cleanup_task = None
        self.checkpoint_task = None
        self.running = False

    async def start(self):
        """Start exactly-once processor"""
        self.running = True

        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_transactions())
        self.checkpoint_task = asyncio.create_task(self._periodic_checkpoint())

        # Recovery on startup
        await self._recover_from_logs()

        logger.info(f"Exactly-once processor {self.node_id} started")

    async def stop(self):
        """Stop exactly-once processor"""
        self.running = False

        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.checkpoint_task:
            self.checkpoint_task.cancel()

        # Commit any pending transactions
        await self._commit_pending_transactions()

        logger.info(f"Exactly-once processor {self.node_id} stopped")

    # =====================================================
    # TRANSACTION COORDINATION (2PC)
    # =====================================================

    async def begin_transaction(
        self,
        data: Dict[str, Any],
        participant_ids: List[str],
        timeout_seconds: int = 30,
    ) -> str:
        """Begin a new distributed transaction"""

        transaction_id = str(uuid.uuid4())
        data_checksum = self._calculate_checksum(data)

        # Check for idempotency
        idempotency_key = self._generate_idempotency_key(data)
        if await self._is_duplicate_request(idempotency_key):
            existing_response = self.idempotency_cache[idempotency_key].response_data
            logger.info(f"Duplicate transaction detected: {transaction_id}")
            self.metrics["duplicate_requests_blocked"] += 1
            return existing_response["transaction_id"]

        # Create transaction
        transaction = ProcessingTransaction(
            transaction_id=transaction_id,
            coordinator_id=self.node_id,
            participant_ids=participant_ids,
            state=TransactionState.PENDING,
            start_time=datetime.now(timezone.utc),
            timeout_duration=timedelta(seconds=timeout_seconds),
            data_checksum=data_checksum,
            metadata={"original_data": data},
        )

        with self.lock:
            self.active_transactions[transaction_id] = transaction

        # Log transaction start
        await self._log_transaction_operation(
            transaction_id, "BEGIN", data, data_checksum
        )

        # Store idempotency key
        await self._store_idempotency_key(
            idempotency_key, {"transaction_id": transaction_id}
        )

        self.metrics["transactions_started"] += 1
        logger.info(
            f"Started transaction {transaction_id} with {len(participant_ids)} participants"
        )

        return transaction_id

    async def prepare_transaction(self, transaction_id: str) -> bool:
        """Prepare phase of 2PC - ask all participants to prepare"""

        if transaction_id not in self.active_transactions:
            logger.error(f"Transaction {transaction_id} not found")
            return False

        transaction = self.active_transactions[transaction_id]

        # Check timeout
        if self._is_transaction_expired(transaction):
            await self._abort_transaction(transaction_id, "TIMEOUT")
            return False

        transaction.state = TransactionState.PROCESSING

        # Simulate prepare phase - in real implementation would contact participants
        prepare_results = []
        for participant_id in transaction.participant_ids:
            try:
                # Simulate network call to participant
                prepare_success = await self._simulate_participant_prepare(
                    participant_id,
                    transaction_id,
                    transaction.metadata["original_data"],
                )
                prepare_results.append(prepare_success)

            except Exception as e:
                logger.error(f"Prepare failed for participant {participant_id}: {e}")
                prepare_results.append(False)

        # Log prepare results
        await self._log_transaction_operation(
            transaction_id,
            "PREPARE",
            {"prepare_results": prepare_results},
            transaction.data_checksum,
        )

        # All participants must agree to commit
        all_prepared = all(prepare_results)

        if not all_prepared:
            await self._abort_transaction(transaction_id, "PREPARE_FAILED")
            return False

        logger.info(f"Transaction {transaction_id} prepared successfully")
        return True

    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit phase of 2PC - commit on all participants"""

        if transaction_id not in self.active_transactions:
            logger.error(f"Transaction {transaction_id} not found")
            return False

        transaction = self.active_transactions[transaction_id]

        # Check if transaction is in correct state
        if transaction.state != TransactionState.PROCESSING:
            logger.error(f"Transaction {transaction_id} not in PROCESSING state")
            return False

        try:
            # Commit on all participants
            commit_results = []
            for participant_id in transaction.participant_ids:
                try:
                    commit_success = await self._simulate_participant_commit(
                        participant_id, transaction_id
                    )
                    commit_results.append(commit_success)

                except Exception as e:
                    logger.error(f"Commit failed for participant {participant_id}: {e}")
                    commit_results.append(False)

            # Check if all commits succeeded
            all_committed = all(commit_results)

            if all_committed:
                transaction.state = TransactionState.COMMITTED

                # Log successful commit
                await self._log_transaction_operation(
                    transaction_id,
                    "COMMIT",
                    {"commit_results": commit_results},
                    transaction.data_checksum,
                )

                # Move to completed transactions
                with self.lock:
                    self.completed_transactions[transaction_id] = transaction
                    del self.active_transactions[transaction_id]

                self.metrics["transactions_committed"] += 1
                logger.info(f"Transaction {transaction_id} committed successfully")
                return True
            else:
                # Partial commit failure - need compensation
                await self._handle_partial_commit_failure(
                    transaction_id, commit_results
                )
                return False

        except Exception as e:
            logger.error(f"Commit failed for transaction {transaction_id}: {e}")
            await self._abort_transaction(transaction_id, f"COMMIT_ERROR: {e}")
            return False

    async def _abort_transaction(self, transaction_id: str, reason: str):
        """Abort transaction and clean up"""

        if transaction_id not in self.active_transactions:
            return

        transaction = self.active_transactions[transaction_id]
        transaction.state = TransactionState.ABORTED

        # Abort on all participants
        for participant_id in transaction.participant_ids:
            try:
                await self._simulate_participant_abort(participant_id, transaction_id)
            except Exception as e:
                logger.error(f"Abort failed for participant {participant_id}: {e}")

        # Log abort
        await self._log_transaction_operation(
            transaction_id, "ABORT", {"reason": reason}, transaction.data_checksum
        )

        # Clean up
        with self.lock:
            del self.active_transactions[transaction_id]

        self.metrics["transactions_aborted"] += 1
        logger.info(f"Transaction {transaction_id} aborted: {reason}")

    # =====================================================
    # IDEMPOTENCY MANAGEMENT
    # =====================================================

    def _generate_idempotency_key(self, data: Dict[str, Any]) -> str:
        """Generate idempotency key from request data"""
        # Create deterministic hash from request data
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    async def _is_duplicate_request(self, idempotency_key: str) -> bool:
        """Check if request is a duplicate"""
        if idempotency_key not in self.idempotency_cache:
            return False

        cached_key = self.idempotency_cache[idempotency_key]

        # Check if key has expired
        if datetime.now(timezone.utc) > cached_key.expires_at:
            del self.idempotency_cache[idempotency_key]
            return False

        return True

    async def _store_idempotency_key(self, key: str, response_data: Dict[str, Any]):
        """Store idempotency key with response"""

        idempotency_key = IdempotencyKey(
            key=key,
            request_hash=key,
            response_data=response_data,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            processing_node=self.node_id,
        )

        self.idempotency_cache[key] = idempotency_key

    # =====================================================
    # CHECKPOINT COORDINATION
    # =====================================================

    async def initiate_checkpoint(self, checkpoint_id: str) -> bool:
        """Initiate coordinated checkpoint across all operators"""

        barrier_id = f"barrier_{checkpoint_id}_{int(time.time())}"

        # Create checkpoint barrier
        barrier = CheckpointBarrier(
            barrier_id=barrier_id,
            checkpoint_id=checkpoint_id,
            source_operator=self.node_id,
            timestamp=datetime.now(timezone.utc),
            alignment_timeout=timedelta(seconds=30),
        )

        self.checkpoint_barriers[barrier_id] = barrier

        # Broadcast barrier to all operators (simulated)
        operators = list(self.operator_states.keys())
        alignment_results = []

        for operator_id in operators:
            try:
                aligned = await self._align_operator_checkpoint(operator_id, barrier)
                alignment_results.append(aligned)
            except Exception as e:
                logger.error(f"Checkpoint alignment failed for {operator_id}: {e}")
                alignment_results.append(False)

        # Check if all operators aligned
        if all(alignment_results):
            await self._complete_checkpoint(checkpoint_id, barrier_id)
            return True
        else:
            await self._abort_checkpoint(checkpoint_id, barrier_id)
            return False

    async def _align_operator_checkpoint(
        self, operator_id: str, barrier: CheckpointBarrier
    ) -> bool:
        """Align operator for checkpoint"""

        # Wait for operator to process all records before barrier
        # In real implementation, this would involve complex stream alignment

        # Simulate operator state snapshot
        operator_state = self.operator_states.get(operator_id, {})

        # Create state snapshot
        snapshot = {
            "operator_id": operator_id,
            "checkpoint_id": barrier.checkpoint_id,
            "state": operator_state.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "barrier_id": barrier.barrier_id,
        }

        # Store aligned snapshot
        if barrier.checkpoint_id not in self.aligned_checkpoints:
            self.aligned_checkpoints[barrier.checkpoint_id] = {}

        self.aligned_checkpoints[barrier.checkpoint_id][operator_id] = snapshot

        logger.debug(
            f"Operator {operator_id} aligned for checkpoint {barrier.checkpoint_id}"
        )
        return True

    async def _complete_checkpoint(self, checkpoint_id: str, barrier_id: str):
        """Complete coordinated checkpoint"""

        # Combine all operator snapshots
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "coordinator_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operator_snapshots": self.aligned_checkpoints.get(checkpoint_id, {}),
            "global_state": {
                "committed_offsets": self.committed_offsets.copy(),
                "completed_transactions": {
                    tid: asdict(txn) for tid, txn in self.completed_transactions.items()
                },
                "metrics": self.metrics.copy(),
            },
        }

        # Persist checkpoint
        await self._persist_checkpoint(checkpoint_id, checkpoint_data)

        # Clean up
        if barrier_id in self.checkpoint_barriers:
            del self.checkpoint_barriers[barrier_id]
        if checkpoint_id in self.aligned_checkpoints:
            del self.aligned_checkpoints[checkpoint_id]

        self.last_checkpoint = datetime.now(timezone.utc)
        self.metrics["checkpoint_operations"] += 1

        logger.info(f"Checkpoint {checkpoint_id} completed successfully")

    async def _abort_checkpoint(self, checkpoint_id: str, barrier_id: str):
        """Abort checkpoint due to alignment failure"""

        # Clean up partial checkpoint data
        if barrier_id in self.checkpoint_barriers:
            del self.checkpoint_barriers[barrier_id]
        if checkpoint_id in self.aligned_checkpoints:
            del self.aligned_checkpoints[checkpoint_id]

        logger.warning(f"Checkpoint {checkpoint_id} aborted due to alignment failure")

    # =====================================================
    # WRITE-AHEAD LOGGING & RECOVERY
    # =====================================================

    async def _log_transaction_operation(
        self, transaction_id: str, operation: str, data: Dict[str, Any], checksum: str
    ):
        """Log transaction operation for recovery"""

        log_entry = TransactionLog(
            log_id=str(uuid.uuid4()),
            transaction_id=transaction_id,
            operation=operation,
            data=data,
            timestamp=datetime.now(timezone.utc),
            checksum=checksum,
            node_id=self.node_id,
        )

        # Append to write-ahead log
        self.write_ahead_log.append(asdict(log_entry))
        self.transaction_logs.append(log_entry)

        # Persist log entry (in real implementation, would write to durable storage)
        await self._persist_log_entry(log_entry)

    async def _recover_from_logs(self):
        """Recover state from transaction logs"""

        try:
            # Load persisted logs (simulated)
            logs = await self._load_persisted_logs()

            # Replay logs to recover state
            for log_data in logs:
                log_entry = TransactionLog(**log_data)
                await self._replay_log_entry(log_entry)

            self.metrics["recovery_operations"] += 1
            logger.info(f"Recovered from {len(logs)} log entries")

        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    async def _replay_log_entry(self, log_entry: TransactionLog):
        """Replay single log entry for recovery"""

        transaction_id = log_entry.transaction_id
        operation = log_entry.operation
        data = log_entry.data

        if operation == "BEGIN":
            # Recreate pending transaction
            if transaction_id not in self.completed_transactions:
                # Transaction was not completed - recreate as pending
                transaction = ProcessingTransaction(
                    transaction_id=transaction_id,
                    coordinator_id=log_entry.node_id,
                    participant_ids=data.get("participant_ids", []),
                    state=TransactionState.PENDING,
                    start_time=log_entry.timestamp,
                    timeout_duration=timedelta(seconds=30),
                    data_checksum=log_entry.checksum,
                    metadata=data,
                )
                self.active_transactions[transaction_id] = transaction

        elif operation == "COMMIT":
            # Mark transaction as committed
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
                transaction.state = TransactionState.COMMITTED
                self.completed_transactions[transaction_id] = transaction
                del self.active_transactions[transaction_id]

        elif operation == "ABORT":
            # Remove aborted transaction
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]

    # =====================================================
    # SIMULATION METHODS (would be real network calls)
    # =====================================================

    async def _simulate_participant_prepare(
        self, participant_id: str, transaction_id: str, data: Dict[str, Any]
    ) -> bool:
        """Simulate participant prepare phase"""
        # Simulate network latency
        await asyncio.sleep(0.01)

        # Simulate prepare success (90% success rate)
        import random

        return random.random() > 0.1

    async def _simulate_participant_commit(
        self, participant_id: str, transaction_id: str
    ) -> bool:
        """Simulate participant commit phase"""
        await asyncio.sleep(0.01)

        # Simulate commit success (95% success rate)
        import random

        return random.random() > 0.05

    async def _simulate_participant_abort(
        self, participant_id: str, transaction_id: str
    ):
        """Simulate participant abort"""
        await asyncio.sleep(0.01)
        logger.debug(
            f"Participant {participant_id} aborted transaction {transaction_id}"
        )

    # =====================================================
    # PERSISTENCE (simulated - would use real storage)
    # =====================================================

    async def _persist_checkpoint(self, checkpoint_id: str, data: Dict[str, Any]):
        """Persist checkpoint data"""
        # In real implementation, would write to distributed storage
        checkpoint_file = f"checkpoint_{checkpoint_id}.json"

        # Simulate persistence
        logger.debug(f"Persisted checkpoint {checkpoint_id}")

    async def _persist_log_entry(self, log_entry: TransactionLog):
        """Persist log entry"""
        # In real implementation, would append to durable WAL
        logger.debug(f"Persisted log entry {log_entry.log_id}")

    async def _load_persisted_logs(self) -> List[Dict[str, Any]]:
        """Load persisted transaction logs"""
        # In real implementation, would read from storage
        return []

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for data integrity"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _is_transaction_expired(self, transaction: ProcessingTransaction) -> bool:
        """Check if transaction has expired"""
        return (
            datetime.now(timezone.utc) - transaction.start_time
        ) > transaction.timeout_duration

    async def _handle_partial_commit_failure(
        self, transaction_id: str, commit_results: List[bool]
    ):
        """Handle partial commit failure with compensation"""

        transaction = self.active_transactions[transaction_id]

        # Identify failed participants
        failed_participants = [
            pid
            for i, pid in enumerate(transaction.participant_ids)
            if not commit_results[i]
        ]

        logger.error(
            f"Partial commit failure for transaction {transaction_id}. "
            f"Failed participants: {failed_participants}"
        )

        # Attempt compensation (rollback committed participants)
        for i, participant_id in enumerate(transaction.participant_ids):
            if commit_results[i]:  # Successfully committed
                try:
                    await self._simulate_participant_abort(
                        participant_id, transaction_id
                    )
                except Exception as e:
                    logger.error(f"Compensation failed for {participant_id}: {e}")

        await self._abort_transaction(transaction_id, "PARTIAL_COMMIT_FAILURE")

    # =====================================================
    # BACKGROUND TASKS
    # =====================================================

    async def _cleanup_expired_transactions(self):
        """Background task to clean up expired transactions"""

        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                expired_transactions = []

                with self.lock:
                    for transaction_id, transaction in self.active_transactions.items():
                        if self._is_transaction_expired(transaction):
                            expired_transactions.append(transaction_id)

                # Abort expired transactions
                for transaction_id in expired_transactions:
                    await self._abort_transaction(transaction_id, "EXPIRED")

                # Clean up old idempotency keys
                expired_keys = []
                for key, idempotency_key in self.idempotency_cache.items():
                    if current_time > idempotency_key.expires_at:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.idempotency_cache[key]

                # Clean up old completed transactions (keep last 1000)
                if len(self.completed_transactions) > 1000:
                    sorted_transactions = sorted(
                        self.completed_transactions.items(),
                        key=lambda x: x[1].start_time,
                    )
                    self.completed_transactions = dict(sorted_transactions[-500:])

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(30)

    async def _periodic_checkpoint(self):
        """Background task for periodic checkpointing"""

        while self.running:
            try:
                # Check if it's time for checkpoint
                if (
                    datetime.now(timezone.utc) - self.last_checkpoint
                ) > self.checkpoint_interval:
                    checkpoint_id = f"auto_{int(time.time())}"
                    await self.initiate_checkpoint(checkpoint_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Checkpoint task error: {e}")
                await asyncio.sleep(60)

    async def _commit_pending_transactions(self):
        """Commit any pending transactions on shutdown"""

        pending_transactions = list(self.active_transactions.keys())

        for transaction_id in pending_transactions:
            try:
                # Attempt to commit pending transaction
                success = await self.prepare_transaction(transaction_id)
                if success:
                    await self.commit_transaction(transaction_id)
                else:
                    await self._abort_transaction(transaction_id, "SHUTDOWN")
            except Exception as e:
                logger.error(
                    f"Error committing pending transaction {transaction_id}: {e}"
                )

    # =====================================================
    # MONITORING & METRICS
    # =====================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current processor status"""

        return {
            "node_id": self.node_id,
            "running": self.running,
            "active_transactions": len(self.active_transactions),
            "completed_transactions": len(self.completed_transactions),
            "idempotency_cache_size": len(self.idempotency_cache),
            "checkpoint_barriers": len(self.checkpoint_barriers),
            "last_checkpoint": (
                self.last_checkpoint.isoformat() if self.last_checkpoint else None
            ),
            "metrics": self.metrics,
            "write_ahead_log_size": len(self.write_ahead_log),
        }


# =====================================================
# HIGH-LEVEL API
# =====================================================


class ExactlyOnceAPI:
    """High-level API for exactly-once processing"""

    def __init__(self, node_id: str = None):
        if node_id is None:
            node_id = f"node_{uuid.uuid4().hex[:8]}"

        self.processor = ExactlyOnceProcessor(node_id)

    async def start(self):
        """Start the exactly-once processor"""
        await self.processor.start()

    async def stop(self):
        """Stop the exactly-once processor"""
        await self.processor.stop()

    async def process_with_exactly_once(
        self, data: Dict[str, Any], participants: List[str]
    ) -> Dict[str, Any]:
        """Process data with exactly-once guarantees"""

        # Begin transaction
        transaction_id = await self.processor.begin_transaction(data, participants)

        try:
            # Prepare phase
            prepared = await self.processor.prepare_transaction(transaction_id)
            if not prepared:
                return {"success": False, "error": "Prepare phase failed"}

            # Commit phase
            committed = await self.processor.commit_transaction(transaction_id)
            if not committed:
                return {"success": False, "error": "Commit phase failed"}

            return {
                "success": True,
                "transaction_id": transaction_id,
                "message": "Processing completed with exactly-once guarantees",
            }

        except Exception as e:
            logger.error(f"Exactly-once processing failed: {e}")
            return {"success": False, "error": str(e)}


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage of exactly-once processor"""

    # Initialize API
    api = ExactlyOnceAPI("fraud-processor-1")
    await api.start()

    try:
        # Process some fraud detection data
        fraud_data = {
            "transaction_id": "txn_12345",
            "user_id": "user_789",
            "amount": 1500.00,
            "risk_score": 0.85,
        }

        participants = ["ml-model-server", "rule-engine", "alert-service"]

        # Process with exactly-once guarantees
        result = await api.process_with_exactly_once(fraud_data, participants)
        print(f"Processing result: {result}")

        # Check processor status
        status = api.processor.get_status()
        print(f"Processor status: {status}")

    finally:
        await api.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
