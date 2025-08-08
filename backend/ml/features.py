#!/usr/bin/env python3
"""
🏪 High-Performance Feature Store
Real-time feature computation and caching system
"""

import hashlib
import json
import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import redis

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Feature metadata for versioning and tracking"""

    feature_name: str
    feature_type: str
    computation_time_ms: float
    version: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    quality_score: float = 1.0


class HighPerformanceFeatureStore:
    """High-performance feature store with Redis caching"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.Redis(decode_responses=True)
        self.feature_cache = {}
        self.feature_metadata = {}
        self.computation_stats = {}
        self.cache_hit_rate = 0.0
        self.total_requests = 0
        self.cache_hits = 0
        self.lock = threading.Lock()

    def get_features(
        self,
        transaction_id: str,
        user_id: str,
        merchant_id: str,
        transaction_data: Dict,
    ) -> Dict[str, float]:
        """Get comprehensive features for a transaction"""

        start_time = time.time()

        # Create cache key
        cache_key = self._create_cache_key(
            transaction_id, user_id, merchant_id, transaction_data
        )

        # Check cache first
        cached_features = self._get_from_cache(cache_key)
        if cached_features:
            self._update_cache_stats(True)
            return cached_features

        # Compute features
        features = self._compute_all_features(user_id, merchant_id, transaction_data)

        # Cache the results
        self._store_in_cache(cache_key, features, ttl=3600)  # 1 hour TTL

        # Update stats
        self._update_cache_stats(False)
        computation_time = (time.time() - start_time) * 1000
        self._update_computation_stats("all_features", computation_time)

        return features

    def _create_cache_key(
        self,
        transaction_id: str,
        user_id: str,
        merchant_id: str,
        transaction_data: Dict,
    ) -> str:
        """Create unique cache key for feature set"""

        # Create hash from key components
        key_data = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "merchant_id": merchant_id,
            "amount": transaction_data.get("amount", 0),
            "timestamp": transaction_data.get("timestamp", ""),
            "hour": datetime.now().hour,  # Include hour for temporal features
        }

        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"features:{key_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Get features from cache"""

        try:
            # Try Redis first
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)

            # Try local cache
            if cache_key in self.feature_cache:
                cache_entry = self.feature_cache[cache_key]
                if cache_entry["expires_at"] > datetime.now():
                    return cache_entry["features"]
                else:
                    # Remove expired entry
                    del self.feature_cache[cache_key]

        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")

        return None

    def _store_in_cache(
        self, cache_key: str, features: Dict[str, float], ttl: int = 3600
    ):
        """Store features in cache"""

        try:
            # Store in Redis
            self.redis_client.setex(cache_key, ttl, json.dumps(features))

            # Store in local cache as backup
            expires_at = datetime.now() + timedelta(seconds=ttl)
            self.feature_cache[cache_key] = {
                "features": features,
                "expires_at": expires_at,
            }

        except Exception as e:
            logger.error(f"Cache storage failed: {e}")

    def _compute_all_features(
        self, user_id: str, merchant_id: str, transaction_data: Dict
    ) -> Dict[str, float]:
        """Compute all features for a transaction"""

        features = {}

        # Compute different feature groups in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit feature computation tasks
            temporal_future = executor.submit(
                self._compute_temporal_features, transaction_data
            )
            velocity_future = executor.submit(
                self._compute_velocity_features, user_id, transaction_data
            )
            behavioral_future = executor.submit(
                self._compute_behavioral_features, user_id, transaction_data
            )
            merchant_future = executor.submit(
                self._compute_merchant_features, merchant_id, transaction_data
            )

            # Collect results
            features.update(temporal_future.result())
            features.update(velocity_future.result())
            features.update(behavioral_future.result())
            features.update(merchant_future.result())

        # Add basic transaction features
        features.update(self._compute_transaction_features(transaction_data))

        return features

    def _compute_temporal_features(self, transaction_data: Dict) -> Dict[str, float]:
        """Compute temporal features"""

        start_time = time.time()
        features = {}

        try:
            timestamp = transaction_data.get("timestamp", datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Basic temporal features
            features["hour"] = float(timestamp.hour)
            features["day_of_week"] = float(timestamp.weekday())
            features["day_of_month"] = float(timestamp.day)
            features["month"] = float(timestamp.month)
            features["quarter"] = float((timestamp.month - 1) // 3 + 1)

            # Advanced temporal features
            features["is_weekend"] = float(timestamp.weekday() >= 5)
            features["is_business_hour"] = float(9 <= timestamp.hour <= 17)
            features["is_night_time"] = float(timestamp.hour < 6 or timestamp.hour > 22)
            features["is_lunch_time"] = float(11 <= timestamp.hour <= 14)

            # Cyclical encoding
            features["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
            features["day_sin"] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            features["day_cos"] = np.cos(2 * np.pi * timestamp.weekday() / 7)
            features["month_sin"] = np.sin(2 * np.pi * timestamp.month / 12)
            features["month_cos"] = np.cos(2 * np.pi * timestamp.month / 12)

            # Time since epoch features
            features["days_since_epoch"] = float(
                (timestamp - datetime(1970, 1, 1)).days
            )
            features["seconds_since_midnight"] = float(
                timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
            )

        except Exception as e:
            logger.error(f"Temporal feature computation failed: {e}")

        computation_time = (time.time() - start_time) * 1000
        self._update_computation_stats("temporal_features", computation_time)

        return features

    def _compute_velocity_features(
        self, user_id: str, transaction_data: Dict
    ) -> Dict[str, float]:
        """Compute velocity features"""

        start_time = time.time()
        features = {}

        try:
            # Simulate velocity calculations (in production, query database)
            current_time = datetime.now()

            # Transaction counts in different time windows
            features["user_tx_count_1h"] = float(np.random.poisson(2))  # Simulate
            features["user_tx_count_6h"] = float(np.random.poisson(8))
            features["user_tx_count_24h"] = float(np.random.poisson(15))
            features["user_tx_count_7d"] = float(np.random.poisson(50))
            features["user_tx_count_30d"] = float(np.random.poisson(200))

            # Amount-based velocity
            amount = transaction_data.get("amount", 0)
            features["user_amount_1h"] = float(amount * np.random.uniform(0.5, 2.0))
            features["user_amount_24h"] = float(amount * np.random.uniform(2.0, 10.0))
            features["user_amount_7d"] = float(amount * np.random.uniform(10.0, 50.0))

            # Velocity ratios
            features["tx_velocity_1h"] = features["user_tx_count_1h"] / 1.0  # per hour
            features["tx_velocity_24h"] = features["user_tx_count_24h"] / 24.0
            features["amount_velocity_1h"] = features["user_amount_1h"] / 1.0
            features["amount_velocity_24h"] = features["user_amount_24h"] / 24.0

            # Rapid fire detection
            features["time_since_last_tx"] = float(
                np.random.exponential(300)
            )  # seconds
            features["is_rapid_fire"] = float(features["time_since_last_tx"] < 60)

        except Exception as e:
            logger.error(f"Velocity feature computation failed: {e}")

        computation_time = (time.time() - start_time) * 1000
        self._update_computation_stats("velocity_features", computation_time)

        return features

    def _compute_behavioral_features(
        self, user_id: str, transaction_data: Dict
    ) -> Dict[str, float]:
        """Compute behavioral features"""

        start_time = time.time()
        features = {}

        try:
            amount = transaction_data.get("amount", 0)

            # User spending patterns (simulated)
            features["user_avg_amount"] = float(amount * np.random.uniform(0.8, 1.2))
            features["user_std_amount"] = float(amount * np.random.uniform(0.1, 0.5))
            features["user_max_amount"] = float(amount * np.random.uniform(1.5, 3.0))
            features["user_min_amount"] = float(amount * np.random.uniform(0.1, 0.5))

            # Amount deviation features
            if features["user_std_amount"] > 0:
                features["amount_zscore"] = (
                    amount - features["user_avg_amount"]
                ) / features["user_std_amount"]
            else:
                features["amount_zscore"] = 0.0

            features["amount_deviation"] = abs(features["amount_zscore"])
            features["is_amount_outlier"] = float(features["amount_deviation"] > 2.0)

            # Merchant diversity
            features["user_merchant_count_7d"] = float(np.random.poisson(5))
            features["user_merchant_count_30d"] = float(np.random.poisson(15))
            features["merchant_diversity_7d"] = features[
                "user_merchant_count_7d"
            ] / max(1, features["user_tx_count_7d"])

            # Category diversity (if available)
            category = transaction_data.get("category", "unknown")
            features["user_category_count_7d"] = float(np.random.poisson(3))
            features["category_diversity_7d"] = features[
                "user_category_count_7d"
            ] / max(1, features["user_tx_count_7d"])

            # Spending consistency
            features["spending_consistency"] = 1.0 / (
                1.0 + features["user_std_amount"] / max(1, features["user_avg_amount"])
            )

        except Exception as e:
            logger.error(f"Behavioral feature computation failed: {e}")

        computation_time = (time.time() - start_time) * 1000
        self._update_computation_stats("behavioral_features", computation_time)

        return features

    def _compute_merchant_features(
        self, merchant_id: str, transaction_data: Dict
    ) -> Dict[str, float]:
        """Compute merchant-related features"""

        start_time = time.time()
        features = {}

        try:
            # Merchant risk scoring (simulated)
            merchant_hash = hash(merchant_id) % 1000
            features["merchant_risk_score"] = float(merchant_hash / 1000.0)

            # Merchant transaction patterns
            features["merchant_tx_count_1h"] = float(np.random.poisson(10))
            features["merchant_tx_count_24h"] = float(np.random.poisson(100))
            features["merchant_avg_amount"] = float(
                transaction_data.get("amount", 0) * np.random.uniform(0.5, 2.0)
            )

            # Merchant fraud rates
            features["merchant_fraud_rate_7d"] = float(
                np.random.beta(1, 20)
            )  # Low fraud rate
            features["merchant_fraud_rate_30d"] = float(np.random.beta(1, 15))

            # Merchant category risk
            category = transaction_data.get("category", "unknown")
            high_risk_categories = ["gambling", "crypto", "adult", "forex"]
            features["is_high_risk_category"] = float(
                category.lower() in high_risk_categories
            )

            # Merchant age and reputation
            features["merchant_age_days"] = float(
                np.random.exponential(365)
            )  # Average 1 year
            features["merchant_reputation_score"] = float(
                np.random.beta(8, 2)
            )  # Generally good

        except Exception as e:
            logger.error(f"Merchant feature computation failed: {e}")

        computation_time = (time.time() - start_time) * 1000
        self._update_computation_stats("merchant_features", computation_time)

        return features

    def _compute_transaction_features(self, transaction_data: Dict) -> Dict[str, float]:
        """Compute basic transaction features"""

        features = {}

        try:
            # Amount features
            amount = transaction_data.get("amount", 0)
            features["amount"] = float(amount)
            features["amount_log"] = float(np.log1p(amount))
            features["amount_sqrt"] = float(np.sqrt(amount))

            # Amount categories
            features["is_micro_transaction"] = float(amount < 1.0)
            features["is_small_transaction"] = float(1.0 <= amount < 50.0)
            features["is_medium_transaction"] = float(50.0 <= amount < 500.0)
            features["is_large_transaction"] = float(500.0 <= amount < 5000.0)
            features["is_very_large_transaction"] = float(amount >= 5000.0)

            # Currency features
            currency = transaction_data.get("currency", "USD")
            features["is_usd"] = float(currency == "USD")
            features["is_eur"] = float(currency == "EUR")
            features["is_exotic_currency"] = float(
                currency not in ["USD", "EUR", "GBP", "JPY"]
            )

            # Location features (if available)
            lat = transaction_data.get("latitude")
            lon = transaction_data.get("longitude")
            if lat is not None and lon is not None:
                features["has_location"] = 1.0
                features["latitude"] = float(lat)
                features["longitude"] = float(lon)
                features["distance_from_equator"] = abs(float(lat))
                features["is_unusual_location"] = float(abs(lat) > 60 or abs(lon) > 150)
            else:
                features["has_location"] = 0.0
                features["latitude"] = 0.0
                features["longitude"] = 0.0
                features["distance_from_equator"] = 0.0
                features["is_unusual_location"] = 0.0

        except Exception as e:
            logger.error(f"Transaction feature computation failed: {e}")

        return features

    def _update_cache_stats(self, cache_hit: bool):
        """Update cache hit rate statistics"""

        with self.lock:
            self.total_requests += 1
            if cache_hit:
                self.cache_hits += 1

            self.cache_hit_rate = self.cache_hits / self.total_requests

    def _update_computation_stats(self, feature_type: str, computation_time_ms: float):
        """Update computation time statistics"""

        with self.lock:
            if feature_type not in self.computation_stats:
                self.computation_stats[feature_type] = {
                    "total_time": 0.0,
                    "count": 0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                }

            stats = self.computation_stats[feature_type]
            stats["total_time"] += computation_time_ms
            stats["count"] += 1
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["min_time"] = min(stats["min_time"], computation_time_ms)
            stats["max_time"] = max(stats["max_time"], computation_time_ms)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get feature store performance statistics"""

        return {
            "cache_hit_rate": self.cache_hit_rate,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "computation_stats": self.computation_stats.copy(),
            "cache_size": len(self.feature_cache),
            "timestamp": datetime.now().isoformat(),
        }

    def clear_cache(self):
        """Clear all caches"""

        try:
            # Clear Redis cache
            keys = self.redis_client.keys("features:*")
            if keys:
                self.redis_client.delete(*keys)

            # Clear local cache
            self.feature_cache.clear()

            logger.info("Feature store cache cleared")

        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")


# Usage example
if __name__ == "__main__":
    print("🏪 Testing High-Performance Feature Store...")

    try:
        # Initialize feature store
        feature_store = HighPerformanceFeatureStore()

        # Test feature computation
        transaction_data = {
            "amount": 150.0,
            "currency": "USD",
            "category": "grocery",
            "timestamp": datetime.now().isoformat(),
            "latitude": 40.7128,
            "longitude": -74.0060,
        }

        # Get features
        features = feature_store.get_features(
            "test_tx_001", "user_001", "merchant_001", transaction_data
        )

        print(f"✅ Features computed: {len(features)} features")
        print(f"   Sample features: {list(features.keys())[:10]}")

        # Test cache hit
        features_cached = feature_store.get_features(
            "test_tx_001", "user_001", "merchant_001", transaction_data
        )

        # Get performance stats
        stats = feature_store.get_performance_stats()
        print(f"✅ Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Total requests: {stats['total_requests']}")

        print("\n🎉 Feature Store test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
