#!/usr/bin/env python3
"""
Redis Cache Manager for Fraud Detection
Handles feature caching, real-time data, and session management
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aioredis
import redis

logger = logging.getLogger(__name__)


@dataclass
class CachedFeatures:
    user_id: str
    features: Dict[str, float]
    computed_at: datetime
    expires_at: datetime
    version: str


class RedisManager:
    """Redis cache manager for fraud detection system"""

    def __init__(self):
        self.redis_url = os.getenv(
            "REDIS_URL", "redis://:RedisStack2024!@localhost:6379"
        )
        self.redis_client = None
        self.async_redis = None

        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.feature_cache_ttl = 1800  # 30 minutes
        self.session_ttl = 86400  # 24 hours
        self.fraud_score_ttl = 300  # 5 minutes

    async def initialize(self):
        """Initialize Redis connections"""
        # Sync Redis client
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30,
        )

        # Async Redis client
        self.async_redis = await aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        # Test connection
        await self.async_redis.ping()
        logger.info("Redis manager initialized")

    # =====================================================
    # FEATURE CACHING
    # =====================================================

    async def cache_user_features(
        self,
        user_id: str,
        features: Dict[str, float],
        version: str = "1.0",
        ttl: int = None,
    ) -> bool:
        """Cache user features with TTL"""
        try:
            cache_key = f"features:user:{user_id}"
            ttl = ttl or self.feature_cache_ttl

            cached_features = CachedFeatures(
                user_id=user_id,
                features=features,
                computed_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl),
                version=version,
            )

            # Store as JSON
            await self.async_redis.setex(
                cache_key, ttl, json.dumps(asdict(cached_features), default=str)
            )

            # Also store individual features for quick access
            feature_hash_key = f"features:hash:{user_id}"
            await self.async_redis.hset(feature_hash_key, mapping=features)
            await self.async_redis.expire(feature_hash_key, ttl)

            logger.debug(f"Cached features for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache features for user {user_id}: {e}")
            return False

    async def get_user_features(self, user_id: str) -> Optional[Dict[str, float]]:
        """Get cached user features"""
        try:
            cache_key = f"features:user:{user_id}"
            cached_data = await self.async_redis.get(cache_key)

            if cached_data:
                features_data = json.loads(cached_data)
                return features_data["features"]

            # Try hash-based cache as fallback
            feature_hash_key = f"features:hash:{user_id}"
            features = await self.async_redis.hgetall(feature_hash_key)

            if features:
                return {k: float(v) for k, v in features.items()}

            return None

        except Exception as e:
            logger.error(f"Failed to get features for user {user_id}: {e}")
            return None

    async def cache_merchant_features(
        self, merchant_id: str, features: Dict[str, float], ttl: int = None
    ) -> bool:
        """Cache merchant features"""
        try:
            cache_key = f"features:merchant:{merchant_id}"
            ttl = ttl or self.feature_cache_ttl

            await self.async_redis.setex(cache_key, ttl, json.dumps(features))

            logger.debug(f"Cached features for merchant {merchant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cache merchant features: {e}")
            return False

    async def get_merchant_features(
        self, merchant_id: str
    ) -> Optional[Dict[str, float]]:
        """Get cached merchant features"""
        try:
            cache_key = f"features:merchant:{merchant_id}"
            cached_data = await self.async_redis.get(cache_key)

            if cached_data:
                return json.loads(cached_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get merchant features: {e}")
            return None

    # =====================================================
    # REAL-TIME FRAUD SCORES
    # =====================================================

    async def cache_fraud_score(
        self,
        transaction_id: str,
        fraud_score: float,
        risk_level: str,
        decision: str,
        ttl: int = None,
    ) -> bool:
        """Cache fraud detection results"""
        try:
            cache_key = f"fraud:score:{transaction_id}"
            ttl = ttl or self.fraud_score_ttl

            score_data = {
                "transaction_id": transaction_id,
                "fraud_score": fraud_score,
                "risk_level": risk_level,
                "decision": decision,
                "cached_at": datetime.now().isoformat(),
            }

            await self.async_redis.setex(cache_key, ttl, json.dumps(score_data))

            # Add to sorted set for real-time monitoring
            await self.async_redis.zadd(
                "fraud:scores:realtime", {transaction_id: fraud_score}
            )

            # Keep only recent scores (last 1000)
            await self.async_redis.zremrangebyrank("fraud:scores:realtime", 0, -1001)

            return True

        except Exception as e:
            logger.error(f"Failed to cache fraud score: {e}")
            return False

    async def get_fraud_score(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get cached fraud score"""
        try:
            cache_key = f"fraud:score:{transaction_id}"
            cached_data = await self.async_redis.get(cache_key)

            if cached_data:
                return json.loads(cached_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get fraud score: {e}")
            return None

    async def get_top_fraud_scores(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top fraud scores from real-time monitoring"""
        try:
            # Get top scores from sorted set
            top_scores = await self.async_redis.zrevrange(
                "fraud:scores:realtime", 0, limit - 1, withscores=True
            )

            results = []
            for transaction_id, score in top_scores:
                # Get full details from cache
                score_data = await self.get_fraud_score(transaction_id)
                if score_data:
                    results.append(score_data)
                else:
                    results.append(
                        {
                            "transaction_id": transaction_id,
                            "fraud_score": score,
                            "risk_level": (
                                "HIGH"
                                if score > 0.7
                                else "MEDIUM" if score > 0.4 else "LOW"
                            ),
                            "decision": "UNKNOWN",
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Failed to get top fraud scores: {e}")
            return []

    # =====================================================
    # SESSION MANAGEMENT
    # =====================================================

    async def cache_session(
        self, session_id: str, session_data: Dict[str, Any], ttl: int = None
    ) -> bool:
        """Cache user session data"""
        try:
            cache_key = f"session:{session_id}"
            ttl = ttl or self.session_ttl

            await self.async_redis.setex(
                cache_key, ttl, json.dumps(session_data, default=str)
            )

            # Add to user's active sessions
            user_sessions_key = f"sessions:user:{session_data.get('user_id')}"
            await self.async_redis.sadd(user_sessions_key, session_id)
            await self.async_redis.expire(user_sessions_key, ttl)

            return True

        except Exception as e:
            logger.error(f"Failed to cache session: {e}")
            return False

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached session data"""
        try:
            cache_key = f"session:{session_id}"
            cached_data = await self.async_redis.get(cache_key)

            if cached_data:
                return json.loads(cached_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        try:
            # Get session data first
            session_data = await self.get_session(session_id)

            # Remove from cache
            cache_key = f"session:{session_id}"
            await self.async_redis.delete(cache_key)

            # Remove from user's active sessions
            if session_data and "user_id" in session_data:
                user_sessions_key = f"sessions:user:{session_data['user_id']}"
                await self.async_redis.srem(user_sessions_key, session_id)

            return True

        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
            return False

    # =====================================================
    # REAL-TIME EVENTS
    # =====================================================

    async def publish_fraud_event(self, event_data: Dict[str, Any]) -> bool:
        """Publish fraud detection event"""
        try:
            channel = "fraud:events"
            message = json.dumps(event_data, default=str)

            await self.async_redis.publish(channel, message)

            # Also add to stream for persistence
            stream_key = "fraud:events:stream"
            await self.async_redis.xadd(
                stream_key, event_data, maxlen=10000  # Keep last 10k events
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish fraud event: {e}")
            return False

    async def subscribe_to_fraud_events(self, callback):
        """Subscribe to fraud events"""
        try:
            pubsub = self.async_redis.pubsub()
            await pubsub.subscribe("fraud:events")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    event_data = json.loads(message["data"])
                    await callback(event_data)

        except Exception as e:
            logger.error(f"Failed to subscribe to fraud events: {e}")

    # =====================================================
    # RATE LIMITING
    # =====================================================

    async def check_rate_limit(
        self, key: str, limit: int, window_seconds: int
    ) -> Dict[str, Any]:
        """Check rate limit using sliding window"""
        try:
            now = datetime.now().timestamp()
            window_start = now - window_seconds

            # Remove old entries
            await self.async_redis.zremrangebyscore(key, 0, window_start)

            # Count current requests
            current_count = await self.async_redis.zcard(key)

            if current_count >= limit:
                # Get oldest entry to calculate reset time
                oldest = await self.async_redis.zrange(key, 0, 0, withscores=True)
                reset_time = (
                    oldest[0][1] + window_seconds if oldest else now + window_seconds
                )

                return {
                    "allowed": False,
                    "current_count": current_count,
                    "limit": limit,
                    "reset_time": reset_time,
                    "retry_after": reset_time - now,
                }

            # Add current request
            await self.async_redis.zadd(key, {str(now): now})
            await self.async_redis.expire(key, window_seconds)

            return {
                "allowed": True,
                "current_count": current_count + 1,
                "limit": limit,
                "remaining": limit - current_count - 1,
                "reset_time": now + window_seconds,
            }

        except Exception as e:
            logger.error(f"Failed to check rate limit: {e}")
            return {"allowed": True, "current_count": 0, "limit": limit}

    # =====================================================
    # ANALYTICS AND MONITORING
    # =====================================================

    async def increment_counter(
        self, key: str, amount: int = 1, ttl: int = None
    ) -> int:
        """Increment a counter with optional TTL"""
        try:
            current_value = await self.async_redis.incrby(key, amount)

            if ttl and current_value == amount:  # First increment
                await self.async_redis.expire(key, ttl)

            return current_value

        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            return 0

    async def get_counter(self, key: str) -> int:
        """Get counter value"""
        try:
            value = await self.async_redis.get(key)
            return int(value) if value else 0

        except Exception as e:
            logger.error(f"Failed to get counter: {e}")
            return 0

    async def set_gauge(self, key: str, value: float, ttl: int = None) -> bool:
        """Set a gauge metric"""
        try:
            if ttl:
                await self.async_redis.setex(key, ttl, str(value))
            else:
                await self.async_redis.set(key, str(value))

            return True

        except Exception as e:
            logger.error(f"Failed to set gauge: {e}")
            return False

    async def get_gauge(self, key: str) -> Optional[float]:
        """Get gauge value"""
        try:
            value = await self.async_redis.get(key)
            return float(value) if value else None

        except Exception as e:
            logger.error(f"Failed to get gauge: {e}")
            return None

    # =====================================================
    # CACHE MANAGEMENT
    # =====================================================

    async def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        try:
            keys = await self.async_redis.keys(pattern)
            if keys:
                deleted = await self.async_redis.delete(*keys)
                logger.info(
                    f"Cleared {deleted} cache entries matching pattern: {pattern}"
                )
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Failed to clear cache pattern: {e}")
            return 0

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.async_redis.info()

            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    async def close(self):
        """Close Redis connections"""
        if self.async_redis:
            await self.async_redis.close()
        if self.redis_client:
            self.redis_client.close()


# Example usage
async def main():
    redis_manager = RedisManager()
    await redis_manager.initialize()

    try:
        # Test feature caching
        features = {
            "velocity_1h": 5.0,
            "amount_avg_7d": 250.50,
            "merchant_risk_score": 0.3,
            "location_risk": 0.1,
        }

        await redis_manager.cache_user_features("user_001", features)
        cached_features = await redis_manager.get_user_features("user_001")
        print(f"Cached features: {cached_features}")

        # Test fraud score caching
        await redis_manager.cache_fraud_score("tx_001", 0.85, "HIGH", "DECLINED")
        fraud_score = await redis_manager.get_fraud_score("tx_001")
        print(f"Fraud score: {fraud_score}")

        # Test rate limiting
        rate_limit = await redis_manager.check_rate_limit("api:user_001", 100, 3600)
        print(f"Rate limit: {rate_limit}")

        # Test cache stats
        stats = await redis_manager.get_cache_stats()
        print(f"Cache stats: {stats}")

        print("✅ Redis manager test completed successfully!")

    except Exception as e:
        print(f"❌ Redis manager test failed: {e}")
    finally:
        await redis_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
