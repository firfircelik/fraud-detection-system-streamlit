#!/usr/bin/env python3
"""
Real-time Feature Engineering Service
Processes streaming transactions and generates ML features
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import redis
from kafka import KafkaConsumer, KafkaProducer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngine:
    def __init__(self):
        # Environment variables
        self.kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.postgres_url = os.getenv(
            "POSTGRES_URL",
            "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection",
        )
        self.batch_size = int(os.getenv("BATCH_SIZE", "1000"))
        self.processing_interval = int(os.getenv("PROCESSING_INTERVAL", "5"))

        # Initialize connections
        self.redis_client = None
        self.kafka_consumer = None
        self.kafka_producer = None
        self.db_engine = None
        self.session_local = None

        self.setup_connections()

    def setup_connections(self):
        """Setup database and messaging connections"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            logger.info("✅ Redis connection established")

            # Database connection
            self.db_engine = create_engine(self.postgres_url)
            self.session_local = sessionmaker(
                autocommit=False, autoflush=False, bind=self.db_engine
            )
            logger.info("✅ PostgreSQL connection established")

            # Kafka connections
            self.kafka_consumer = KafkaConsumer(
                "transactions",
                bootstrap_servers=self.kafka_servers.split(","),
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                group_id="feature-engine-group",
                auto_offset_reset="latest",
            )

            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers.split(","),
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            )
            logger.info("✅ Kafka connections established")

        except Exception as e:
            logger.error(f"❌ Failed to setup connections: {e}")
            raise

    def extract_temporal_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features from transaction"""
        timestamp = datetime.fromisoformat(
            transaction["timestamp"].replace("Z", "+00:00")
        )

        features = {
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "day_of_month": timestamp.day,
            "month": timestamp.month,
            "quarter": (timestamp.month - 1) // 3 + 1,
            "year": timestamp.year,
            "is_weekend": timestamp.weekday() >= 5,
            "is_business_hour": 9 <= timestamp.hour <= 17,
            "is_night_time": timestamp.hour < 6 or timestamp.hour > 22,
        }

        # Cyclical encoding
        import math

        features.update(
            {
                "hour_sin": math.sin(2 * math.pi * timestamp.hour / 24),
                "hour_cos": math.cos(2 * math.pi * timestamp.hour / 24),
                "day_sin": math.sin(2 * math.pi * timestamp.weekday() / 7),
                "day_cos": math.cos(2 * math.pi * timestamp.weekday() / 7),
                "month_sin": math.sin(2 * math.pi * timestamp.month / 12),
                "month_cos": math.cos(2 * math.pi * timestamp.month / 12),
            }
        )

        return features

    def extract_velocity_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract velocity features from Redis cache"""
        user_id = transaction["user_id"]
        current_time = datetime.now()

        # Time windows in seconds
        windows = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000}

        features = {}

        try:
            for window_name, window_seconds in windows.items():
                # Transaction count
                tx_key = f"user_tx_count:{user_id}:{window_name}"
                tx_count = self.redis_client.get(tx_key) or 0
                features[f"user_tx_count_{window_name}"] = int(tx_count)

                # Amount sum
                amount_key = f"user_amount:{user_id}:{window_name}"
                amount_sum = self.redis_client.get(amount_key) or 0.0
                features[f"user_amount_{window_name}"] = float(amount_sum)

                # Update counters
                self.redis_client.incr(tx_key)
                self.redis_client.expire(tx_key, window_seconds)

                current_amount = float(transaction["amount"])
                self.redis_client.incrbyfloat(amount_key, current_amount)
                self.redis_client.expire(amount_key, window_seconds)

        except Exception as e:
            logger.error(f"Error extracting velocity features: {e}")
            # Return default values
            for window_name in windows.keys():
                features[f"user_tx_count_{window_name}"] = 0
                features[f"user_amount_{window_name}"] = 0.0

        return features

    def extract_behavioral_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract behavioral features from database"""
        user_id = transaction["user_id"]
        amount = float(transaction["amount"])

        features = {
            "user_avg_amount": 0.0,
            "user_std_amount": 0.0,
            "user_median_amount": 0.0,
            "amount_zscore": 0.0,
            "amount_percentile": 0.5,
            "is_amount_outlier": False,
        }

        try:
            with self.session_local() as session:
                # Get user's historical transaction amounts
                query = text(
                    """
                    SELECT amount 
                    FROM transactions 
                    WHERE user_id = :user_id 
                    AND transaction_timestamp > NOW() - INTERVAL '30 days'
                    ORDER BY transaction_timestamp DESC 
                    LIMIT 100
                """
                )

                result = session.execute(query, {"user_id": user_id})
                amounts = [row[0] for row in result.fetchall()]

                if amounts:
                    df = pd.Series(amounts)
                    features["user_avg_amount"] = float(df.mean())
                    features["user_std_amount"] = (
                        float(df.std()) if len(amounts) > 1 else 0.0
                    )
                    features["user_median_amount"] = float(df.median())

                    # Z-score calculation
                    if features["user_std_amount"] > 0:
                        features["amount_zscore"] = (
                            amount - features["user_avg_amount"]
                        ) / features["user_std_amount"]

                    # Percentile calculation
                    features["amount_percentile"] = (
                        float(df.rank(pct=True).iloc[-1]) if len(amounts) > 1 else 0.5
                    )

                    # Outlier detection (using IQR method)
                    q1, q3 = df.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    features["is_amount_outlier"] = (
                        amount < lower_bound or amount > upper_bound
                    )

        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")

        return features

    def extract_merchant_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Extract merchant-related features"""
        merchant_id = transaction["merchant_id"]

        features = {
            "merchant_risk_score": 0.0,
            "merchant_fraud_rate_7d": 0.0,
            "merchant_fraud_rate_30d": 0.0,
            "merchant_tx_count_1h": 0,
            "merchant_tx_count_24h": 0,
            "is_high_risk_category": False,
        }

        try:
            # Get merchant risk score from cache
            risk_key = f"merchant_risk:{merchant_id}"
            cached_risk = self.redis_client.get(risk_key)
            if cached_risk:
                features["merchant_risk_score"] = float(cached_risk)

            # High-risk categories
            high_risk_categories = ["gambling", "crypto", "adult", "forex"]
            category = transaction.get("category", "").lower()
            features["is_high_risk_category"] = any(
                risk_cat in category for risk_cat in high_risk_categories
            )

            # Merchant transaction counts
            for window in ["1h", "24h"]:
                count_key = f"merchant_tx_count:{merchant_id}:{window}"
                count = self.redis_client.get(count_key) or 0
                features[f"merchant_tx_count_{window}"] = int(count)

                # Update counter
                self.redis_client.incr(count_key)
                window_seconds = 3600 if window == "1h" else 86400
                self.redis_client.expire(count_key, window_seconds)

        except Exception as e:
            logger.error(f"Error extracting merchant features: {e}")

        return features

    def extract_geographic_features(
        self, transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract geographic features"""
        features = {
            "distance_from_home": 0.0,
            "distance_from_last_tx": 0.0,
            "is_unusual_location": False,
            "country_risk_score": 0.0,
            "location_entropy": 0.0,
        }

        try:
            lat = transaction.get("latitude")
            lon = transaction.get("longitude")
            country = transaction.get("country", "UNKNOWN")

            if lat and lon:
                # Country risk scores (simplified)
                country_risks = {
                    "US": 0.1,
                    "GB": 0.1,
                    "CA": 0.1,
                    "AU": 0.1,
                    "DE": 0.1,
                    "FR": 0.15,
                    "IT": 0.2,
                    "ES": 0.2,
                    "BR": 0.3,
                    "IN": 0.25,
                    "CN": 0.4,
                    "RU": 0.6,
                    "NG": 0.7,
                    "UNKNOWN": 0.8,
                }
                features["country_risk_score"] = country_risks.get(country, 0.5)

                # Check for unusual locations (simplified)
                # In production, this would use proper geolocation services
                if abs(float(lat)) > 60 or abs(float(lon)) > 150:
                    features["is_unusual_location"] = True

        except Exception as e:
            logger.error(f"Error extracting geographic features: {e}")

        return features

    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction and extract all features"""
        try:
            # Extract all feature types
            temporal_features = self.extract_temporal_features(transaction)
            velocity_features = self.extract_velocity_features(transaction)
            behavioral_features = self.extract_behavioral_features(transaction)
            merchant_features = self.extract_merchant_features(transaction)
            geographic_features = self.extract_geographic_features(transaction)

            # Combine all features
            all_features = {
                "transaction_id": transaction["transaction_id"],
                "feature_version": "3.0.0",
                "created_at": datetime.now().isoformat(),
                **temporal_features,
                **velocity_features,
                **behavioral_features,
                **merchant_features,
                **geographic_features,
            }

            return all_features

        except Exception as e:
            logger.error(
                f"Error processing transaction {transaction.get('transaction_id', 'unknown')}: {e}"
            )
            return None

    def store_features(self, features: Dict[str, Any]):
        """Store features in database and cache"""
        try:
            # Store in database
            with self.session_local() as session:
                # Insert into ml_features table
                insert_query = text(
                    """
                    INSERT INTO ml_features (
                        transaction_id, feature_version, hour_of_day, day_of_week,
                        is_weekend, is_business_hour, user_tx_count_1h, user_tx_count_24h,
                        user_amount_1h, user_amount_24h, merchant_risk_score,
                        is_high_risk_category, country_risk_score, created_at
                    ) VALUES (
                        :transaction_id, :feature_version, :hour_of_day, :day_of_week,
                        :is_weekend, :is_business_hour, :user_tx_count_1h, :user_tx_count_24h,
                        :user_amount_1h, :user_amount_24h, :merchant_risk_score,
                        :is_high_risk_category, :country_risk_score, NOW()
                    )
                    ON CONFLICT (transaction_id) DO NOTHING
                """
                )

                session.execute(insert_query, features)
                session.commit()

            # Store in Redis cache for fast access
            cache_key = f"features:{features['transaction_id']}"
            self.redis_client.setex(cache_key, 3600, json.dumps(features))  # 1 hour TTL

        except Exception as e:
            logger.error(f"Error storing features: {e}")

    def run(self):
        """Main processing loop"""
        logger.info("🚀 Starting Feature Engine...")

        try:
            for message in self.kafka_consumer:
                transaction = message.value
                logger.info(
                    f"Processing transaction: {transaction.get('transaction_id', 'unknown')}"
                )

                # Extract features
                features = self.process_transaction(transaction)

                if features:
                    # Store features
                    self.store_features(features)

                    # Send to ML pipeline
                    self.kafka_producer.send("ml_features", features)

                    logger.info(
                        f"✅ Features extracted for transaction: {features['transaction_id']}"
                    )
                else:
                    logger.warning(f"❌ Failed to extract features for transaction")

        except KeyboardInterrupt:
            logger.info("🛑 Feature Engine stopped by user")
        except Exception as e:
            logger.error(f"❌ Feature Engine error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup connections"""
        try:
            if self.kafka_consumer:
                self.kafka_consumer.close()
            if self.kafka_producer:
                self.kafka_producer.close()
            if self.redis_client:
                self.redis_client.close()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point"""
    feature_engine = FeatureEngine()
    feature_engine.run()


if __name__ == "__main__":
    main()
