#!/usr/bin/env python3
"""
Elasticsearch Integration for Advanced Search and Analytics
Full-text search, geo-spatial indexing, and ML-powered anomaly detection
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.helpers import async_bulk, bulk
from geopy import distance

logger = logging.getLogger(__name__)


@dataclass
class TransactionDocument:
    """Transaction document for Elasticsearch"""

    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    location: Optional[Dict[str, float]] = None
    device_info: Optional[Dict[str, Any]] = None
    fraud_score: Optional[float] = None
    risk_level: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AuditLogDocument:
    """Audit log document for Elasticsearch"""

    log_id: str
    user_id: Optional[str]
    action: str
    resource: str
    timestamp: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    outcome: str = "success"
    details: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Search result with highlighting and scoring"""

    document_id: str
    source: Dict[str, Any]
    score: float
    highlights: Optional[Dict[str, List[str]]] = None
    explanation: Optional[Dict[str, Any]] = None


@dataclass
class AnomalyDetectionResult:
    """ML-powered anomaly detection result"""

    document_id: str
    anomaly_score: float
    is_anomaly: bool
    contributing_factors: List[str]
    timestamp: datetime


class ElasticsearchManager:
    """
    Advanced Elasticsearch integration for fraud detection
    Features: Full-text search, geo-spatial queries, ML anomaly detection
    """

    def __init__(self, hosts: List[str], username: str = None, password: str = None):
        self.hosts = hosts
        self.username = username
        self.password = password

        # Initialize clients
        auth = (username, password) if username and password else None
        self.client = Elasticsearch(hosts=hosts, basic_auth=auth, verify_certs=False)
        self.async_client = AsyncElasticsearch(
            hosts=hosts, basic_auth=auth, verify_certs=False
        )

        # Index configurations
        self.indices = {
            "transactions": "fraud_transactions",
            "audit_logs": "fraud_audit_logs",
            "user_profiles": "fraud_user_profiles",
            "merchant_profiles": "fraud_merchant_profiles",
            "anomalies": "fraud_anomalies",
        }

        # ML configurations
        self.ml_jobs = {}
        self.anomaly_detectors = {}

        # Search templates
        self.search_templates = {}

        # Performance metrics
        self.metrics = {
            "documents_indexed": 0,
            "searches_performed": 0,
            "anomalies_detected": 0,
            "ml_jobs_created": 0,
        }

    async def initialize(self):
        """Initialize Elasticsearch indices and configurations"""
        try:
            # Create indices with optimized mappings
            await self._create_transaction_index()
            await self._create_audit_log_index()
            await self._create_user_profile_index()
            await self._create_merchant_profile_index()
            await self._create_anomaly_index()

            # Set up search templates
            await self._setup_search_templates()

            # Initialize ML jobs
            await self._initialize_ml_jobs()

            logger.info("Elasticsearch manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {e}")
            raise

    # =====================================================
    # INDEX MANAGEMENT
    # =====================================================

    async def _create_transaction_index(self):
        """Create optimized transaction index with geo-spatial support"""

        mapping = {
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "index.mapping.total_fields.limit": 2000,
                "analysis": {
                    "analyzer": {
                        "fraud_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"],
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "transaction_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "merchant_id": {"type": "keyword"},
                    "amount": {"type": "double"},
                    "currency": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "location": {"type": "geo_point"},
                    "device_info": {
                        "properties": {
                            "device_id": {"type": "keyword"},
                            "device_type": {"type": "keyword"},
                            "os": {"type": "keyword"},
                            "browser": {"type": "keyword"},
                            "ip_address": {"type": "ip"},
                        }
                    },
                    "fraud_score": {"type": "double"},
                    "risk_level": {"type": "keyword"},
                    "features": {"type": "object", "dynamic": True},
                    "metadata": {
                        "properties": {
                            "merchant_category": {"type": "keyword"},
                            "payment_method": {"type": "keyword"},
                            "channel": {"type": "keyword"},
                            "description": {
                                "type": "text",
                                "analyzer": "fraud_analyzer",
                            },
                        }
                    },
                }
            },
        }

        index_name = self.indices["transactions"]
        if not await self._index_exists(index_name):
            await self.async_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created transaction index: {index_name}")

    async def _create_audit_log_index(self):
        """Create audit log index with security-focused mapping"""

        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
                "index.mapping.total_fields.limit": 1000,
            },
            "mappings": {
                "properties": {
                    "log_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "action": {"type": "keyword"},
                    "resource": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "ip_address": {"type": "ip"},
                    "user_agent": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "outcome": {"type": "keyword"},
                    "details": {"type": "object", "dynamic": True},
                }
            },
        }

        index_name = self.indices["audit_logs"]
        if not await self._index_exists(index_name):
            await self.async_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created audit log index: {index_name}")

    async def _create_user_profile_index(self):
        """Create user profile index for behavioral analysis"""

        mapping = {
            "settings": {"number_of_shards": 2, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "profile_created": {"type": "date"},
                    "last_updated": {"type": "date"},
                    "behavioral_features": {
                        "properties": {
                            "avg_transaction_amount": {"type": "double"},
                            "transaction_frequency": {"type": "double"},
                            "preferred_merchants": {"type": "keyword"},
                            "common_locations": {"type": "geo_point"},
                            "device_fingerprints": {"type": "keyword"},
                            "time_patterns": {"type": "object"},
                        }
                    },
                    "risk_indicators": {
                        "properties": {
                            "historical_fraud_score": {"type": "double"},
                            "velocity_alerts": {"type": "integer"},
                            "suspicious_patterns": {"type": "keyword"},
                        }
                    },
                }
            },
        }

        index_name = self.indices["user_profiles"]
        if not await self._index_exists(index_name):
            await self.async_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created user profile index: {index_name}")

    async def _create_merchant_profile_index(self):
        """Create merchant profile index for merchant analysis"""

        mapping = {
            "settings": {"number_of_shards": 2, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "merchant_id": {"type": "keyword"},
                    "merchant_name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "category": {"type": "keyword"},
                    "location": {"type": "geo_point"},
                    "business_metrics": {
                        "properties": {
                            "daily_volume": {"type": "double"},
                            "avg_transaction_amount": {"type": "double"},
                            "fraud_rate": {"type": "double"},
                            "chargeback_rate": {"type": "double"},
                        }
                    },
                    "risk_profile": {
                        "properties": {
                            "risk_score": {"type": "double"},
                            "monitoring_level": {"type": "keyword"},
                            "compliance_status": {"type": "keyword"},
                        }
                    },
                }
            },
        }

        index_name = self.indices["merchant_profiles"]
        if not await self._index_exists(index_name):
            await self.async_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created merchant profile index: {index_name}")

    async def _create_anomaly_index(self):
        """Create anomaly detection results index"""

        mapping = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "anomaly_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "anomaly_score": {"type": "double"},
                    "is_anomaly": {"type": "boolean"},
                    "contributing_factors": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "ml_job_id": {"type": "keyword"},
                    "explanation": {"type": "text"},
                }
            },
        }

        index_name = self.indices["anomalies"]
        if not await self._index_exists(index_name):
            await self.async_client.indices.create(index=index_name, body=mapping)
            logger.info(f"Created anomaly index: {index_name}")

    # =====================================================
    # DOCUMENT INDEXING
    # =====================================================

    async def index_transaction(self, transaction: TransactionDocument) -> bool:
        """Index a single transaction document"""
        try:
            doc = asdict(transaction)

            # Convert datetime to ISO format
            if isinstance(doc["timestamp"], datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()

            # Prepare location for geo_point
            if (
                doc.get("location")
                and "lat" in doc["location"]
                and "lon" in doc["location"]
            ):
                doc["location"] = {
                    "lat": doc["location"]["lat"],
                    "lon": doc["location"]["lon"],
                }

            result = await self.async_client.index(
                index=self.indices["transactions"],
                id=transaction.transaction_id,
                body=doc,
            )

            self.metrics["documents_indexed"] += 1
            logger.debug(f"Indexed transaction {transaction.transaction_id}")
            return result["result"] == "created" or result["result"] == "updated"

        except Exception as e:
            logger.error(
                f"Failed to index transaction {transaction.transaction_id}: {e}"
            )
            return False

    async def bulk_index_transactions(
        self, transactions: List[TransactionDocument]
    ) -> Dict[str, int]:
        """Bulk index multiple transactions"""
        try:
            actions = []
            for transaction in transactions:
                doc = asdict(transaction)

                # Convert datetime to ISO format
                if isinstance(doc["timestamp"], datetime):
                    doc["timestamp"] = doc["timestamp"].isoformat()

                # Prepare location for geo_point
                if (
                    doc.get("location")
                    and "lat" in doc["location"]
                    and "lon" in doc["location"]
                ):
                    doc["location"] = {
                        "lat": doc["location"]["lat"],
                        "lon": doc["location"]["lon"],
                    }

                actions.append(
                    {
                        "_index": self.indices["transactions"],
                        "_id": transaction.transaction_id,
                        "_source": doc,
                    }
                )

            # Perform bulk indexing
            success_count, failed_items = await async_bulk(
                self.async_client,
                actions,
                chunk_size=1000,
                max_chunk_bytes=10 * 1024 * 1024,  # 10MB chunks
            )

            self.metrics["documents_indexed"] += success_count

            result = {"success_count": success_count, "failed_count": len(failed_items)}

            if failed_items:
                logger.warning(f"Failed to index {len(failed_items)} transactions")

            logger.info(f"Bulk indexed {success_count} transactions")
            return result

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {"success_count": 0, "failed_count": len(transactions)}

    # =====================================================
    # SEARCH CAPABILITIES
    # =====================================================

    async def search_transactions(
        self,
        query: Dict[str, Any],
        size: int = 100,
        from_: int = 0,
        sort: List[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Advanced transaction search with highlighting"""
        try:
            search_body = {
                "query": query,
                "size": size,
                "from": from_,
                "highlight": {
                    "fields": {"metadata.description": {}, "device_info.*": {}}
                },
            }

            if sort:
                search_body["sort"] = sort

            response = await self.async_client.search(
                index=self.indices["transactions"], body=search_body
            )

            results = []
            for hit in response["hits"]["hits"]:
                result = SearchResult(
                    document_id=hit["_id"],
                    source=hit["_source"],
                    score=hit["_score"],
                    highlights=hit.get("highlight"),
                )
                results.append(result)

            self.metrics["searches_performed"] += 1
            return results

        except Exception as e:
            logger.error(f"Transaction search failed: {e}")
            return []

    async def geo_search_transactions(
        self,
        center_lat: float,
        center_lon: float,
        distance_km: float,
        additional_filters: Dict[str, Any] = None,
    ) -> List[SearchResult]:
        """Geo-spatial search for transactions"""
        try:
            geo_query = {
                "bool": {
                    "filter": [
                        {
                            "geo_distance": {
                                "distance": f"{distance_km}km",
                                "location": {"lat": center_lat, "lon": center_lon},
                            }
                        }
                    ]
                }
            }

            # Add additional filters
            if additional_filters:
                if "must" in additional_filters:
                    geo_query["bool"]["must"] = additional_filters["must"]
                if "filter" in additional_filters:
                    geo_query["bool"]["filter"].extend(additional_filters["filter"])

            return await self.search_transactions(geo_query)

        except Exception as e:
            logger.error(f"Geo search failed: {e}")
            return []

    async def complex_fraud_search(
        self,
        user_id: str = None,
        merchant_id: str = None,
        min_amount: float = None,
        max_amount: float = None,
        min_fraud_score: float = None,
        time_range: Tuple[datetime, datetime] = None,
        risk_levels: List[str] = None,
    ) -> List[SearchResult]:
        """Complex search for fraud investigation"""
        try:
            must_clauses = []
            filter_clauses = []

            # User filter
            if user_id:
                filter_clauses.append({"term": {"user_id": user_id}})

            # Merchant filter
            if merchant_id:
                filter_clauses.append({"term": {"merchant_id": merchant_id}})

            # Amount range
            if min_amount is not None or max_amount is not None:
                amount_range = {}
                if min_amount is not None:
                    amount_range["gte"] = min_amount
                if max_amount is not None:
                    amount_range["lte"] = max_amount
                filter_clauses.append({"range": {"amount": amount_range}})

            # Fraud score filter
            if min_fraud_score is not None:
                filter_clauses.append(
                    {"range": {"fraud_score": {"gte": min_fraud_score}}}
                )

            # Time range
            if time_range:
                start_time, end_time = time_range
                filter_clauses.append(
                    {
                        "range": {
                            "timestamp": {
                                "gte": start_time.isoformat(),
                                "lte": end_time.isoformat(),
                            }
                        }
                    }
                )

            # Risk levels
            if risk_levels:
                filter_clauses.append({"terms": {"risk_level": risk_levels}})

            query = {"bool": {"filter": filter_clauses}}

            if must_clauses:
                query["bool"]["must"] = must_clauses

            return await self.search_transactions(
                query, sort=[{"timestamp": {"order": "desc"}}]
            )

        except Exception as e:
            logger.error(f"Complex fraud search failed: {e}")
            return []

    # =====================================================
    # AGGREGATION ANALYTICS
    # =====================================================

    async def fraud_analytics_aggregation(
        self, time_bucket: str = "1h"
    ) -> Dict[str, Any]:
        """Complex aggregation for fraud analytics"""
        try:
            aggregation_body = {
                "size": 0,
                "aggs": {
                    "fraud_over_time": {
                        "date_histogram": {
                            "field": "timestamp",
                            "fixed_interval": time_bucket,
                        },
                        "aggs": {
                            "total_amount": {"sum": {"field": "amount"}},
                            "avg_fraud_score": {"avg": {"field": "fraud_score"}},
                            "fraud_transactions": {
                                "filter": {"range": {"fraud_score": {"gte": 0.7}}},
                                "aggs": {
                                    "count": {
                                        "value_count": {"field": "transaction_id"}
                                    },
                                    "total_fraud_amount": {"sum": {"field": "amount"}},
                                },
                            },
                        },
                    },
                    "risk_level_distribution": {"terms": {"field": "risk_level"}},
                    "top_merchants_by_risk": {
                        "terms": {"field": "merchant_id", "size": 10},
                        "aggs": {
                            "avg_fraud_score": {"avg": {"field": "fraud_score"}},
                            "total_amount": {"sum": {"field": "amount"}},
                        },
                    },
                    "geographic_distribution": {
                        "geo_hash_grid": {"field": "location", "precision": 5},
                        "aggs": {"avg_fraud_score": {"avg": {"field": "fraud_score"}}},
                    },
                },
            }

            response = await self.async_client.search(
                index=self.indices["transactions"], body=aggregation_body
            )

            return response["aggregations"]

        except Exception as e:
            logger.error(f"Fraud analytics aggregation failed: {e}")
            return {}

    # =====================================================
    # ML ANOMALY DETECTION
    # =====================================================

    async def _initialize_ml_jobs(self):
        """Initialize ML jobs for anomaly detection"""
        try:
            # Transaction amount anomaly detection
            amount_job = {
                "job_id": "transaction_amount_anomaly",
                "description": "Detect anomalous transaction amounts",
                "analysis_config": {
                    "bucket_span": "15m",
                    "detectors": [
                        {
                            "function": "high_mean",
                            "field_name": "amount",
                            "partition_field_name": "user_id",
                        }
                    ],
                },
                "data_description": {"time_field": "timestamp"},
            }

            # Velocity anomaly detection
            velocity_job = {
                "job_id": "transaction_velocity_anomaly",
                "description": "Detect anomalous transaction velocity",
                "analysis_config": {
                    "bucket_span": "5m",
                    "detectors": [
                        {"function": "high_count", "partition_field_name": "user_id"}
                    ],
                },
                "data_description": {"time_field": "timestamp"},
            }

            # Store ML job configurations
            self.ml_jobs["amount_anomaly"] = amount_job
            self.ml_jobs["velocity_anomaly"] = velocity_job

            logger.info("ML jobs initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ML jobs: {e}")

    async def detect_anomalies(
        self, documents: List[Dict[str, Any]], job_type: str = "amount_anomaly"
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies using ML algorithms"""
        try:
            results = []

            if job_type == "amount_anomaly":
                results = await self._detect_amount_anomalies(documents)
            elif job_type == "velocity_anomaly":
                results = await self._detect_velocity_anomalies(documents)
            elif job_type == "pattern_anomaly":
                results = await self._detect_pattern_anomalies(documents)

            # Store anomaly results
            if results:
                await self._store_anomaly_results(results)

            self.metrics["anomalies_detected"] += len(results)
            return results

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    async def _detect_amount_anomalies(
        self, documents: List[Dict[str, Any]]
    ) -> List[AnomalyDetectionResult]:
        """Detect amount-based anomalies"""
        results = []

        # Group by user
        user_amounts = {}
        for doc in documents:
            user_id = doc.get("user_id")
            amount = doc.get("amount", 0)
            if user_id:
                if user_id not in user_amounts:
                    user_amounts[user_id] = []
                user_amounts[user_id].append((doc, amount))

        # Detect anomalies for each user
        for user_id, user_data in user_amounts.items():
            amounts = [amount for _, amount in user_data]
            if len(amounts) > 3:  # Need minimum data points
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)

                for doc, amount in user_data:
                    # Z-score based anomaly detection
                    z_score = (
                        abs((amount - mean_amount) / std_amount)
                        if std_amount > 0
                        else 0
                    )

                    if z_score > 2.5:  # Anomaly threshold
                        anomaly = AnomalyDetectionResult(
                            document_id=doc.get("transaction_id", ""),
                            anomaly_score=float(z_score),
                            is_anomaly=True,
                            contributing_factors=["unusual_amount"],
                            timestamp=datetime.now(timezone.utc),
                        )
                        results.append(anomaly)

        return results

    async def _detect_velocity_anomalies(
        self, documents: List[Dict[str, Any]]
    ) -> List[AnomalyDetectionResult]:
        """Detect velocity-based anomalies"""
        results = []

        # Group by user and time windows
        user_velocities = {}
        time_window = timedelta(minutes=15)

        for doc in documents:
            user_id = doc.get("user_id")
            timestamp_str = doc.get("timestamp")

            if user_id and timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    time_bucket = timestamp.replace(
                        minute=(timestamp.minute // 15) * 15, second=0, microsecond=0
                    )

                    key = f"{user_id}_{time_bucket.isoformat()}"
                    if key not in user_velocities:
                        user_velocities[key] = []
                    user_velocities[key].append(doc)
                except:
                    continue

        # Detect high velocity
        for key, transactions in user_velocities.items():
            if len(transactions) > 10:  # High velocity threshold
                for doc in transactions:
                    anomaly = AnomalyDetectionResult(
                        document_id=doc.get("transaction_id", ""),
                        anomaly_score=len(transactions) / 10.0,  # Normalize to 1.0+
                        is_anomaly=True,
                        contributing_factors=["high_velocity"],
                        timestamp=datetime.now(timezone.utc),
                    )
                    results.append(anomaly)

        return results

    async def _detect_pattern_anomalies(
        self, documents: List[Dict[str, Any]]
    ) -> List[AnomalyDetectionResult]:
        """Detect pattern-based anomalies"""
        results = []

        # Detect round amount patterns
        for doc in documents:
            amount = doc.get("amount", 0)

            # Check for round amounts (potential testing)
            if amount > 0 and amount == int(amount) and amount % 100 == 0:
                anomaly = AnomalyDetectionResult(
                    document_id=doc.get("transaction_id", ""),
                    anomaly_score=0.8,
                    is_anomaly=True,
                    contributing_factors=["round_amount_pattern"],
                    timestamp=datetime.now(timezone.utc),
                )
                results.append(anomaly)

        return results

    async def _store_anomaly_results(self, results: List[AnomalyDetectionResult]):
        """Store anomaly detection results"""
        try:
            actions = []
            for result in results:
                doc = asdict(result)
                doc["timestamp"] = doc["timestamp"].isoformat()
                doc["anomaly_id"] = str(uuid.uuid4())

                actions.append({"_index": self.indices["anomalies"], "_source": doc})

            if actions:
                await async_bulk(self.async_client, actions)
                logger.info(f"Stored {len(results)} anomaly results")

        except Exception as e:
            logger.error(f"Failed to store anomaly results: {e}")

    # =====================================================
    # SEARCH TEMPLATES
    # =====================================================

    async def _setup_search_templates(self):
        """Set up predefined search templates"""
        try:
            # High-risk transactions template
            high_risk_template = {
                "script": {
                    "lang": "mustache",
                    "source": {
                        "query": {
                            "bool": {
                                "filter": [
                                    {
                                        "range": {
                                            "fraud_score": {
                                                "gte": "{{min_fraud_score}}"
                                            }
                                        }
                                    },
                                    {
                                        "range": {
                                            "timestamp": {
                                                "gte": "{{start_time}}",
                                                "lte": "{{end_time}}",
                                            }
                                        }
                                    },
                                ]
                            }
                        },
                        "sort": [{"fraud_score": {"order": "desc"}}],
                    },
                }
            }

            # User activity template
            user_activity_template = {
                "script": {
                    "lang": "mustache",
                    "source": {
                        "query": {
                            "bool": {"filter": [{"term": {"user_id": "{{user_id}}"}}]}
                        },
                        "aggs": {
                            "activity_over_time": {
                                "date_histogram": {
                                    "field": "timestamp",
                                    "fixed_interval": "{{time_interval}}",
                                }
                            }
                        },
                    },
                }
            }

            self.search_templates["high_risk"] = high_risk_template
            self.search_templates["user_activity"] = user_activity_template

            logger.info("Search templates set up")

        except Exception as e:
            logger.error(f"Failed to set up search templates: {e}")

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    async def _index_exists(self, index_name: str) -> bool:
        """Check if index exists"""
        try:
            return await self.async_client.indices.exists(index=index_name)
        except:
            return False

    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get Elasticsearch cluster health"""
        try:
            health = await self.async_client.cluster.health()
            return health
        except Exception as e:
            logger.error(f"Failed to get cluster health: {e}")
            return {}

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for all fraud detection indices"""
        try:
            stats = {}
            for name, index in self.indices.items():
                if await self._index_exists(index):
                    index_stats = await self.async_client.indices.stats(index=index)
                    stats[name] = {
                        "document_count": index_stats["indices"][index]["total"][
                            "docs"
                        ]["count"],
                        "size_bytes": index_stats["indices"][index]["total"]["store"][
                            "size_in_bytes"
                        ],
                    }
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()

    async def close(self):
        """Close Elasticsearch connections"""
        try:
            await self.async_client.close()
            self.client.close()
            logger.info("Elasticsearch connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage of Elasticsearch manager"""

    # Initialize manager
    es_manager = ElasticsearchManager(
        hosts=["localhost:9200"], username="elastic", password="password"
    )

    await es_manager.initialize()

    try:
        # Example transaction document
        transaction = TransactionDocument(
            transaction_id="txn_12345",
            user_id="user_789",
            merchant_id="merchant_456",
            amount=1500.00,
            currency="USD",
            timestamp=datetime.now(timezone.utc),
            location={"lat": 40.7128, "lon": -74.0060},
            fraud_score=0.85,
            risk_level="HIGH",
            metadata={
                "description": "Suspicious high-value transaction",
                "merchant_category": "electronics",
            },
        )

        # Index transaction
        success = await es_manager.index_transaction(transaction)
        print(f"Transaction indexed: {success}")

        # Search for high-risk transactions
        high_risk_query = {"range": {"fraud_score": {"gte": 0.8}}}

        results = await es_manager.search_transactions(high_risk_query)
        print(f"Found {len(results)} high-risk transactions")

        # Perform fraud analytics
        analytics = await es_manager.fraud_analytics_aggregation()
        print(f"Analytics results: {analytics}")

        # Get cluster health
        health = await es_manager.get_cluster_health()
        print(f"Cluster health: {health['status']}")

    finally:
        await es_manager.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
