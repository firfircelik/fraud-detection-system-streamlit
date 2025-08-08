#!/usr/bin/env python3
"""
Geospatial Analysis Engine for Fraud Detection
Location-based risk assessment, geographic clustering, and spatial analytics
"""

import asyncio
import json
import logging
import math
import sqlite3
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import folium
import geohash
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class GeoLocation:
    """Geographic location with metadata"""

    location_id: str
    latitude: float
    longitude: float
    altitude: Optional[float]
    accuracy: float  # meters
    address: Optional[str]
    city: Optional[str]
    country: Optional[str]
    postal_code: Optional[str]
    timezone: Optional[str]
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class GeoTransaction:
    """Transaction with geographic information"""

    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    location: GeoLocation
    merchant_location: Optional[GeoLocation]
    device_location: Optional[GeoLocation]
    timestamp: datetime
    risk_score: float
    properties: Dict[str, Any]


@dataclass
class GeoCluster:
    """Geographic cluster of transactions or locations"""

    cluster_id: str
    center_lat: float
    center_lon: float
    radius_km: float
    transaction_ids: List[str]
    location_ids: List[str]
    cluster_type: str  # fraud_hotspot, merchant_cluster, user_activity
    risk_level: str  # low, medium, high, critical
    confidence_score: float
    temporal_pattern: Dict[str, Any]
    properties: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime]


@dataclass
class RiskZone:
    """High-risk geographic zone"""

    zone_id: str
    name: str
    geometry: Polygon  # Shapely polygon
    risk_level: str
    risk_factors: List[str]
    active_incidents: int
    historical_incidents: int
    population_density: Optional[float]
    economic_indicators: Dict[str, float]
    created_at: datetime
    last_updated: datetime


@dataclass
class TravelPattern:
    """User travel behavior analysis"""

    user_id: str
    home_location: GeoLocation
    work_location: Optional[GeoLocation]
    frequent_locations: List[GeoLocation]
    travel_radius: float  # km
    velocity_profile: Dict[str, float]  # max_km_per_hour, avg_km_per_hour
    anomaly_threshold: float
    last_known_location: GeoLocation
    suspicious_movements: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class GeospatialAnalysisEngine:
    """
    Advanced geospatial analysis for fraud detection
    Features: Location clustering, risk zones, travel analysis, geographic anomaly detection
    """

    def __init__(self, cache_size: int = 10000):

        # Core storage
        self.locations: Dict[str, GeoLocation] = {}
        self.transactions: Dict[str, GeoTransaction] = {}
        self.clusters: Dict[str, GeoCluster] = {}
        self.risk_zones: Dict[str, RiskZone] = {}
        self.travel_patterns: Dict[str, TravelPattern] = {}

        # Geospatial indexing
        self.location_index: Dict[str, List[str]] = defaultdict(
            list
        )  # geohash -> location_ids
        self.transaction_index: Dict[str, List[str]] = defaultdict(
            list
        )  # geohash -> transaction_ids
        self.geohash_precision = 7  # ~153m resolution

        # Risk assessment
        self.risk_models = {
            "velocity_check": self._velocity_risk_analysis,
            "location_frequency": self._location_frequency_analysis,
            "geographic_outlier": self._geographic_outlier_analysis,
            "cluster_proximity": self._cluster_proximity_analysis,
            "zone_risk": self._zone_risk_analysis,
        }

        # Clustering algorithms
        self.clustering_algorithms = {
            "dbscan": self._dbscan_clustering,
            "grid_based": self._grid_based_clustering,
            "hotspot": self._hotspot_clustering,
            "temporal_spatial": self._temporal_spatial_clustering,
        }

        # Geographic utilities
        self.geocoder = Nominatim(user_agent="fraud_detection_system")
        self.distance_cache: Dict[Tuple[str, str], float] = {}
        self.cache_size = cache_size

        # Analytics cache
        self.analytics_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Real-time monitoring
        self.active_monitoring = True
        self.monitoring_queue: deque = deque(maxlen=1000)
        self.alert_thresholds = {
            "velocity_anomaly": 500.0,  # km/h
            "location_jump": 1000.0,  # km
            "cluster_density": 0.8,  # confidence threshold
            "risk_zone_entry": 0.7,  # risk threshold
        }

        # Performance metrics
        self.metrics = {
            "locations_processed": 0,
            "transactions_analyzed": 0,
            "clusters_detected": 0,
            "risk_zones_identified": 0,
            "anomalies_detected": 0,
            "travel_patterns_analyzed": 0,
        }

        # Thread safety
        self.lock = threading.RLock()

    # =====================================================
    # LOCATION MANAGEMENT
    # =====================================================

    def add_location(
        self, latitude: float, longitude: float, location_data: Dict[str, Any] = None
    ) -> str:
        """Add geographic location"""

        location_id = str(uuid.uuid4())

        if location_data is None:
            location_data = {}

        # Reverse geocoding for address information
        address_info = self._reverse_geocode(latitude, longitude)

        location = GeoLocation(
            location_id=location_id,
            latitude=latitude,
            longitude=longitude,
            altitude=location_data.get("altitude"),
            accuracy=location_data.get("accuracy", 10.0),
            address=address_info.get("address"),
            city=address_info.get("city"),
            country=address_info.get("country"),
            postal_code=address_info.get("postal_code"),
            timezone=location_data.get("timezone"),
            properties=location_data.get("properties", {}),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with self.lock:
            self.locations[location_id] = location

            # Add to geospatial index
            geo_hash = geohash.encode(
                latitude, longitude, precision=self.geohash_precision
            )
            self.location_index[geo_hash].append(location_id)

        self.metrics["locations_processed"] += 1
        logger.info(f"Added location {location_id} at ({latitude}, {longitude})")

        return location_id

    def add_transaction(self, transaction_data: Dict[str, Any]) -> str:
        """Add transaction with geographic data"""

        transaction_id = transaction_data["transaction_id"]

        # Process location data
        location_data = transaction_data.get("location", {})
        location_id = self.add_location(
            latitude=location_data["latitude"],
            longitude=location_data["longitude"],
            location_data=location_data,
        )
        location = self.locations[location_id]

        # Process merchant location if available
        merchant_location = None
        if "merchant_location" in transaction_data:
            merchant_data = transaction_data["merchant_location"]
            merchant_location_id = self.add_location(
                latitude=merchant_data["latitude"],
                longitude=merchant_data["longitude"],
                location_data=merchant_data,
            )
            merchant_location = self.locations[merchant_location_id]

        # Process device location if available
        device_location = None
        if "device_location" in transaction_data:
            device_data = transaction_data["device_location"]
            device_location_id = self.add_location(
                latitude=device_data["latitude"],
                longitude=device_data["longitude"],
                location_data=device_data,
            )
            device_location = self.locations[device_location_id]

        transaction = GeoTransaction(
            transaction_id=transaction_id,
            user_id=transaction_data["user_id"],
            merchant_id=transaction_data["merchant_id"],
            amount=transaction_data["amount"],
            currency=transaction_data.get("currency", "USD"),
            location=location,
            merchant_location=merchant_location,
            device_location=device_location,
            timestamp=datetime.fromisoformat(transaction_data["timestamp"]),
            risk_score=0.0,  # Will be calculated
            properties=transaction_data.get("properties", {}),
        )

        # Calculate initial risk score
        risk_score = self.analyze_transaction_risk(transaction)
        transaction.risk_score = risk_score

        with self.lock:
            self.transactions[transaction_id] = transaction

            # Add to geospatial index
            geo_hash = geohash.encode(
                location.latitude, location.longitude, precision=self.geohash_precision
            )
            self.transaction_index[geo_hash].append(transaction_id)

        # Update travel patterns
        self._update_travel_pattern(transaction)

        self.metrics["transactions_analyzed"] += 1
        logger.info(
            f"Added transaction {transaction_id} with risk score {risk_score:.3f}"
        )

        return transaction_id

    def _reverse_geocode(self, latitude: float, longitude: float) -> Dict[str, str]:
        """Reverse geocoding to get address information"""

        try:
            location = self.geocoder.reverse(f"{latitude}, {longitude}", language="en")
            if location and location.raw:
                address_components = location.raw.get("address", {})
                return {
                    "address": location.address,
                    "city": address_components.get("city")
                    or address_components.get("town"),
                    "country": address_components.get("country"),
                    "postal_code": address_components.get("postcode"),
                }
        except Exception as e:
            logger.warning(
                f"Reverse geocoding failed for ({latitude}, {longitude}): {e}"
            )

        return {}

    # =====================================================
    # RISK ANALYSIS
    # =====================================================

    def analyze_transaction_risk(self, transaction: GeoTransaction) -> float:
        """Comprehensive geographic risk analysis"""

        risk_scores = {}

        # Run all risk models
        for model_name, model_func in self.risk_models.items():
            try:
                score = model_func(transaction)
                risk_scores[model_name] = score
            except Exception as e:
                logger.warning(f"Risk model {model_name} failed: {e}")
                risk_scores[model_name] = 0.0

        # Weighted combination of risk scores
        weights = {
            "velocity_check": 0.25,
            "location_frequency": 0.20,
            "geographic_outlier": 0.20,
            "cluster_proximity": 0.20,
            "zone_risk": 0.15,
        }

        total_risk = sum(
            weights.get(model, 0.1) * score for model, score in risk_scores.items()
        )

        # Normalize to 0-1 range
        final_risk = min(1.0, max(0.0, total_risk))

        logger.debug(
            f"Transaction {transaction.transaction_id} risk breakdown: {risk_scores}, final: {final_risk:.3f}"
        )

        return final_risk

    def _velocity_risk_analysis(self, transaction: GeoTransaction) -> float:
        """Analyze user velocity between transactions"""

        user_id = transaction.user_id
        current_location = transaction.location
        current_time = transaction.timestamp

        # Find previous transaction for this user
        previous_transaction = None
        for trans in self.transactions.values():
            if (
                trans.user_id == user_id
                and trans.timestamp < current_time
                and (
                    previous_transaction is None
                    or trans.timestamp > previous_transaction.timestamp
                )
            ):
                previous_transaction = trans

        if not previous_transaction:
            return 0.0  # No previous transaction to compare

        # Calculate distance and time difference
        distance_km = self._calculate_distance(
            previous_transaction.location.latitude,
            previous_transaction.location.longitude,
            current_location.latitude,
            current_location.longitude,
        )

        time_diff = (
            current_time - previous_transaction.timestamp
        ).total_seconds() / 3600  # hours

        if time_diff <= 0:
            return 1.0  # Same time, different location is suspicious

        velocity_kmh = distance_km / time_diff

        # Risk based on velocity thresholds
        if velocity_kmh > 800:  # Faster than commercial aircraft
            return 1.0
        elif velocity_kmh > 300:  # Very fast travel
            return 0.8
        elif velocity_kmh > 100:  # Highway speed
            return 0.3
        else:
            return 0.0

    def _location_frequency_analysis(self, transaction: GeoTransaction) -> float:
        """Analyze frequency of user at this location"""

        user_id = transaction.user_id
        current_location = transaction.location

        # Count user transactions in nearby area (1km radius)
        nearby_transactions = self._find_nearby_transactions(
            current_location.latitude,
            current_location.longitude,
            radius_km=1.0,
            user_id=user_id,
        )

        frequency = len(nearby_transactions)

        # Risk inversely related to frequency
        if frequency == 1:  # First time at this location
            return 0.6
        elif frequency <= 3:  # Infrequent location
            return 0.3
        else:  # Frequent location
            return 0.0

    def _geographic_outlier_analysis(self, transaction: GeoTransaction) -> float:
        """Detect geographic outliers for user behavior"""

        user_id = transaction.user_id
        current_location = transaction.location

        # Get user's historical locations
        user_transactions = [
            t for t in self.transactions.values() if t.user_id == user_id
        ]

        if len(user_transactions) < 5:
            return 0.2  # Limited history, moderate risk

        # Calculate distances from user's typical locations
        distances = []
        for trans in user_transactions[-20:]:  # Last 20 transactions
            if trans.transaction_id != transaction.transaction_id:
                distance = self._calculate_distance(
                    trans.location.latitude,
                    trans.location.longitude,
                    current_location.latitude,
                    current_location.longitude,
                )
                distances.append(distance)

        if not distances:
            return 0.0

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # Current distance from user's center
        center_lat = np.mean([t.location.latitude for t in user_transactions[-10:]])
        center_lon = np.mean([t.location.longitude for t in user_transactions[-10:]])

        current_distance = self._calculate_distance(
            center_lat,
            center_lon,
            current_location.latitude,
            current_location.longitude,
        )

        # Z-score based risk
        if std_distance > 0:
            z_score = abs(current_distance - mean_distance) / std_distance
            return min(1.0, z_score / 3.0)  # Risk increases with z-score

        return 0.0

    def _cluster_proximity_analysis(self, transaction: GeoTransaction) -> float:
        """Analyze proximity to known fraud clusters"""

        current_location = transaction.location

        # Find nearby fraud clusters
        nearby_clusters = []
        for cluster in self.clusters.values():
            if cluster.cluster_type == "fraud_hotspot":
                distance = self._calculate_distance(
                    cluster.center_lat,
                    cluster.center_lon,
                    current_location.latitude,
                    current_location.longitude,
                )

                if distance <= cluster.radius_km:
                    nearby_clusters.append((cluster, distance))

        if not nearby_clusters:
            return 0.0

        # Risk based on closest high-risk cluster
        max_risk = 0.0
        for cluster, distance in nearby_clusters:
            if cluster.risk_level == "critical":
                cluster_risk = 0.9
            elif cluster.risk_level == "high":
                cluster_risk = 0.7
            elif cluster.risk_level == "medium":
                cluster_risk = 0.4
            else:
                cluster_risk = 0.2

            # Adjust risk based on distance from cluster center
            distance_factor = 1.0 - (distance / cluster.radius_km)
            risk = cluster_risk * distance_factor * cluster.confidence_score

            max_risk = max(max_risk, risk)

        return max_risk

    def _zone_risk_analysis(self, transaction: GeoTransaction) -> float:
        """Analyze risk based on geographic zones"""

        current_location = transaction.location
        point = Point(current_location.longitude, current_location.latitude)

        # Check if transaction is in any risk zones
        for zone in self.risk_zones.values():
            if zone.geometry.contains(point):
                if zone.risk_level == "critical":
                    return 0.9
                elif zone.risk_level == "high":
                    return 0.7
                elif zone.risk_level == "medium":
                    return 0.4
                else:
                    return 0.2

        return 0.0

    # =====================================================
    # CLUSTERING ALGORITHMS
    # =====================================================

    def detect_geographic_clusters(
        self, algorithm: str = "dbscan", parameters: Dict[str, Any] = None
    ) -> List[GeoCluster]:
        """Detect geographic clusters of transactions"""

        if parameters is None:
            parameters = {}

        if algorithm in self.clustering_algorithms:
            clusters = self.clustering_algorithms[algorithm](parameters)
        else:
            logger.warning(f"Unknown clustering algorithm: {algorithm}")
            clusters = []

        # Store clusters
        for cluster in clusters:
            self.clusters[cluster.cluster_id] = cluster

        self.metrics["clusters_detected"] += len(clusters)
        logger.info(f"Detected {len(clusters)} clusters using {algorithm}")

        return clusters

    def _dbscan_clustering(self, parameters: Dict[str, Any]) -> List[GeoCluster]:
        """DBSCAN clustering for geographic data"""

        eps_km = parameters.get("eps_km", 0.5)  # 500m radius
        min_samples = parameters.get("min_samples", 5)
        time_window_hours = parameters.get("time_window_hours", 24)

        # Get recent transactions
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_transactions = [
            t for t in self.transactions.values() if t.timestamp >= cutoff_time
        ]

        if len(recent_transactions) < min_samples:
            return []

        # Prepare data for clustering
        coordinates = np.array(
            [[t.location.latitude, t.location.longitude] for t in recent_transactions]
        )

        # Convert to radians for haversine distance
        coordinates_rad = np.radians(coordinates)

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=eps_km / 6371.0,  # Convert km to radians (Earth radius ~6371km)
            min_samples=min_samples,
            metric="haversine",
        ).fit(coordinates_rad)

        # Create clusters
        clusters = []
        labels = clustering.labels_

        for cluster_label in set(labels):
            if cluster_label == -1:  # Noise points
                continue

            # Get transactions in this cluster
            cluster_transactions = [
                recent_transactions[i]
                for i, label in enumerate(labels)
                if label == cluster_label
            ]

            if len(cluster_transactions) < min_samples:
                continue

            # Calculate cluster properties
            cluster_coords = [
                (t.location.latitude, t.location.longitude)
                for t in cluster_transactions
            ]

            center_lat = np.mean([coord[0] for coord in cluster_coords])
            center_lon = np.mean([coord[1] for coord in cluster_coords])

            # Calculate radius
            distances = [
                self._calculate_distance(center_lat, center_lon, lat, lon)
                for lat, lon in cluster_coords
            ]
            radius_km = max(distances) if distances else 0.0

            # Analyze cluster risk
            risk_level, confidence = self._analyze_cluster_risk(cluster_transactions)

            cluster = GeoCluster(
                cluster_id=str(uuid.uuid4()),
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                transaction_ids=[t.transaction_id for t in cluster_transactions],
                location_ids=[t.location.location_id for t in cluster_transactions],
                cluster_type="transaction_cluster",
                risk_level=risk_level,
                confidence_score=confidence,
                temporal_pattern=self._analyze_temporal_pattern(cluster_transactions),
                properties={
                    "algorithm": "dbscan",
                    "eps_km": eps_km,
                    "min_samples": min_samples,
                    "transaction_count": len(cluster_transactions),
                },
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            )

            clusters.append(cluster)

        return clusters

    def _grid_based_clustering(self, parameters: Dict[str, Any]) -> List[GeoCluster]:
        """Grid-based clustering for hotspot detection"""

        grid_size_km = parameters.get("grid_size_km", 1.0)
        min_transactions = parameters.get("min_transactions", 10)
        time_window_hours = parameters.get("time_window_hours", 24)

        # Get recent transactions
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_transactions = [
            t for t in self.transactions.values() if t.timestamp >= cutoff_time
        ]

        # Grid-based grouping
        grid_cells = defaultdict(list)

        for transaction in recent_transactions:
            # Calculate grid cell
            lat_grid = int(
                transaction.location.latitude / (grid_size_km / 111.0)
            )  # ~111km per degree
            lon_grid = int(
                transaction.location.longitude
                / (
                    grid_size_km
                    / (111.0 * math.cos(math.radians(transaction.location.latitude)))
                )
            )

            grid_key = (lat_grid, lon_grid)
            grid_cells[grid_key].append(transaction)

        # Create clusters from grid cells
        clusters = []

        for grid_key, cell_transactions in grid_cells.items():
            if len(cell_transactions) < min_transactions:
                continue

            # Calculate cell center
            center_lat = np.mean([t.location.latitude for t in cell_transactions])
            center_lon = np.mean([t.location.longitude for t in cell_transactions])

            # Analyze cluster risk
            risk_level, confidence = self._analyze_cluster_risk(cell_transactions)

            cluster = GeoCluster(
                cluster_id=str(uuid.uuid4()),
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=grid_size_km / 2,  # Half of grid size
                transaction_ids=[t.transaction_id for t in cell_transactions],
                location_ids=[t.location.location_id for t in cell_transactions],
                cluster_type="grid_hotspot",
                risk_level=risk_level,
                confidence_score=confidence,
                temporal_pattern=self._analyze_temporal_pattern(cell_transactions),
                properties={
                    "algorithm": "grid_based",
                    "grid_size_km": grid_size_km,
                    "grid_key": grid_key,
                    "transaction_count": len(cell_transactions),
                },
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            )

            clusters.append(cluster)

        return clusters

    def _hotspot_clustering(self, parameters: Dict[str, Any]) -> List[GeoCluster]:
        """Fraud hotspot detection based on risk scores"""

        min_risk_score = parameters.get("min_risk_score", 0.7)
        radius_km = parameters.get("radius_km", 2.0)
        min_transactions = parameters.get("min_transactions", 5)
        time_window_hours = parameters.get("time_window_hours", 48)

        # Get high-risk recent transactions
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        high_risk_transactions = [
            t
            for t in self.transactions.values()
            if t.timestamp >= cutoff_time and t.risk_score >= min_risk_score
        ]

        if len(high_risk_transactions) < min_transactions:
            return []

        clusters = []
        processed = set()

        for transaction in high_risk_transactions:
            if transaction.transaction_id in processed:
                continue

            # Find nearby high-risk transactions
            nearby_transactions = []
            for other_transaction in high_risk_transactions:
                if other_transaction.transaction_id != transaction.transaction_id:
                    distance = self._calculate_distance(
                        transaction.location.latitude,
                        transaction.location.longitude,
                        other_transaction.location.latitude,
                        other_transaction.location.longitude,
                    )

                    if distance <= radius_km:
                        nearby_transactions.append(other_transaction)

            # Include original transaction
            cluster_transactions = [transaction] + nearby_transactions

            if len(cluster_transactions) >= min_transactions:
                # Mark as processed
                for t in cluster_transactions:
                    processed.add(t.transaction_id)

                # Calculate cluster center
                center_lat = np.mean(
                    [t.location.latitude for t in cluster_transactions]
                )
                center_lon = np.mean(
                    [t.location.longitude for t in cluster_transactions]
                )

                # Calculate actual radius
                distances = [
                    self._calculate_distance(
                        center_lat,
                        center_lon,
                        t.location.latitude,
                        t.location.longitude,
                    )
                    for t in cluster_transactions
                ]
                actual_radius = max(distances) if distances else 0.0

                cluster = GeoCluster(
                    cluster_id=str(uuid.uuid4()),
                    center_lat=center_lat,
                    center_lon=center_lon,
                    radius_km=actual_radius,
                    transaction_ids=[t.transaction_id for t in cluster_transactions],
                    location_ids=[t.location.location_id for t in cluster_transactions],
                    cluster_type="fraud_hotspot",
                    risk_level="high",
                    confidence_score=np.mean(
                        [t.risk_score for t in cluster_transactions]
                    ),
                    temporal_pattern=self._analyze_temporal_pattern(
                        cluster_transactions
                    ),
                    properties={
                        "algorithm": "hotspot",
                        "min_risk_score": min_risk_score,
                        "search_radius_km": radius_km,
                        "transaction_count": len(cluster_transactions),
                        "avg_risk_score": np.mean(
                            [t.risk_score for t in cluster_transactions]
                        ),
                    },
                    created_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=48),
                )

                clusters.append(cluster)

        return clusters

    def _temporal_spatial_clustering(
        self, parameters: Dict[str, Any]
    ) -> List[GeoCluster]:
        """Temporal-spatial clustering for pattern detection"""

        spatial_eps_km = parameters.get("spatial_eps_km", 1.0)
        temporal_eps_hours = parameters.get("temporal_eps_hours", 2.0)
        min_samples = parameters.get("min_samples", 5)
        time_window_hours = parameters.get("time_window_hours", 72)

        # Get recent transactions
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_transactions = [
            t for t in self.transactions.values() if t.timestamp >= cutoff_time
        ]

        if len(recent_transactions) < min_samples:
            return []

        # Prepare spatio-temporal data
        features = []
        for transaction in recent_transactions:
            # Normalize spatial coordinates (rough conversion to km)
            lat_km = transaction.location.latitude * 111.0
            lon_km = (
                transaction.location.longitude
                * 111.0
                * math.cos(math.radians(transaction.location.latitude))
            )

            # Normalize temporal coordinate (hours since cutoff)
            time_hours = (transaction.timestamp - cutoff_time).total_seconds() / 3600

            features.append([lat_km, lon_km, time_hours])

        features = np.array(features)

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # DBSCAN clustering
        clustering = DBSCAN(
            eps=0.5, min_samples=min_samples  # Adjusted for scaled features
        ).fit(features_scaled)

        # Create clusters
        clusters = []
        labels = clustering.labels_

        for cluster_label in set(labels):
            if cluster_label == -1:  # Noise points
                continue

            cluster_transactions = [
                recent_transactions[i]
                for i, label in enumerate(labels)
                if label == cluster_label
            ]

            if len(cluster_transactions) < min_samples:
                continue

            # Calculate spatial center
            center_lat = np.mean([t.location.latitude for t in cluster_transactions])
            center_lon = np.mean([t.location.longitude for t in cluster_transactions])

            # Calculate spatial radius
            distances = [
                self._calculate_distance(
                    center_lat, center_lon, t.location.latitude, t.location.longitude
                )
                for t in cluster_transactions
            ]
            radius_km = max(distances) if distances else 0.0

            # Analyze temporal pattern
            temporal_pattern = self._analyze_temporal_pattern(cluster_transactions)

            # Determine cluster type based on temporal spread
            time_span_hours = temporal_pattern.get("duration_hours", 0)
            if time_span_hours <= temporal_eps_hours:
                cluster_type = "burst_activity"
            else:
                cluster_type = "sustained_activity"

            risk_level, confidence = self._analyze_cluster_risk(cluster_transactions)

            cluster = GeoCluster(
                cluster_id=str(uuid.uuid4()),
                center_lat=center_lat,
                center_lon=center_lon,
                radius_km=radius_km,
                transaction_ids=[t.transaction_id for t in cluster_transactions],
                location_ids=[t.location.location_id for t in cluster_transactions],
                cluster_type=cluster_type,
                risk_level=risk_level,
                confidence_score=confidence,
                temporal_pattern=temporal_pattern,
                properties={
                    "algorithm": "temporal_spatial",
                    "spatial_eps_km": spatial_eps_km,
                    "temporal_eps_hours": temporal_eps_hours,
                    "transaction_count": len(cluster_transactions),
                    "time_span_hours": time_span_hours,
                },
                created_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=72),
            )

            clusters.append(cluster)

        return clusters

    # =====================================================
    # TRAVEL PATTERN ANALYSIS
    # =====================================================

    def _update_travel_pattern(self, transaction: GeoTransaction):
        """Update user travel pattern analysis"""

        user_id = transaction.user_id

        # Get or create travel pattern
        if user_id not in self.travel_patterns:
            self.travel_patterns[user_id] = self._create_travel_pattern(user_id)

        pattern = self.travel_patterns[user_id]

        # Update last known location
        pattern.last_known_location = transaction.location

        # Check for suspicious movement
        if self._is_suspicious_movement(pattern, transaction):
            suspicious_event = {
                "transaction_id": transaction.transaction_id,
                "location": (
                    transaction.location.latitude,
                    transaction.location.longitude,
                ),
                "timestamp": transaction.timestamp.isoformat(),
                "reason": "velocity_anomaly",
                "details": self._get_movement_details(pattern, transaction),
            }
            pattern.suspicious_movements.append(suspicious_event)

        # Update frequent locations
        self._update_frequent_locations(pattern, transaction.location)

        # Update velocity profile
        self._update_velocity_profile(pattern, transaction)

        pattern.updated_at = datetime.now(timezone.utc)

        self.metrics["travel_patterns_analyzed"] += 1

    def _create_travel_pattern(self, user_id: str) -> TravelPattern:
        """Create initial travel pattern for user"""

        # Analyze user's historical transactions
        user_transactions = [
            t for t in self.transactions.values() if t.user_id == user_id
        ]

        if not user_transactions:
            # No history, create minimal pattern
            return TravelPattern(
                user_id=user_id,
                home_location=None,
                work_location=None,
                frequent_locations=[],
                travel_radius=50.0,  # Default 50km radius
                velocity_profile={"max_km_per_hour": 100.0, "avg_km_per_hour": 25.0},
                anomaly_threshold=500.0,  # 500 km/h threshold
                last_known_location=None,
                suspicious_movements=[],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

        # Analyze transaction locations
        locations = [t.location for t in user_transactions]

        # Detect home location (most frequent location)
        location_counts = defaultdict(int)
        for location in locations:
            # Group nearby locations (within 1km)
            location_key = f"{location.latitude:.3f},{location.longitude:.3f}"
            location_counts[location_key] += 1

        # Find most frequent location as home
        if location_counts:
            home_key = max(location_counts, key=location_counts.get)
            home_lat, home_lon = map(float, home_key.split(","))
            home_location = next(
                (
                    loc
                    for loc in locations
                    if abs(loc.latitude - home_lat) < 0.001
                    and abs(loc.longitude - home_lon) < 0.001
                ),
                locations[0],
            )
        else:
            home_location = locations[0]

        # Calculate travel radius (95th percentile of distances from home)
        if home_location:
            distances = [
                self._calculate_distance(
                    home_location.latitude,
                    home_location.longitude,
                    loc.latitude,
                    loc.longitude,
                )
                for loc in locations
            ]
            travel_radius = np.percentile(distances, 95) if distances else 50.0
        else:
            travel_radius = 50.0

        # Calculate velocity profile
        velocities = []
        sorted_transactions = sorted(user_transactions, key=lambda x: x.timestamp)

        for i in range(1, len(sorted_transactions)):
            prev_trans = sorted_transactions[i - 1]
            curr_trans = sorted_transactions[i]

            distance = self._calculate_distance(
                prev_trans.location.latitude,
                prev_trans.location.longitude,
                curr_trans.location.latitude,
                curr_trans.location.longitude,
            )

            time_diff = (
                curr_trans.timestamp - prev_trans.timestamp
            ).total_seconds() / 3600

            if time_diff > 0:
                velocity = distance / time_diff
                velocities.append(velocity)

        if velocities:
            max_velocity = max(velocities)
            avg_velocity = np.mean(velocities)
        else:
            max_velocity = 100.0
            avg_velocity = 25.0

        return TravelPattern(
            user_id=user_id,
            home_location=home_location,
            work_location=None,  # TODO: Implement work location detection
            frequent_locations=self._detect_frequent_locations(locations),
            travel_radius=travel_radius,
            velocity_profile={
                "max_km_per_hour": max_velocity,
                "avg_km_per_hour": avg_velocity,
            },
            anomaly_threshold=max(500.0, max_velocity * 2),  # 2x max observed velocity
            last_known_location=locations[-1] if locations else None,
            suspicious_movements=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def _is_suspicious_movement(
        self, pattern: TravelPattern, transaction: GeoTransaction
    ) -> bool:
        """Check if movement is suspicious"""

        if not pattern.last_known_location:
            return False

        # Calculate velocity since last transaction
        user_transactions = [
            t
            for t in self.transactions.values()
            if t.user_id == transaction.user_id and t.timestamp < transaction.timestamp
        ]

        if not user_transactions:
            return False

        last_transaction = max(user_transactions, key=lambda x: x.timestamp)

        distance = self._calculate_distance(
            last_transaction.location.latitude,
            last_transaction.location.longitude,
            transaction.location.latitude,
            transaction.location.longitude,
        )

        time_diff = (
            transaction.timestamp - last_transaction.timestamp
        ).total_seconds() / 3600

        if time_diff <= 0:
            return True  # Same time, different location

        velocity = distance / time_diff

        return velocity > pattern.anomaly_threshold

    def _get_movement_details(
        self, pattern: TravelPattern, transaction: GeoTransaction
    ) -> Dict[str, Any]:
        """Get detailed movement analysis"""

        if not pattern.last_known_location:
            return {}

        user_transactions = [
            t
            for t in self.transactions.values()
            if t.user_id == transaction.user_id and t.timestamp < transaction.timestamp
        ]

        if not user_transactions:
            return {}

        last_transaction = max(user_transactions, key=lambda x: x.timestamp)

        distance = self._calculate_distance(
            last_transaction.location.latitude,
            last_transaction.location.longitude,
            transaction.location.latitude,
            transaction.location.longitude,
        )

        time_diff = (transaction.timestamp - last_transaction.timestamp).total_seconds()
        velocity_kmh = (
            (distance / (time_diff / 3600)) if time_diff > 0 else float("inf")
        )

        return {
            "distance_km": distance,
            "time_diff_seconds": time_diff,
            "velocity_kmh": velocity_kmh,
            "threshold_kmh": pattern.anomaly_threshold,
            "previous_location": {
                "latitude": last_transaction.location.latitude,
                "longitude": last_transaction.location.longitude,
                "timestamp": last_transaction.timestamp.isoformat(),
            },
        }

    def _update_frequent_locations(self, pattern: TravelPattern, location: GeoLocation):
        """Update frequent locations list"""

        # Check if location is close to any existing frequent location
        for freq_loc in pattern.frequent_locations:
            distance = self._calculate_distance(
                freq_loc.latitude,
                freq_loc.longitude,
                location.latitude,
                location.longitude,
            )

            if distance <= 1.0:  # Within 1km
                return  # Already have this location

        # Add new frequent location if within travel radius
        if pattern.home_location:
            distance_from_home = self._calculate_distance(
                pattern.home_location.latitude,
                pattern.home_location.longitude,
                location.latitude,
                location.longitude,
            )

            if distance_from_home <= pattern.travel_radius:
                pattern.frequent_locations.append(location)

                # Keep only top 10 frequent locations
                if len(pattern.frequent_locations) > 10:
                    pattern.frequent_locations = pattern.frequent_locations[-10:]

    def _update_velocity_profile(
        self, pattern: TravelPattern, transaction: GeoTransaction
    ):
        """Update velocity profile statistics"""

        user_transactions = [
            t
            for t in self.transactions.values()
            if t.user_id == transaction.user_id and t.timestamp < transaction.timestamp
        ]

        if not user_transactions:
            return

        # Calculate velocities for recent transactions
        recent_transactions = sorted(user_transactions[-20:], key=lambda x: x.timestamp)
        recent_transactions.append(transaction)

        velocities = []
        for i in range(1, len(recent_transactions)):
            prev_trans = recent_transactions[i - 1]
            curr_trans = recent_transactions[i]

            distance = self._calculate_distance(
                prev_trans.location.latitude,
                prev_trans.location.longitude,
                curr_trans.location.latitude,
                curr_trans.location.longitude,
            )

            time_diff = (
                curr_trans.timestamp - prev_trans.timestamp
            ).total_seconds() / 3600

            if time_diff > 0:
                velocity = distance / time_diff
                velocities.append(velocity)

        if velocities:
            # Update profile with moving average
            alpha = 0.1  # Smoothing factor

            new_max = max(velocities)
            new_avg = np.mean(velocities)

            pattern.velocity_profile["max_km_per_hour"] = (
                alpha * new_max
                + (1 - alpha) * pattern.velocity_profile["max_km_per_hour"]
            )

            pattern.velocity_profile["avg_km_per_hour"] = (
                alpha * new_avg
                + (1 - alpha) * pattern.velocity_profile["avg_km_per_hour"]
            )

    def _detect_frequent_locations(
        self, locations: List[GeoLocation]
    ) -> List[GeoLocation]:
        """Detect frequent locations from history"""

        # Group nearby locations
        location_groups = []

        for location in locations:
            # Find existing group within 1km
            found_group = False
            for group in location_groups:
                representative = group[0]
                distance = self._calculate_distance(
                    representative.latitude,
                    representative.longitude,
                    location.latitude,
                    location.longitude,
                )

                if distance <= 1.0:  # Within 1km
                    group.append(location)
                    found_group = True
                    break

            if not found_group:
                location_groups.append([location])

        # Select most frequent groups
        frequent_groups = sorted(location_groups, key=len, reverse=True)[:5]

        # Return representative location from each group
        frequent_locations = []
        for group in frequent_groups:
            if len(group) >= 2:  # At least 2 occurrences to be "frequent"
                # Use centroid as representative
                center_lat = np.mean([loc.latitude for loc in group])
                center_lon = np.mean([loc.longitude for loc in group])

                # Find closest actual location to centroid
                closest_location = min(
                    group,
                    key=lambda loc: self._calculate_distance(
                        center_lat, center_lon, loc.latitude, loc.longitude
                    ),
                )

                frequent_locations.append(closest_location)

        return frequent_locations

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in kilometers"""

        # Use cache for performance
        cache_key = (lat1, lon1, lat2, lon2)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        # Calculate using geodesic distance
        distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers

        # Cache result
        if len(self.distance_cache) < self.cache_size:
            self.distance_cache[cache_key] = distance

        return distance

    def _find_nearby_transactions(
        self, latitude: float, longitude: float, radius_km: float, user_id: str = None
    ) -> List[GeoTransaction]:
        """Find transactions within radius"""

        nearby_transactions = []

        for transaction in self.transactions.values():
            if user_id and transaction.user_id != user_id:
                continue

            distance = self._calculate_distance(
                latitude,
                longitude,
                transaction.location.latitude,
                transaction.location.longitude,
            )

            if distance <= radius_km:
                nearby_transactions.append(transaction)

        return nearby_transactions

    def _analyze_cluster_risk(
        self, transactions: List[GeoTransaction]
    ) -> Tuple[str, float]:
        """Analyze risk level of transaction cluster"""

        if not transactions:
            return "low", 0.0

        # Calculate average risk score
        avg_risk = np.mean([t.risk_score for t in transactions])

        # Analyze transaction patterns
        unique_users = len(set(t.user_id for t in transactions))
        unique_merchants = len(set(t.merchant_id for t in transactions))
        total_amount = sum(t.amount for t in transactions)

        # Risk factors
        risk_factors = 0

        # High average risk score
        if avg_risk > 0.7:
            risk_factors += 2
        elif avg_risk > 0.5:
            risk_factors += 1

        # Multiple users in small area (potential fraud ring)
        if unique_users > len(transactions) * 0.7:
            risk_factors += 2

        # High transaction volume
        if total_amount > 50000:  # $50k threshold
            risk_factors += 1

        # Determine risk level
        if risk_factors >= 4:
            risk_level = "critical"
            confidence = 0.9
        elif risk_factors >= 3:
            risk_level = "high"
            confidence = 0.8
        elif risk_factors >= 2:
            risk_level = "medium"
            confidence = 0.6
        else:
            risk_level = "low"
            confidence = 0.4

        return risk_level, confidence

    def _analyze_temporal_pattern(
        self, transactions: List[GeoTransaction]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in transaction cluster"""

        if not transactions:
            return {}

        timestamps = [t.timestamp for t in transactions]

        # Time span analysis
        min_time = min(timestamps)
        max_time = max(timestamps)
        duration = (max_time - min_time).total_seconds() / 3600  # hours

        # Time distribution analysis
        hours = [t.hour for t in timestamps]
        hour_distribution = {h: hours.count(h) for h in range(24)}
        peak_hour = max(hour_distribution, key=hour_distribution.get)

        # Day of week analysis
        weekdays = [t.weekday() for t in timestamps]
        weekday_distribution = {d: weekdays.count(d) for d in range(7)}

        return {
            "start_time": min_time.isoformat(),
            "end_time": max_time.isoformat(),
            "duration_hours": duration,
            "transaction_count": len(transactions),
            "peak_hour": peak_hour,
            "hour_distribution": hour_distribution,
            "weekday_distribution": weekday_distribution,
            "transactions_per_hour": len(transactions) / max(1, duration),
        }

    # =====================================================
    # QUERY AND ANALYSIS METHODS
    # =====================================================

    def get_user_travel_pattern(self, user_id: str) -> Optional[TravelPattern]:
        """Get travel pattern for user"""
        return self.travel_patterns.get(user_id)

    def get_location_risk_score(self, latitude: float, longitude: float) -> float:
        """Get risk score for specific location"""

        # Check risk zones
        point = Point(longitude, latitude)
        for zone in self.risk_zones.values():
            if zone.geometry.contains(point):
                if zone.risk_level == "critical":
                    return 0.9
                elif zone.risk_level == "high":
                    return 0.7
                elif zone.risk_level == "medium":
                    return 0.4
                else:
                    return 0.2

        # Check proximity to fraud clusters
        nearby_clusters = []
        for cluster in self.clusters.values():
            if cluster.cluster_type == "fraud_hotspot":
                distance = self._calculate_distance(
                    cluster.center_lat, cluster.center_lon, latitude, longitude
                )

                if distance <= cluster.radius_km * 2:  # Extended radius
                    nearby_clusters.append((cluster, distance))

        if nearby_clusters:
            # Risk based on closest cluster
            closest_cluster, distance = min(nearby_clusters, key=lambda x: x[1])
            distance_factor = max(0, 1 - (distance / (closest_cluster.radius_km * 2)))

            if closest_cluster.risk_level == "critical":
                base_risk = 0.8
            elif closest_cluster.risk_level == "high":
                base_risk = 0.6
            elif closest_cluster.risk_level == "medium":
                base_risk = 0.4
            else:
                base_risk = 0.2

            return base_risk * distance_factor * closest_cluster.confidence_score

        return 0.1  # Low default risk

    def get_area_statistics(
        self, latitude: float, longitude: float, radius_km: float
    ) -> Dict[str, Any]:
        """Get statistics for geographic area"""

        # Find transactions in area
        area_transactions = self._find_nearby_transactions(
            latitude, longitude, radius_km
        )

        if not area_transactions:
            return {
                "transaction_count": 0,
                "avg_risk_score": 0.0,
                "unique_users": 0,
                "unique_merchants": 0,
                "total_amount": 0.0,
                "time_span_hours": 0.0,
            }

        # Calculate statistics
        timestamps = [t.timestamp for t in area_transactions]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600

        return {
            "transaction_count": len(area_transactions),
            "avg_risk_score": np.mean([t.risk_score for t in area_transactions]),
            "unique_users": len(set(t.user_id for t in area_transactions)),
            "unique_merchants": len(set(t.merchant_id for t in area_transactions)),
            "total_amount": sum(t.amount for t in area_transactions),
            "time_span_hours": time_span,
            "transactions_per_hour": len(area_transactions) / max(1, time_span),
        }

    def create_risk_zone(
        self, coordinates: List[Tuple[float, float]], zone_data: Dict[str, Any]
    ) -> str:
        """Create geographic risk zone"""

        zone_id = str(uuid.uuid4())

        # Create polygon from coordinates
        polygon_coords = [(lon, lat) for lat, lon in coordinates]
        geometry = Polygon(polygon_coords)

        risk_zone = RiskZone(
            zone_id=zone_id,
            name=zone_data.get("name", f"Risk Zone {zone_id[:8]}"),
            geometry=geometry,
            risk_level=zone_data.get("risk_level", "medium"),
            risk_factors=zone_data.get("risk_factors", []),
            active_incidents=zone_data.get("active_incidents", 0),
            historical_incidents=zone_data.get("historical_incidents", 0),
            population_density=zone_data.get("population_density"),
            economic_indicators=zone_data.get("economic_indicators", {}),
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
        )

        self.risk_zones[zone_id] = risk_zone

        self.metrics["risk_zones_identified"] += 1
        logger.info(f"Created risk zone {zone_id}: {risk_zone.name}")

        return zone_id

    def export_map_data(
        self, center_lat: float, center_lon: float, radius_km: float = 10.0
    ) -> Dict[str, Any]:
        """Export data for map visualization"""

        # Find data in area
        area_transactions = self._find_nearby_transactions(
            center_lat, center_lon, radius_km
        )

        # Find clusters in area
        area_clusters = []
        for cluster in self.clusters.values():
            distance = self._calculate_distance(
                center_lat, center_lon, cluster.center_lat, cluster.center_lon
            )
            if distance <= radius_km:
                area_clusters.append(cluster)

        # Find risk zones in area
        center_point = Point(center_lon, center_lat)
        area_zones = []
        for zone in self.risk_zones.values():
            if (
                zone.geometry.distance(center_point) <= radius_km / 111.0
            ):  # Rough conversion
                area_zones.append(zone)

        return {
            "center": {"latitude": center_lat, "longitude": center_lon},
            "radius_km": radius_km,
            "transactions": [
                {
                    "id": t.transaction_id,
                    "latitude": t.location.latitude,
                    "longitude": t.location.longitude,
                    "risk_score": t.risk_score,
                    "amount": t.amount,
                    "timestamp": t.timestamp.isoformat(),
                    "user_id": t.user_id,
                    "merchant_id": t.merchant_id,
                }
                for t in area_transactions
            ],
            "clusters": [
                {
                    "id": c.cluster_id,
                    "center_lat": c.center_lat,
                    "center_lon": c.center_lon,
                    "radius_km": c.radius_km,
                    "risk_level": c.risk_level,
                    "confidence": c.confidence_score,
                    "type": c.cluster_type,
                    "transaction_count": len(c.transaction_ids),
                }
                for c in area_clusters
            ],
            "risk_zones": [
                {
                    "id": z.zone_id,
                    "name": z.name,
                    "coordinates": list(z.geometry.exterior.coords),
                    "risk_level": z.risk_level,
                    "active_incidents": z.active_incidents,
                }
                for z in area_zones
            ],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get analysis metrics"""
        return self.metrics.copy()


# =====================================================
# USAGE EXAMPLE
# =====================================================


def main():
    """Example usage of geospatial analysis engine"""

    # Initialize engine
    engine = GeospatialAnalysisEngine()

    # Sample transaction data
    sample_transactions = [
        {
            "transaction_id": "tx_001",
            "user_id": "user_123",
            "merchant_id": "merchant_456",
            "amount": 150.00,
            "timestamp": "2024-01-15T14:30:00Z",
            "location": {"latitude": 40.7128, "longitude": -74.0060, "accuracy": 5.0},
        },
        {
            "transaction_id": "tx_002",
            "user_id": "user_123",
            "merchant_id": "merchant_789",
            "amount": 75.00,
            "timestamp": "2024-01-15T16:45:00Z",
            "location": {"latitude": 40.7589, "longitude": -73.9851, "accuracy": 8.0},
        },
    ]

    # Add transactions
    for tx_data in sample_transactions:
        engine.add_transaction(tx_data)

    # Detect clusters
    clusters = engine.detect_geographic_clusters(
        "dbscan", {"eps_km": 1.0, "min_samples": 2, "time_window_hours": 24}
    )

    print(f"Detected {len(clusters)} clusters")

    # Get location risk score
    risk_score = engine.get_location_risk_score(40.7128, -74.0060)
    print(f"Location risk score: {risk_score:.3f}")

    # Get travel pattern
    travel_pattern = engine.get_user_travel_pattern("user_123")
    if travel_pattern:
        print(f"User travel radius: {travel_pattern.travel_radius:.1f} km")
        print(f"Suspicious movements: {len(travel_pattern.suspicious_movements)}")

    # Export map data
    map_data = engine.export_map_data(40.7128, -74.0060, radius_km=5.0)
    print(
        f"Map data: {len(map_data['transactions'])} transactions, {len(map_data['clusters'])} clusters"
    )

    # Get metrics
    metrics = engine.get_metrics()
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
