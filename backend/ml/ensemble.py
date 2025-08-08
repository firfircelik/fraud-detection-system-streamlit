#!/usr/bin/env python3
"""
🧠 Lightweight Ensemble Model Manager
Simplified ensemble learning system without heavy TensorFlow dependencies
"""

import json
import logging
import pickle
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Individual model prediction result"""

    model_name: str
    fraud_probability: float
    confidence: float
    inference_time_ms: float
    feature_importance: Dict[str, float]
    model_version: str
    timestamp: datetime


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""

    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    confidence: float
    individual_predictions: List[ModelPrediction]
    model_weights: Dict[str, float]
    ensemble_method: str
    total_inference_time_ms: float
    feature_importance: Dict[str, float]
    explanation: Dict[str, Any]
    timestamp: datetime


@dataclass
class ModelPerformanceMetrics:
    """Model performance tracking"""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    avg_inference_time_ms: float
    prediction_count: int
    last_updated: datetime
    drift_score: float
    is_healthy: bool


class LightweightModelPerformanceTracker:
    """Lightweight performance tracker without Redis dependency"""

    def __init__(self):
        self.performance_cache = {}
        self.prediction_history = {}
        self.drift_threshold = 0.1
        self.performance_window = timedelta(hours=24)

    def track_prediction(
        self,
        model_name: str,
        prediction: float,
        actual: Optional[bool] = None,
        inference_time_ms: float = 0.0,
    ) -> None:
        """Track a single prediction for performance monitoring"""

        try:
            # Store prediction in memory
            if model_name not in self.prediction_history:
                self.prediction_history[model_name] = []

            prediction_data = {
                "prediction": prediction,
                "actual": actual,
                "inference_time_ms": inference_time_ms,
                "timestamp": datetime.now(),
            }

            self.prediction_history[model_name].append(prediction_data)

            # Keep only recent predictions (last 1000)
            if len(self.prediction_history[model_name]) > 1000:
                self.prediction_history[model_name] = self.prediction_history[
                    model_name
                ][-1000:]

            # Update performance metrics if we have actual result
            if actual is not None:
                self._update_performance_metrics(model_name)

        except Exception as e:
            logger.error(f"Failed to track prediction for {model_name}: {e}")

    def _update_performance_metrics(self, model_name: str) -> None:
        """Update performance metrics for a model"""

        try:
            recent_predictions = self._get_recent_predictions(model_name)

            if len(recent_predictions) < 10:  # Need minimum predictions for metrics
                return

            # Calculate metrics
            predictions_with_actual = [
                p for p in recent_predictions if p["actual"] is not None
            ]

            if len(predictions_with_actual) < 10:
                return

            y_true = [p["actual"] for p in predictions_with_actual]
            y_pred = [
                1 if p["prediction"] > 0.5 else 0 for p in predictions_with_actual
            ]
            y_prob = [p["prediction"] for p in predictions_with_actual]

            # Calculate performance metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc_roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5

            # Calculate average inference time
            inference_times = [p["inference_time_ms"] for p in recent_predictions]
            avg_inference_time = np.mean(inference_times)

            # Calculate drift score (simplified)
            drift_score = self._calculate_drift_score(recent_predictions)

            # Create performance metrics
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                avg_inference_time_ms=avg_inference_time,
                prediction_count=len(recent_predictions),
                last_updated=datetime.now(),
                drift_score=drift_score,
                is_healthy=accuracy > 0.8 and drift_score < self.drift_threshold,
            )

            # Store in cache
            self.performance_cache[model_name] = metrics

            # Alert if performance degraded
            if not metrics.is_healthy:
                logger.warning(
                    f"Performance degradation detected for {model_name}: accuracy={accuracy:.3f}, drift={drift_score:.3f}"
                )

        except Exception as e:
            logger.error(f"Failed to update performance metrics for {model_name}: {e}")

    def _get_recent_predictions(self, model_name: str, limit: int = 1000) -> List[Dict]:
        """Get recent predictions for a model"""

        if model_name not in self.prediction_history:
            return []

        return self.prediction_history[model_name][-limit:]

    def _calculate_drift_score(self, predictions: List[Dict]) -> float:
        """Calculate drift score for model predictions"""

        try:
            if len(predictions) < 100:
                return 0.0

            # Split predictions into recent and historical
            mid_point = len(predictions) // 2
            recent_preds = [p["prediction"] for p in predictions[:mid_point]]
            historical_preds = [p["prediction"] for p in predictions[mid_point:]]

            # Calculate distribution difference (simplified KL divergence)
            recent_mean = np.mean(recent_preds)
            historical_mean = np.mean(historical_preds)

            drift_score = abs(recent_mean - historical_mean)
            return min(drift_score, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.error(f"Failed to calculate drift score: {e}")
            return 0.0

    def get_model_performance(
        self, model_name: str
    ) -> Optional[ModelPerformanceMetrics]:
        """Get current performance metrics for a model"""
        return self.performance_cache.get(model_name)

    def get_all_model_performance(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get performance metrics for all models"""
        return self.performance_cache.copy()


class LightweightEnsembleManager:
    """
    Lightweight ensemble model manager using scikit-learn models
    """

    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_metadata = {}
        self.performance_tracker = LightweightModelPerformanceTracker()
        self.ensemble_methods = ["weighted_average", "voting", "dynamic"]
        self.current_ensemble_method = "dynamic"
        self.min_confidence_threshold = 0.7
        self.model_lock = threading.Lock()

        # Initialize default models
        self._initialize_default_models()

        logger.info("LightweightEnsembleManager initialized successfully")

    def _initialize_default_models(self) -> None:
        """Initialize default scikit-learn models"""

        try:
            logger.info("Initializing lightweight ML models...")

            # Initialize scikit-learn models (much faster and lighter)
            self.models = {
                "random_forest": {
                    "model": RandomForestClassifier(n_estimators=100, random_state=42),
                    "type": "tree_ensemble",
                    "version": "1.0.0",
                    "is_active": True,
                    "trained": False,
                },
                "logistic_regression": {
                    "model": LogisticRegression(random_state=42),
                    "type": "linear_model",
                    "version": "1.0.0",
                    "is_active": True,
                    "trained": False,
                },
                "isolation_forest": {
                    "model": IsolationForest(contamination=0.1, random_state=42),
                    "type": "anomaly_detection",
                    "version": "1.0.0",
                    "is_active": True,
                    "trained": False,
                },
                "svm": {
                    "model": SVC(probability=True, random_state=42),
                    "type": "support_vector",
                    "version": "1.0.0",
                    "is_active": True,
                    "trained": False,
                },
            }

            # Initialize equal weights
            self.model_weights = {
                name: 1.0 / len(self.models) for name in self.models.keys()
            }

            # Train models with synthetic data for demo
            self._train_with_synthetic_data()

            logger.info(f"Initialized {len(self.models)} lightweight models")

        except Exception as e:
            logger.error(f"Failed to initialize default models: {e}")
            raise

    def _train_with_synthetic_data(self) -> None:
        """Train models with synthetic data for demonstration"""

        try:
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 1000
            n_features = 50

            # Create features
            X = np.random.random((n_samples, n_features))

            # Create labels with some pattern
            fraud_probability = (
                X[:, 0] * 0.3  # Amount-like feature
                + X[:, 1] * 0.2  # Time-like feature
                + X[:, 2] * 0.1  # User-like feature
                + np.random.random(n_samples) * 0.4
            )
            y = (fraud_probability > 0.5).astype(int)

            # Train each model
            for name, model_info in self.models.items():
                try:
                    model = model_info["model"]

                    if name == "isolation_forest":
                        # Isolation Forest is unsupervised
                        model.fit(X)
                    else:
                        # Supervised models
                        model.fit(X, y)

                    model_info["trained"] = True
                    logger.info(f"Trained {name} model successfully")

                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    model_info["is_active"] = False

        except Exception as e:
            logger.error(f"Failed to train models with synthetic data: {e}")

    def predict_single_model(
        self, model_name: str, features: np.ndarray
    ) -> ModelPrediction:
        """Get prediction from a single model"""

        start_time = time.time()

        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")

            model_info = self.models[model_name]
            if not model_info["is_active"] or not model_info["trained"]:
                raise ValueError(f"Model '{model_name}' is not active or trained")

            model = model_info["model"]

            # Make prediction
            if model_name == "isolation_forest":
                # Isolation Forest returns -1 for outliers, 1 for inliers
                anomaly_score = model.decision_function(features)[0]
                # Convert to probability (0-1 range)
                fraud_probability = max(0.0, min(1.0, (1 - anomaly_score) / 2))
            else:
                # Regular classifiers
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(features)[0]
                    fraud_probability = proba[1] if len(proba) > 1 else proba[0]
                else:
                    # Fallback for models without predict_proba
                    prediction = model.predict(features)[0]
                    fraud_probability = float(prediction)

            # Calculate confidence (simplified)
            confidence = (
                abs(fraud_probability - 0.5) * 2
            )  # Distance from 0.5, scaled to 0-1

            # Calculate feature importance (simplified)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = {
                    f"feature_{i}": float(imp) for i, imp in enumerate(importances[:10])
                }
            else:
                feature_importance = {
                    f"feature_{i}": np.random.random()
                    for i in range(min(10, features.shape[1]))
                }

            inference_time_ms = (time.time() - start_time) * 1000

            # Track prediction
            self.performance_tracker.track_prediction(
                model_name, fraud_probability, None, inference_time_ms
            )

            return ModelPrediction(
                model_name=model_name,
                fraud_probability=fraud_probability,
                confidence=confidence,
                inference_time_ms=inference_time_ms,
                feature_importance=feature_importance,
                model_version=model_info["version"],
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {e}")
            # Return default prediction on error
            return ModelPrediction(
                model_name=model_name,
                fraud_probability=0.5,
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                feature_importance={},
                model_version=self.models.get(model_name, {}).get("version", "unknown"),
                timestamp=datetime.now(),
            )

    def predict_ensemble(
        self, features: np.ndarray, transaction_id: str = None
    ) -> EnsemblePrediction:
        """Get ensemble prediction from all active models"""

        start_time = time.time()
        individual_predictions = []

        # Get predictions from all active models
        active_models = [
            name
            for name, info in self.models.items()
            if info["is_active"] and info["trained"]
        ]

        for model_name in active_models:
            try:
                prediction = self.predict_single_model(model_name, features)
                individual_predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to get prediction from {model_name}: {e}")

        if not individual_predictions:
            raise ValueError("No active models available for prediction")

        # Calculate ensemble prediction
        ensemble_result = self._calculate_ensemble_prediction(
            individual_predictions, transaction_id
        )

        total_time_ms = (time.time() - start_time) * 1000
        ensemble_result.total_inference_time_ms = total_time_ms

        return ensemble_result

    def _calculate_ensemble_prediction(
        self, predictions: List[ModelPrediction], transaction_id: str = None
    ) -> EnsemblePrediction:
        """Calculate final ensemble prediction from individual predictions"""

        if self.current_ensemble_method == "weighted_average":
            ensemble_prob = self._weighted_average_ensemble(predictions)
        elif self.current_ensemble_method == "voting":
            ensemble_prob = self._voting_ensemble(predictions)
        elif self.current_ensemble_method == "dynamic":
            ensemble_prob = self._dynamic_ensemble(predictions)
        else:
            ensemble_prob = self._weighted_average_ensemble(predictions)

        # Calculate overall confidence
        confidences = [p.confidence for p in predictions]
        overall_confidence = np.mean(confidences) if confidences else 0.0

        # Determine risk level and decision
        risk_level, decision = self._determine_risk_and_decision(
            ensemble_prob, overall_confidence
        )

        # Calculate aggregated feature importance
        feature_importance = self._aggregate_feature_importance(predictions)

        # Generate explanation
        explanation = self._generate_explanation(predictions, ensemble_prob)

        return EnsemblePrediction(
            transaction_id=transaction_id or f"tx_{int(time.time())}",
            fraud_probability=ensemble_prob,
            risk_level=risk_level,
            decision=decision,
            confidence=overall_confidence,
            individual_predictions=predictions,
            model_weights=self.model_weights.copy(),
            ensemble_method=self.current_ensemble_method,
            total_inference_time_ms=0.0,  # Will be set by caller
            feature_importance=feature_importance,
            explanation=explanation,
            timestamp=datetime.now(),
        )

    def _weighted_average_ensemble(self, predictions: List[ModelPrediction]) -> float:
        """Calculate weighted average ensemble prediction"""

        weighted_sum = 0.0
        total_weight = 0.0

        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 0.0)
            weighted_sum += pred.fraud_probability * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _voting_ensemble(self, predictions: List[ModelPrediction]) -> float:
        """Calculate voting ensemble prediction"""

        votes = [1 if p.fraud_probability > 0.5 else 0 for p in predictions]
        fraud_votes = sum(votes)
        total_votes = len(votes)

        return fraud_votes / total_votes if total_votes > 0 else 0.5

    def _dynamic_ensemble(self, predictions: List[ModelPrediction]) -> float:
        """Calculate dynamic ensemble based on model performance"""

        # Get current performance metrics for weighting
        performance_weights = {}
        total_performance = 0.0

        for pred in predictions:
            performance = self.performance_tracker.get_model_performance(
                pred.model_name
            )
            if performance and performance.is_healthy:
                # Weight by F1 score and confidence
                weight = performance.f1_score * pred.confidence
                performance_weights[pred.model_name] = weight
                total_performance += weight
            else:
                # Default weight for models without performance data
                performance_weights[pred.model_name] = 0.1
                total_performance += 0.1

        # Normalize performance weights
        if total_performance > 0:
            for model_name in performance_weights:
                performance_weights[model_name] /= total_performance

        # Calculate weighted prediction
        weighted_sum = 0.0
        for pred in predictions:
            weight = performance_weights.get(pred.model_name, 0.0)
            weighted_sum += pred.fraud_probability * weight

        return weighted_sum

    def _determine_risk_and_decision(
        self, fraud_probability: float, confidence: float
    ) -> Tuple[str, str]:
        """Determine risk level and decision based on probability and confidence"""

        # Adjust thresholds based on confidence
        confidence_factor = max(0.5, confidence)  # Minimum 0.5 confidence factor

        if fraud_probability >= 0.8 * confidence_factor:
            return "CRITICAL", "DECLINED"
        elif fraud_probability >= 0.6 * confidence_factor:
            return "HIGH", "REVIEW"
        elif fraud_probability >= 0.4 * confidence_factor:
            return "MEDIUM", "REVIEW"
        elif fraud_probability >= 0.2 * confidence_factor:
            return "LOW", "APPROVED"
        else:
            return "MINIMAL", "APPROVED"

    def _aggregate_feature_importance(
        self, predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Aggregate feature importance across all models"""

        aggregated_importance = {}

        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 0.0)

            for feature, importance in pred.feature_importance.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = 0.0
                aggregated_importance[feature] += importance * weight

        return aggregated_importance

    def _generate_explanation(
        self, predictions: List[ModelPrediction], ensemble_prob: float
    ) -> Dict[str, Any]:
        """Generate explanation for the ensemble prediction"""

        explanation = {
            "ensemble_probability": ensemble_prob,
            "model_agreement": self._calculate_model_agreement(predictions),
            "top_contributing_models": self._get_top_contributing_models(predictions),
            "confidence_analysis": self._analyze_confidence(predictions),
            "risk_factors": self._identify_risk_factors(predictions),
        }

        return explanation

    def _calculate_model_agreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate agreement between models"""

        if len(predictions) < 2:
            return 1.0

        probs = [p.fraud_probability for p in predictions]
        std_dev = np.std(probs)

        # Convert standard deviation to agreement score (0-1)
        agreement = max(0.0, 1.0 - (std_dev * 2))  # Scale std dev
        return agreement

    def _get_top_contributing_models(
        self, predictions: List[ModelPrediction]
    ) -> List[Dict]:
        """Get top contributing models to the prediction"""

        contributions = []

        for pred in predictions:
            weight = self.model_weights.get(pred.model_name, 0.0)
            contribution = pred.fraud_probability * weight

            contributions.append(
                {
                    "model_name": pred.model_name,
                    "probability": pred.fraud_probability,
                    "weight": weight,
                    "contribution": contribution,
                    "confidence": pred.confidence,
                }
            )

        # Sort by contribution
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return contributions[:3]  # Top 3 contributors

    def _analyze_confidence(
        self, predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Analyze confidence across models"""

        confidences = [p.confidence for p in predictions]

        return {
            "mean_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "confidence_std": np.std(confidences),
        }

    def _identify_risk_factors(self, predictions: List[ModelPrediction]) -> List[str]:
        """Identify key risk factors from predictions"""

        risk_factors = []

        # Analyze model agreement
        agreement = self._calculate_model_agreement(predictions)
        if agreement < 0.5:
            risk_factors.append("Low model agreement - conflicting signals")

        # Analyze confidence levels
        confidences = [p.confidence for p in predictions]
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.6:
            risk_factors.append("Low prediction confidence")

        # Analyze individual model risks
        high_risk_models = [p for p in predictions if p.fraud_probability > 0.7]
        if len(high_risk_models) > len(predictions) / 2:
            risk_factors.append("Majority of models indicate high fraud risk")

        return risk_factors

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current status of the ensemble"""

        active_models = [
            name
            for name, info in self.models.items()
            if info["is_active"] and info["trained"]
        ]

        status = {
            "total_models": len(self.models),
            "active_models": len(active_models),
            "model_list": active_models,
            "model_weights": self.model_weights.copy(),
            "ensemble_method": self.current_ensemble_method,
            "performance_data": self.performance_tracker.get_all_model_performance(),
        }

        return status


# Usage example and testing
if __name__ == "__main__":
    print("🧪 Testing LightweightEnsembleManager...")

    try:
        # Initialize ensemble manager
        ensemble = LightweightEnsembleManager()

        # Get status
        status = ensemble.get_ensemble_status()
        print(
            f"✅ Ensemble initialized: {status['active_models']}/{status['total_models']} active models"
        )
        print(f"✅ Models: {status['model_list']}")

        # Create sample features
        sample_features = np.random.random((1, 50))

        # Get ensemble prediction
        prediction = ensemble.predict_ensemble(sample_features, "test_transaction_001")

        print(f"\n🧠 Ensemble Prediction Results:")
        print(f"   Fraud Probability: {prediction.fraud_probability:.4f}")
        print(f"   Risk Level: {prediction.risk_level}")
        print(f"   Decision: {prediction.decision}")
        print(f"   Confidence: {prediction.confidence:.4f}")
        print(f"   Total Inference Time: {prediction.total_inference_time_ms:.2f}ms")
        print(f"   Models Used: {len(prediction.individual_predictions)}")

        # Print individual model results
        print(f"\n📊 Individual Model Results:")
        for pred in prediction.individual_predictions:
            print(
                f"   {pred.model_name}: {pred.fraud_probability:.4f} "
                f"(confidence: {pred.confidence:.3f}, "
                f"time: {pred.inference_time_ms:.1f}ms)"
            )

        print(f"\n⚖️ Model Weights: {prediction.model_weights}")
        print(f"🔍 Ensemble Method: {prediction.ensemble_method}")
        print(f"\n🎉 LightweightEnsembleManager test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
