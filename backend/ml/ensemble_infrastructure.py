#!/usr/bin/env python3
"""
Enhanced Ensemble Model Infrastructure
A/B testing framework, dynamic weighting, and advanced ensemble techniques
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics"""

    model_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    log_loss: float
    inference_time_ms: float
    prediction_count: int
    timestamp: datetime


@dataclass
class ABTestResult:
    """A/B test result"""

    test_id: str
    model_a_id: str
    model_b_id: str
    model_a_performance: ModelPerformance
    model_b_performance: ModelPerformance
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    winner: str
    test_duration_hours: float
    sample_size: int
    timestamp: datetime


@dataclass
class EnsembleConfig:
    """Ensemble configuration"""

    ensemble_id: str
    name: str
    models: List[str]
    weights: Dict[str, float]
    aggregation_method: str  # 'weighted_average', 'voting', 'stacking'
    performance_threshold: float
    auto_reweight: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class PredictionResult:
    """Enhanced prediction result with ensemble details"""

    prediction_id: str
    ensemble_id: str
    fraud_probability: float
    decision: str
    risk_level: str
    confidence: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    aggregation_method: str
    processing_time_ms: float
    feature_importance: Dict[str, float]
    explanation: Dict[str, Any]
    timestamp: datetime


class EnhancedEnsembleManager:
    """Advanced ensemble management with A/B testing and dynamic weighting"""

    def __init__(
        self, models_dir: str = "./models", performance_window_hours: int = 24
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)

        # Model storage
        self.models = {}
        self.model_metadata = {}
        self.ensembles = {}

        # Performance tracking
        self.performance_history = {}
        self.performance_window = timedelta(hours=performance_window_hours)

        # A/B testing
        self.active_ab_tests = {}
        self.ab_test_results = {}

        # Dynamic weighting parameters
        self.min_weight = 0.05
        self.max_weight = 0.5
        self.performance_decay = 0.95  # Exponential decay for older performance

        # Thread safety
        self.lock = threading.RLock()

        # Default ensemble configurations
        self.default_ensembles = {
            "conservative": {
                "models": ["random_forest", "logistic_regression"],
                "weights": {"random_forest": 0.6, "logistic_regression": 0.4},
                "aggregation_method": "weighted_average",
                "performance_threshold": 0.85,
            },
            "aggressive": {
                "models": ["xgboost", "lightgbm", "extra_trees"],
                "weights": {"xgboost": 0.4, "lightgbm": 0.4, "extra_trees": 0.2},
                "aggregation_method": "weighted_average",
                "performance_threshold": 0.90,
            },
            "balanced": {
                "models": ["random_forest", "xgboost", "logistic_regression", "svm"],
                "weights": {
                    "random_forest": 0.3,
                    "xgboost": 0.3,
                    "logistic_regression": 0.2,
                    "svm": 0.2,
                },
                "aggregation_method": "weighted_average",
                "performance_threshold": 0.88,
            },
        }

    def initialize_default_models(self):
        """Initialize default ML models"""
        try:
            # Random Forest
            self.models["random_forest"] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            # XGBoost
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )

            # LightGBM
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )

            # Logistic Regression
            self.models["logistic_regression"] = LogisticRegression(
                C=1.0, penalty="l2", solver="liblinear", random_state=42, max_iter=1000
            )

            # Extra Trees
            self.models["extra_trees"] = ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )

            # SVM (for smaller datasets)
            self.models["svm"] = SVC(
                C=1.0, kernel="rbf", probability=True, random_state=42
            )

            # Initialize default ensembles
            for name, config in self.default_ensembles.items():
                self.create_ensemble(
                    name=name,
                    models=config["models"],
                    weights=config["weights"],
                    aggregation_method=config["aggregation_method"],
                    performance_threshold=config["performance_threshold"],
                )

            logger.info("Default models and ensembles initialized")

        except Exception as e:
            logger.error(f"Error initializing default models: {e}")
            raise

    # =====================================================
    # ENSEMBLE MANAGEMENT
    # =====================================================

    def create_ensemble(
        self,
        name: str,
        models: List[str],
        weights: Dict[str, float] = None,
        aggregation_method: str = "weighted_average",
        performance_threshold: float = 0.85,
        auto_reweight: bool = True,
    ) -> str:
        """Create a new ensemble configuration"""

        ensemble_id = f"ensemble_{name}_{int(time.time())}"

        # Validate models exist
        for model_name in models:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

        # Default equal weights if not provided
        if weights is None:
            weight_value = 1.0 / len(models)
            weights = {model: weight_value for model in models}

        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        ensemble_config = EnsembleConfig(
            ensemble_id=ensemble_id,
            name=name,
            models=models,
            weights=normalized_weights,
            aggregation_method=aggregation_method,
            performance_threshold=performance_threshold,
            auto_reweight=auto_reweight,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with self.lock:
            self.ensembles[ensemble_id] = ensemble_config

        logger.info(f"Created ensemble {name} with ID {ensemble_id}")
        return ensemble_id

    def predict_ensemble(
        self, ensemble_id: str, features: np.ndarray, transaction_id: str = None
    ) -> PredictionResult:
        """Make prediction using ensemble"""

        start_time = time.time()

        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} not found")

        ensemble = self.ensembles[ensemble_id]

        # Get individual model predictions
        individual_predictions = {}
        prediction_times = {}

        for model_name in ensemble.models:
            if model_name in self.models:
                model_start = time.time()

                try:
                    if hasattr(self.models[model_name], "predict_proba"):
                        prob = self.models[model_name].predict_proba(features)[0][1]
                    else:
                        prob = self.models[model_name].predict(features)[0]

                    individual_predictions[model_name] = float(prob)
                    prediction_times[model_name] = (time.time() - model_start) * 1000

                except Exception as e:
                    logger.error(f"Error in model {model_name}: {e}")
                    individual_predictions[model_name] = 0.5  # Default probability
                    prediction_times[model_name] = 0

        # Aggregate predictions
        if ensemble.aggregation_method == "weighted_average":
            fraud_probability = self._weighted_average_prediction(
                individual_predictions, ensemble.weights
            )
        elif ensemble.aggregation_method == "voting":
            fraud_probability = self._voting_prediction(individual_predictions)
        elif ensemble.aggregation_method == "stacking":
            fraud_probability = self._stacking_prediction(
                individual_predictions, ensemble.weights
            )
        else:
            fraud_probability = self._weighted_average_prediction(
                individual_predictions, ensemble.weights
            )

        # Determine decision and risk level
        decision, risk_level = self._determine_decision(fraud_probability)

        # Calculate confidence
        confidence = self._calculate_confidence(
            individual_predictions, ensemble.weights
        )

        # Feature importance (simplified - in practice, would aggregate from models)
        feature_importance = self._calculate_ensemble_feature_importance(
            ensemble.models, features
        )

        processing_time = (time.time() - start_time) * 1000

        # Create prediction result
        result = PredictionResult(
            prediction_id=str(uuid.uuid4()),
            ensemble_id=ensemble_id,
            fraud_probability=fraud_probability,
            decision=decision,
            risk_level=risk_level,
            confidence=confidence,
            individual_predictions=individual_predictions,
            model_weights=ensemble.weights,
            aggregation_method=ensemble.aggregation_method,
            processing_time_ms=processing_time,
            feature_importance=feature_importance,
            explanation={
                "method": ensemble.aggregation_method,
                "model_count": len(ensemble.models),
                "prediction_times_ms": prediction_times,
            },
            timestamp=datetime.now(timezone.utc),
        )

        return result

    # =====================================================
    # A/B TESTING FRAMEWORK
    # =====================================================

    def start_ab_test(
        self,
        model_a_id: str,
        model_b_id: str,
        traffic_split: float = 0.5,
        duration_hours: int = 24,
        significance_level: float = 0.05,
    ) -> str:
        """Start A/B test between two models/ensembles"""

        test_id = f"ab_test_{int(time.time())}_{hashlib.md5(f'{model_a_id}_{model_b_id}'.encode()).hexdigest()[:8]}"

        ab_test = {
            "test_id": test_id,
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "traffic_split": traffic_split,
            "duration_hours": duration_hours,
            "significance_level": significance_level,
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc) + timedelta(hours=duration_hours),
            "model_a_results": [],
            "model_b_results": [],
            "status": "active",
        }

        with self.lock:
            self.active_ab_tests[test_id] = ab_test

        logger.info(f"Started A/B test {test_id}: {model_a_id} vs {model_b_id}")
        return test_id

    def route_ab_test_prediction(
        self, test_id: str, features: np.ndarray, actual_label: int = None
    ) -> Tuple[PredictionResult, str]:
        """Route prediction through A/B test and track results"""

        if test_id not in self.active_ab_tests:
            raise ValueError(f"A/B test {test_id} not found")

        ab_test = self.active_ab_tests[test_id]

        # Check if test is still active
        if datetime.now(timezone.utc) > ab_test["end_time"]:
            ab_test["status"] = "completed"
            return self._finalize_ab_test(test_id)

        # Route traffic based on split
        route_to_a = np.random.random() < ab_test["traffic_split"]

        if route_to_a:
            model_id = ab_test["model_a_id"]
            result_list = ab_test["model_a_results"]
            variant = "A"
        else:
            model_id = ab_test["model_b_id"]
            result_list = ab_test["model_b_results"]
            variant = "B"

        # Make prediction
        if model_id in self.ensembles:
            prediction = self.predict_ensemble(model_id, features)
        else:
            # Single model prediction
            prediction = self._predict_single_model(model_id, features)

        # Track result
        result_record = {
            "prediction": prediction.fraud_probability,
            "actual": actual_label,
            "timestamp": datetime.now(timezone.utc),
            "processing_time_ms": prediction.processing_time_ms,
        }

        with self.lock:
            result_list.append(result_record)

        return prediction, variant

    def _finalize_ab_test(self, test_id: str) -> ABTestResult:
        """Finalize A/B test and determine winner"""

        ab_test = self.active_ab_tests[test_id]

        # Calculate performance metrics
        model_a_perf = self._calculate_ab_performance(
            ab_test["model_a_results"], ab_test["model_a_id"]
        )
        model_b_perf = self._calculate_ab_performance(
            ab_test["model_b_results"], ab_test["model_b_id"]
        )

        # Statistical significance test
        a_accuracies = [
            r["prediction"]
            for r in ab_test["model_a_results"]
            if r["actual"] is not None
        ]
        b_accuracies = [
            r["prediction"]
            for r in ab_test["model_b_results"]
            if r["actual"] is not None
        ]

        if len(a_accuracies) > 10 and len(b_accuracies) > 10:
            t_stat, p_value = stats.ttest_ind(a_accuracies, b_accuracies)
            significant = p_value < ab_test["significance_level"]
            confidence_interval = stats.t.interval(
                0.95,
                len(a_accuracies) + len(b_accuracies) - 2,
                loc=np.mean(a_accuracies) - np.mean(b_accuracies),
                scale=stats.sem(a_accuracies + b_accuracies),
            )
        else:
            p_value = 1.0
            significant = False
            confidence_interval = (0, 0)

        # Determine winner
        if significant:
            if model_a_perf.accuracy > model_b_perf.accuracy:
                winner = ab_test["model_a_id"]
            else:
                winner = ab_test["model_b_id"]
        else:
            winner = "inconclusive"

        # Calculate test duration
        duration = (
            datetime.now(timezone.utc) - ab_test["start_time"]
        ).total_seconds() / 3600

        result = ABTestResult(
            test_id=test_id,
            model_a_id=ab_test["model_a_id"],
            model_b_id=ab_test["model_b_id"],
            model_a_performance=model_a_perf,
            model_b_performance=model_b_perf,
            statistical_significance=significant,
            p_value=p_value,
            confidence_interval=confidence_interval,
            winner=winner,
            test_duration_hours=duration,
            sample_size=len(ab_test["model_a_results"])
            + len(ab_test["model_b_results"]),
            timestamp=datetime.now(timezone.utc),
        )

        # Store result and clean up
        with self.lock:
            self.ab_test_results[test_id] = result
            ab_test["status"] = "completed"

        logger.info(f"A/B test {test_id} completed. Winner: {winner}")
        return result

    # =====================================================
    # DYNAMIC WEIGHTING
    # =====================================================

    def update_ensemble_weights(
        self, ensemble_id: str, performance_data: Dict[str, List[float]]
    ):
        """Update ensemble weights based on recent performance"""

        if ensemble_id not in self.ensembles:
            raise ValueError(f"Ensemble {ensemble_id} not found")

        ensemble = self.ensembles[ensemble_id]

        if not ensemble.auto_reweight:
            return

        # Calculate performance-based weights
        new_weights = {}
        total_performance = 0

        for model_name in ensemble.models:
            if model_name in performance_data and performance_data[model_name]:
                # Calculate exponentially weighted average performance
                performances = performance_data[model_name]
                weights = [self.performance_decay**i for i in range(len(performances))]
                weighted_perf = np.average(performances, weights=weights)

                # Ensure minimum weight
                weighted_perf = max(weighted_perf, self.min_weight)
                new_weights[model_name] = weighted_perf
                total_performance += weighted_perf
            else:
                # Default weight for models without recent performance data
                new_weights[model_name] = self.min_weight
                total_performance += self.min_weight

        # Normalize weights
        if total_performance > 0:
            for model_name in new_weights:
                new_weights[model_name] = max(
                    min(new_weights[model_name] / total_performance, self.max_weight),
                    self.min_weight,
                )

        # Update ensemble
        with self.lock:
            ensemble.weights = new_weights
            ensemble.updated_at = datetime.now(timezone.utc)

        logger.info(f"Updated weights for ensemble {ensemble_id}: {new_weights}")

    def track_model_performance(
        self,
        model_id: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        auc_roc: float,
        log_loss_val: float,
        inference_time_ms: float,
    ):
        """Track model performance metrics"""

        performance = ModelPerformance(
            model_id=model_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            log_loss=log_loss_val,
            inference_time_ms=inference_time_ms,
            prediction_count=1,
            timestamp=datetime.now(timezone.utc),
        )

        with self.lock:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []

            self.performance_history[model_id].append(performance)

            # Keep only recent performance data
            cutoff_time = datetime.now(timezone.utc) - self.performance_window
            self.performance_history[model_id] = [
                p
                for p in self.performance_history[model_id]
                if p.timestamp > cutoff_time
            ]

    # =====================================================
    # HELPER METHODS
    # =====================================================

    def _weighted_average_prediction(
        self, predictions: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted average prediction"""
        total_weight = 0
        weighted_sum = 0

        for model_name, prediction in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                weighted_sum += prediction * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _voting_prediction(
        self, predictions: Dict[str, float], threshold: float = 0.5
    ) -> float:
        """Calculate majority voting prediction"""
        votes = sum(1 for pred in predictions.values() if pred > threshold)
        return votes / len(predictions)

    def _stacking_prediction(
        self, predictions: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate stacked prediction (meta-learning approach)"""
        # Simplified stacking - in practice, would use trained meta-model
        return self._weighted_average_prediction(predictions, weights)

    def _determine_decision(self, fraud_probability: float) -> Tuple[str, str]:
        """Determine decision and risk level based on fraud probability"""
        if fraud_probability >= 0.8:
            return "DECLINED", "CRITICAL"
        elif fraud_probability >= 0.6:
            return "REVIEW", "HIGH"
        elif fraud_probability >= 0.4:
            return "REVIEW", "MEDIUM"
        elif fraud_probability >= 0.2:
            return "APPROVED", "LOW"
        else:
            return "APPROVED", "MINIMAL"

    def _calculate_confidence(
        self, predictions: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) < 2:
            return 0.5

        weighted_predictions = []
        for model_name, prediction in predictions.items():
            if model_name in weights:
                weighted_predictions.append(prediction * weights[model_name])

        if not weighted_predictions:
            return 0.5

        # Calculate variance (lower variance = higher confidence)
        variance = np.var(weighted_predictions)
        confidence = 1.0 / (1.0 + variance)

        return min(max(confidence, 0.0), 1.0)

    def _calculate_ensemble_feature_importance(
        self, model_names: List[str], features: np.ndarray
    ) -> Dict[str, float]:
        """Calculate aggregated feature importance"""
        # Simplified implementation - would need proper feature names and model introspection
        n_features = features.shape[1] if len(features.shape) > 1 else len(features)

        # Generate dummy importance scores
        importance = {}
        for i in range(min(n_features, 10)):  # Top 10 features
            importance[f"feature_{i}"] = np.random.random()

        return importance

    def _predict_single_model(
        self, model_id: str, features: np.ndarray
    ) -> PredictionResult:
        """Make prediction with single model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        start_time = time.time()

        model = self.models[model_id]
        if hasattr(model, "predict_proba"):
            fraud_probability = float(model.predict_proba(features)[0][1])
        else:
            fraud_probability = float(model.predict(features)[0])

        decision, risk_level = self._determine_decision(fraud_probability)
        processing_time = (time.time() - start_time) * 1000

        return PredictionResult(
            prediction_id=str(uuid.uuid4()),
            ensemble_id=model_id,
            fraud_probability=fraud_probability,
            decision=decision,
            risk_level=risk_level,
            confidence=0.8,  # Single model default confidence
            individual_predictions={model_id: fraud_probability},
            model_weights={model_id: 1.0},
            aggregation_method="single_model",
            processing_time_ms=processing_time,
            feature_importance={},
            explanation={"method": "single_model"},
            timestamp=datetime.now(timezone.utc),
        )

    def _calculate_ab_performance(
        self, results: List[Dict], model_id: str
    ) -> ModelPerformance:
        """Calculate performance metrics for A/B test"""
        if not results:
            return ModelPerformance(
                model_id=model_id,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=0.0,
                log_loss=0.0,
                inference_time_ms=0.0,
                prediction_count=0,
                timestamp=datetime.now(timezone.utc),
            )

        # Filter results with actual labels
        labeled_results = [r for r in results if r["actual"] is not None]

        if not labeled_results:
            avg_time = np.mean([r["processing_time_ms"] for r in results])
            return ModelPerformance(
                model_id=model_id,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                auc_roc=0.0,
                log_loss=0.0,
                inference_time_ms=avg_time,
                prediction_count=len(results),
                timestamp=datetime.now(timezone.utc),
            )

        # Calculate metrics
        predictions = [r["prediction"] > 0.5 for r in labeled_results]
        actuals = [r["actual"] for r in labeled_results]
        probabilities = [r["prediction"] for r in labeled_results]

        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions, zero_division=0)
        f1 = f1_score(actuals, predictions, zero_division=0)

        try:
            auc = roc_auc_score(actuals, probabilities)
        except:
            auc = 0.5

        try:
            logloss = log_loss(actuals, probabilities)
        except:
            logloss = 1.0

        avg_time = np.mean([r["processing_time_ms"] for r in results])

        return ModelPerformance(
            model_id=model_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            log_loss=logloss,
            inference_time_ms=avg_time,
            prediction_count=len(results),
            timestamp=datetime.now(timezone.utc),
        )

    # =====================================================
    # MANAGEMENT METHODS
    # =====================================================

    def get_ensemble_status(self, ensemble_id: str) -> Dict[str, Any]:
        """Get ensemble status and performance summary"""
        if ensemble_id not in self.ensembles:
            return {}

        ensemble = self.ensembles[ensemble_id]

        # Get recent performance for each model
        model_performance = {}
        for model_name in ensemble.models:
            if model_name in self.performance_history:
                recent_perf = self.performance_history[model_name][
                    -5:
                ]  # Last 5 records
                if recent_perf:
                    avg_accuracy = np.mean([p.accuracy for p in recent_perf])
                    avg_time = np.mean([p.inference_time_ms for p in recent_perf])
                    model_performance[model_name] = {
                        "accuracy": avg_accuracy,
                        "inference_time_ms": avg_time,
                        "sample_count": len(recent_perf),
                    }

        return {
            "ensemble_id": ensemble_id,
            "name": ensemble.name,
            "models": ensemble.models,
            "weights": ensemble.weights,
            "aggregation_method": ensemble.aggregation_method,
            "performance_threshold": ensemble.performance_threshold,
            "auto_reweight": ensemble.auto_reweight,
            "created_at": ensemble.created_at.isoformat(),
            "updated_at": ensemble.updated_at.isoformat(),
            "model_performance": model_performance,
        }

    def list_active_ab_tests(self) -> List[Dict[str, Any]]:
        """List all active A/B tests"""
        active_tests = []

        for test_id, test_data in self.active_ab_tests.items():
            if test_data["status"] == "active":
                active_tests.append(
                    {
                        "test_id": test_id,
                        "model_a_id": test_data["model_a_id"],
                        "model_b_id": test_data["model_b_id"],
                        "traffic_split": test_data["traffic_split"],
                        "start_time": test_data["start_time"].isoformat(),
                        "end_time": test_data["end_time"].isoformat(),
                        "sample_size_a": len(test_data["model_a_results"]),
                        "sample_size_b": len(test_data["model_b_results"]),
                    }
                )

        return active_tests

    def get_ab_test_results(self) -> List[ABTestResult]:
        """Get all completed A/B test results"""
        return list(self.ab_test_results.values())

    def save_ensemble_config(self, filepath: str):
        """Save ensemble configurations to file"""
        config_data = {}
        for ensemble_id, ensemble in self.ensembles.items():
            config_data[ensemble_id] = asdict(ensemble)

        with open(filepath, "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info(f"Ensemble configurations saved to {filepath}")

    def load_ensemble_config(self, filepath: str):
        """Load ensemble configurations from file"""
        with open(filepath, "r") as f:
            config_data = json.load(f)

        for ensemble_id, config in config_data.items():
            # Convert datetime strings back to datetime objects
            config["created_at"] = datetime.fromisoformat(config["created_at"])
            config["updated_at"] = datetime.fromisoformat(config["updated_at"])

            ensemble = EnsembleConfig(**config)
            self.ensembles[ensemble_id] = ensemble

        logger.info(f"Ensemble configurations loaded from {filepath}")


# =====================================================
# USAGE EXAMPLE
# =====================================================


def main():
    """Example usage of Enhanced Ensemble Manager"""

    # Initialize ensemble manager
    ensemble_manager = EnhancedEnsembleManager()
    ensemble_manager.initialize_default_models()

    # Example features (would be real transaction features in practice)
    sample_features = np.random.random((1, 50))

    # Make ensemble prediction
    result = ensemble_manager.predict_ensemble("ensemble_balanced_*", sample_features)
    print(f"Fraud probability: {result.fraud_probability:.3f}")
    print(f"Decision: {result.decision}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")

    # Start A/B test
    test_id = ensemble_manager.start_ab_test(
        "ensemble_conservative_*", "ensemble_aggressive_*", duration_hours=1
    )

    print(f"Started A/B test: {test_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
