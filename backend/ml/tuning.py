#!/usr/bin/env python3
"""
🎯 Hyperparameter Tuning Module
Advanced hyperparameter optimization using Optuna
"""

import json
import logging
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Advanced hyperparameter tuning with Optuna"""

    def __init__(self, n_trials: int = 100, cv_folds: int = 5):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.studies = {}
        self.best_params = {}

    def tune_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters"""

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": 42,
            }

            model = RandomForestClassifier(**params)
            cv_scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=make_scorer(f1_score), n_jobs=-1
            )
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.studies["random_forest"] = study
        self.best_params["random_forest"] = study.best_params

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }

    def tune_logistic_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune Logistic Regression hyperparameters"""

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "penalty": trial.suggest_categorical(
                    "penalty", ["l1", "l2", "elasticnet"]
                ),
                "solver": "saga",  # Supports all penalties
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "random_state": 42,
            }

            if params["penalty"] == "elasticnet":
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

            model = LogisticRegression(**params)
            cv_scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=make_scorer(f1_score), n_jobs=-1
            )
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.studies["logistic_regression"] = study
        self.best_params["logistic_regression"] = study.best_params

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }

    def tune_svm(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Tune SVM hyperparameters"""

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["rbf", "poly", "sigmoid"]
                ),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "probability": True,
                "random_state": 42,
            }

            if params["kernel"] == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)

            model = SVC(**params)
            cv_scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring=make_scorer(f1_score), n_jobs=-1
            )
            return cv_scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.studies["svm"] = study
        self.best_params["svm"] = study.best_params

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
        }

    def tune_all_models(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Tune all models and return results"""

        results = {}

        logger.info("Starting hyperparameter tuning for all models...")

        # Tune Random Forest
        logger.info("Tuning Random Forest...")
        results["random_forest"] = self.tune_random_forest(X, y)

        # Tune Logistic Regression
        logger.info("Tuning Logistic Regression...")
        results["logistic_regression"] = self.tune_logistic_regression(X, y)

        # Tune SVM
        logger.info("Tuning SVM...")
        results["svm"] = self.tune_svm(X, y)

        logger.info("Hyperparameter tuning completed!")

        return results

    def get_tuning_summary(self) -> Dict[str, Any]:
        """Get summary of all tuning results"""

        summary = {
            "timestamp": datetime.now().isoformat(),
            "models_tuned": list(self.best_params.keys()),
            "best_parameters": self.best_params.copy(),
            "study_statistics": {},
        }

        for model_name, study in self.studies.items():
            summary["study_statistics"][model_name] = {
                "n_trials": len(study.trials),
                "best_score": study.best_value,
                "study_direction": study.direction.name,
            }

        return summary

    def save_results(self, filepath: str):
        """Save tuning results to file"""

        results = {
            "best_params": self.best_params,
            "tuning_summary": self.get_tuning_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Tuning results saved to {filepath}")


# Usage example
if __name__ == "__main__":
    print("🎯 Testing Hyperparameter Tuner...")

    try:
        # Generate sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 20
        X = np.random.random((n_samples, n_features))
        y = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

        # Initialize tuner
        tuner = HyperparameterTuner(n_trials=20)  # Reduced for testing

        # Tune Random Forest only for quick test
        print("Tuning Random Forest...")
        rf_results = tuner.tune_random_forest(X, y)

        print("✅ Hyperparameter tuning completed!")
        print(f"   Best RF score: {rf_results['best_score']:.4f}")
        print(f"   Best RF params: {rf_results['best_params']}")
        print(f"   Trials completed: {rf_results['n_trials']}")

        print("\n🎉 Hyperparameter Tuner test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
