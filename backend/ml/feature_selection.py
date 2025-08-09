#!/usr/bin/env python3
"""
Automated Feature Selection Pipeline
Advanced feature selection with importance tracking and correlation analysis
"""

import asyncio
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance information"""

    feature_name: str
    importance_score: float
    method: str  # 'random_forest', 'mutual_info', 'correlation', etc.
    model_type: str
    timestamp: datetime
    dataset_id: str


@dataclass
class FeatureSelectionResult:
    """Feature selection result"""

    selection_id: str
    method: str
    selected_features: List[str]
    removed_features: List[str]
    importance_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    timestamp: datetime
    parameters: Dict[str, Any]


@dataclass
class CorrelationAnalysis:
    """Correlation analysis result"""

    feature_pairs: List[Tuple[str, str]]
    correlation_coefficient: float
    p_value: float
    correlation_type: str  # 'pearson', 'spearman'
    is_redundant: bool
    redundancy_threshold: float


class AutomatedFeatureSelector:
    """Automated feature selection with multiple algorithms"""

    def __init__(self, postgres_dsn: str = None):
        self.postgres_dsn = (
            postgres_dsn
            or "postgresql://fraud_admin:FraudDetection2024!@localhost:5432/fraud_detection"
        )
        self.pool = None

        # Feature selection methods
        self.selection_methods = {
            "variance_threshold": self._variance_threshold_selection,
            "correlation_filter": self._correlation_filter_selection,
            "univariate_selection": self._univariate_selection,
            "recursive_elimination": self._recursive_elimination_selection,
            "lasso_selection": self._lasso_selection,
            "random_forest_importance": self._random_forest_selection,
            "mutual_information": self._mutual_information_selection,
            "pca_selection": self._pca_selection,
        }

        # Default parameters
        self.default_params = {
            "variance_threshold": {"threshold": 0.01},
            "correlation_threshold": 0.95,
            "k_best_features": 50,
            "cv_folds": 5,
            "random_state": 42,
        }

        # Feature importance tracking
        self.feature_importance_history = {}
        self.correlation_cache = {}

    async def initialize(self):
        """Initialize database connection"""
        self.pool = await asyncpg.create_pool(
            self.postgres_dsn, min_size=5, max_size=20, command_timeout=60
        )

        await self._create_feature_selection_tables()
        logger.info("Automated Feature Selector initialized")

    async def _create_feature_selection_tables(self):
        """Create tables for feature selection tracking"""
        async with self.pool.acquire() as conn:
            # Feature importance tracking table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_importance_tracking (
                    id BIGSERIAL PRIMARY KEY,
                    feature_name VARCHAR(255) NOT NULL,
                    importance_score DOUBLE PRECISION NOT NULL,
                    method VARCHAR(100) NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    dataset_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'
                )
            """
            )

            # Feature selection results table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_selection_results (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    selection_id VARCHAR(100) UNIQUE NOT NULL,
                    method VARCHAR(100) NOT NULL,
                    selected_features TEXT[] NOT NULL,
                    removed_features TEXT[] NOT NULL,
                    importance_scores JSONB NOT NULL,
                    performance_metrics JSONB NOT NULL,
                    correlation_matrix JSONB,
                    parameters JSONB NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Feature correlation analysis table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_correlations (
                    id BIGSERIAL PRIMARY KEY,
                    feature_1 VARCHAR(255) NOT NULL,
                    feature_2 VARCHAR(255) NOT NULL,
                    correlation_coefficient DOUBLE PRECISION NOT NULL,
                    p_value DOUBLE PRECISION,
                    correlation_type VARCHAR(50) NOT NULL,
                    is_redundant BOOLEAN DEFAULT FALSE,
                    redundancy_threshold DOUBLE PRECISION,
                    dataset_id VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """
            )

            # Create indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_importance_name ON feature_importance_tracking(feature_name)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_importance_method ON feature_importance_tracking(method)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_selection_method ON feature_selection_results(method)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_correlations_features ON feature_correlations(feature_1, feature_2)"
            )

    # =====================================================
    # MAIN FEATURE SELECTION METHODS
    # =====================================================

    async def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str] = None,
        target_features: int = 50,
    ) -> FeatureSelectionResult:
        """
        Perform automated feature selection using multiple methods
        """
        if methods is None:
            methods = [
                "correlation_filter",
                "random_forest_importance",
                "mutual_information",
            ]

        logger.info(f"Starting feature selection with methods: {methods}")

        # Store original features
        original_features = list(X.columns)
        current_features = original_features.copy()

        # Results tracking
        method_results = {}
        combined_importance = {}

        # Apply each selection method
        for method in methods:
            if method in self.selection_methods:
                logger.info(f"Applying {method} feature selection")

                try:
                    result = await self.selection_methods[method](
                        X[current_features], y, target_features
                    )

                    method_results[method] = result

                    # Update combined importance scores
                    for feature, score in result["importance_scores"].items():
                        if feature in combined_importance:
                            combined_importance[feature] += score
                        else:
                            combined_importance[feature] = score

                    # Update current features for next method
                    if "selected_features" in result:
                        current_features = result["selected_features"]

                except Exception as e:
                    logger.error(f"Error in {method}: {e}")
                    continue

        # Normalize combined importance scores
        if combined_importance:
            max_score = max(combined_importance.values())
            combined_importance = {
                k: v / max_score for k, v in combined_importance.items()
            }

        # Select final features based on combined scores
        final_features = self._select_final_features(
            combined_importance, target_features
        )

        # Calculate performance metrics
        performance_metrics = await self._calculate_performance_metrics(
            X[final_features], y
        )

        # Perform correlation analysis
        correlation_matrix = await self._analyze_feature_correlations(X[final_features])

        # Create result object
        selection_result = FeatureSelectionResult(
            selection_id=f"selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            method="combined_" + "_".join(methods),
            selected_features=final_features,
            removed_features=[f for f in original_features if f not in final_features],
            importance_scores=combined_importance,
            performance_metrics=performance_metrics,
            correlation_matrix=correlation_matrix,
            timestamp=datetime.now(timezone.utc),
            parameters={
                "methods": methods,
                "target_features": target_features,
                "original_feature_count": len(original_features),
            },
        )

        # Store results
        await self._store_selection_result(selection_result)

        logger.info(
            f"Feature selection completed: {len(final_features)} features selected"
        )
        return selection_result

    # =====================================================
    # INDIVIDUAL SELECTION METHODS
    # =====================================================

    async def _variance_threshold_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Remove features with low variance"""
        threshold = self.default_params["variance_threshold"]["threshold"]

        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)

        selected_features = list(X.columns[selector.get_support()])

        # Calculate variance scores
        variances = X.var()
        importance_scores = {
            feature: variances[feature] for feature in selected_features
        }

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "method_params": {"threshold": threshold},
        }

    async def _correlation_filter_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Remove highly correlated features"""
        correlation_matrix = X.corr().abs()
        threshold = self.default_params["correlation_threshold"]

        # Find correlated feature pairs
        upper_tri = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Remove features with high correlation
        to_drop = [
            column for column in upper_tri.columns if any(upper_tri[column] > threshold)
        ]

        selected_features = [col for col in X.columns if col not in to_drop]

        # Calculate correlation-based importance (inverse of max correlation)
        importance_scores = {}
        for feature in selected_features:
            max_corr = correlation_matrix[feature].drop(feature).abs().max()
            importance_scores[feature] = 1 - max_corr

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "removed_features": to_drop,
            "method_params": {"correlation_threshold": threshold},
        }

    async def _univariate_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Select features using univariate statistical tests"""
        k = min(target_features, X.shape[1])

        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_features = list(X.columns[selector.get_support()])
        scores = selector.scores_

        importance_scores = {
            feature: scores[i]
            for i, feature in enumerate(X.columns)
            if feature in selected_features
        }

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "method_params": {"k": k, "score_func": "f_classif"},
        }

    async def _recursive_elimination_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Recursive feature elimination with cross-validation"""
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=self.default_params["random_state"]
        )

        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=self.default_params["cv_folds"],
            scoring="accuracy",
            n_jobs=-1,
        )

        selector.fit(X, y)

        selected_features = list(X.columns[selector.get_support()])

        # Get feature rankings (lower is better)
        rankings = selector.ranking_
        importance_scores = {
            feature: 1.0 / rankings[i]
            for i, feature in enumerate(X.columns)
            if feature in selected_features
        }

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "optimal_features": selector.n_features_,
            "method_params": {"cv_folds": self.default_params["cv_folds"]},
        }

    async def _lasso_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """LASSO regularization for feature selection"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LassoCV(
            cv=self.default_params["cv_folds"],
            random_state=self.default_params["random_state"],
        )
        lasso.fit(X_scaled, y)

        # Select features with non-zero coefficients
        selected_mask = np.abs(lasso.coef_) > 0
        selected_features = list(X.columns[selected_mask])

        importance_scores = {
            feature: abs(lasso.coef_[i])
            for i, feature in enumerate(X.columns)
            if feature in selected_features
        }

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "alpha": lasso.alpha_,
            "method_params": {"cv_folds": self.default_params["cv_folds"]},
        }

    async def _random_forest_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Random Forest feature importance"""
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.default_params["random_state"],
            n_jobs=-1,
        )

        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_pairs = list(zip(X.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select top features
        n_select = min(target_features, len(feature_importance_pairs))
        selected_features = [pair[0] for pair in feature_importance_pairs[:n_select]]

        importance_scores = {
            feature: importance
            for feature, importance in feature_importance_pairs[:n_select]
        }

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "method_params": {"n_estimators": 200},
        }

    async def _mutual_information_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """Mutual information feature selection"""
        k = min(target_features, X.shape[1])

        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)

        selected_features = list(X.columns[selector.get_support()])
        scores = selector.scores_

        importance_scores = {
            feature: scores[i]
            for i, feature in enumerate(X.columns)
            if feature in selected_features
        }

        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1
        importance_scores = {k: v / max_score for k, v in importance_scores.items()}

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "method_params": {"k": k},
        }

    async def _pca_selection(
        self, X: pd.DataFrame, y: pd.Series, target_features: int
    ) -> Dict[str, Any]:
        """PCA-based feature selection"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of components
        pca = PCA()
        pca.fit(X_scaled)

        # Select components that explain 95% of variance
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_ratio >= 0.95) + 1
        n_components = min(n_components, target_features)

        # Get feature contributions to top components
        components = pca.components_[:n_components]
        feature_contributions = np.mean(np.abs(components), axis=0)

        # Select features with highest contributions
        feature_importance_pairs = list(zip(X.columns, feature_contributions))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        selected_features = [
            pair[0] for pair in feature_importance_pairs[:target_features]
        ]

        importance_scores = {
            feature: contribution
            for feature, contribution in feature_importance_pairs[:target_features]
        }

        return {
            "selected_features": selected_features,
            "importance_scores": importance_scores,
            "n_components": n_components,
            "explained_variance_ratio": cumsum_ratio[n_components - 1],
            "method_params": {"variance_threshold": 0.95},
        }

    # =====================================================
    # FEATURE ANALYSIS AND UTILITIES
    # =====================================================

    def _select_final_features(
        self, importance_scores: Dict[str, float], target_features: int
    ) -> List[str]:
        """Select final features based on combined importance scores"""
        sorted_features = sorted(
            importance_scores.items(), key=lambda x: x[1], reverse=True
        )

        return [feature for feature, _ in sorted_features[:target_features]]

    async def _calculate_performance_metrics(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for selected features"""
        if X.empty:
            return {}

        # Use simple RandomForest for performance evaluation
        rf = RandomForestClassifier(
            n_estimators=100, random_state=self.default_params["random_state"]
        )

        # Cross-validation scores
        cv_scores = cross_val_score(
            rf, X, y, cv=self.default_params["cv_folds"], scoring="accuracy"
        )

        return {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "feature_count": len(X.columns),
            "data_points": len(X),
        }

    async def _analyze_feature_correlations(
        self, X: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between selected features"""
        if X.empty:
            return {}

        correlation_matrix = X.corr()

        # Convert to nested dictionary format
        corr_dict = {}
        for i, feature1 in enumerate(correlation_matrix.columns):
            corr_dict[feature1] = {}
            for j, feature2 in enumerate(correlation_matrix.columns):
                corr_dict[feature1][feature2] = float(correlation_matrix.iloc[i, j])

        return corr_dict

    async def track_feature_importance(
        self,
        feature_name: str,
        importance_score: float,
        method: str,
        model_type: str,
        dataset_id: str = None,
    ):
        """Track feature importance over time"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO feature_importance_tracking 
                (feature_name, importance_score, method, model_type, dataset_id)
                VALUES ($1, $2, $3, $4, $5)
            """,
                feature_name,
                importance_score,
                method,
                model_type,
                dataset_id,
            )

    async def _store_selection_result(self, result: FeatureSelectionResult):
        """Store feature selection result in database"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO feature_selection_results 
                (selection_id, method, selected_features, removed_features, 
                 importance_scores, performance_metrics, correlation_matrix, parameters)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                result.selection_id,
                result.method,
                result.selected_features,
                result.removed_features,
                json.dumps(result.importance_scores),
                json.dumps(result.performance_metrics),
                (
                    json.dumps(result.correlation_matrix)
                    if result.correlation_matrix
                    else None
                ),
                json.dumps(result.parameters),
            )

    async def get_feature_importance_history(self, feature_name: str) -> List[Dict]:
        """Get feature importance history for a specific feature"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT method, model_type, importance_score, timestamp, dataset_id
                FROM feature_importance_tracking 
                WHERE feature_name = $1 
                ORDER BY timestamp DESC
                LIMIT 100
            """,
                feature_name,
            )

            return [dict(row) for row in rows]

    async def get_top_features_by_method(
        self, method: str, limit: int = 20
    ) -> List[Dict]:
        """Get top features by selection method"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT feature_name, AVG(importance_score) as avg_importance,
                       COUNT(*) as occurrence_count,
                       MAX(timestamp) as last_seen
                FROM feature_importance_tracking 
                WHERE method = $1 
                GROUP BY feature_name
                ORDER BY avg_importance DESC
                LIMIT $2
            """,
                method,
                limit,
            )

            return [dict(row) for row in rows]

    async def close(self):
        """Close database connection"""
        if self.pool:
            await self.pool.close()
            logger.info("Feature Selector database connection closed")


# =====================================================
# USAGE EXAMPLE
# =====================================================


async def main():
    """Example usage"""
    # Initialize feature selector
    selector = AutomatedFeatureSelector()
    await selector.initialize()

    try:
        # Example with synthetic data
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=100,
            n_informative=20,
            n_redundant=10,
            random_state=42,
        )

        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        # Perform feature selection
        result = await selector.select_features(
            X_df,
            y_series,
            methods=[
                "correlation_filter",
                "random_forest_importance",
                "mutual_information",
            ],
            target_features=30,
        )

        print(f"Selected {len(result.selected_features)} features")
        print(f"Performance metrics: {result.performance_metrics}")

    finally:
        await selector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
