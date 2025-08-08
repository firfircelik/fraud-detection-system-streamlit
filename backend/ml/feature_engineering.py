#!/usr/bin/env python3
"""
🔧 Advanced Feature Engineering for Fraud Detection
Gelişmiş özellik mühendisliği teknikleri
"""

import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.manifold import TSNE
from sklearn.preprocessing import (QuantileTransformer, RobustScaler,
                                   StandardScaler)

warnings.filterwarnings("ignore")


class AdvancedFeatureEngineer:
    """Gelişmiş feature engineering sınıfı"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.clusterers = {}

    def create_temporal_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """Gelişmiş temporal feature'lar oluştur"""

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Temel zaman özellikleri
        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["day_of_month"] = df[timestamp_col].dt.day
        df["month"] = df[timestamp_col].dt.month
        df["quarter"] = df[timestamp_col].dt.quarter
        df["year"] = df[timestamp_col].dt.year

        # Gelişmiş temporal features
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
        df["is_business_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)
        df["is_night_time"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Time since features
        df["days_since_epoch"] = (
            df[timestamp_col] - pd.Timestamp("1970-01-01")
        ).dt.days
        df["seconds_since_midnight"] = (
            df["hour"] * 3600
            + df[timestamp_col].dt.minute * 60
            + df[timestamp_col].dt.second
        )

        return df

    def create_velocity_features(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """Velocity ve frequency özellikleri"""

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([user_col, timestamp_col])

        # Time differences
        df["time_diff_prev"] = (
            df.groupby(user_col)[timestamp_col].diff().dt.total_seconds()
        )
        df["time_diff_next"] = (
            df.groupby(user_col)[timestamp_col].diff(-1).dt.total_seconds().abs()
        )

        # Velocity features (transactions per time window)
        for window in ["1H", "6H", "1D", "7D", "30D"]:
            df[f"tx_count_{window}"] = (
                df.groupby(user_col)[timestamp_col].rolling(window).count().values
            )
            df[f"tx_velocity_{window}"] = (
                df[f"tx_count_{window}"] / pd.Timedelta(window).total_seconds() * 3600
            )

        # Amount velocity
        if "amount" in df.columns:
            for window in ["1H", "6H", "1D"]:
                df[f"amount_sum_{window}"] = (
                    df.groupby(user_col)["amount"].rolling(window).sum().values
                )
                df[f"amount_velocity_{window}"] = (
                    df[f"amount_sum_{window}"]
                    / pd.Timedelta(window).total_seconds()
                    * 3600
                )

        # Rapid fire detection
        df["is_rapid_fire"] = (df["time_diff_prev"] < 300).astype(
            int
        )  # Less than 5 minutes
        df["rapid_fire_count"] = (
            df.groupby(user_col)["is_rapid_fire"].rolling(window=10).sum().values
        )

        return df

    def create_behavioral_features(
        self, df: pd.DataFrame, user_col: str = "user_id"
    ) -> pd.DataFrame:
        """Kullanıcı davranış özellikleri"""

        df = df.copy()

        # User-level aggregations
        user_stats = (
            df.groupby(user_col)
            .agg(
                {
                    "amount": ["count", "sum", "mean", "std", "min", "max", "median"],
                    "merchant_id": "nunique",
                    "category": "nunique" if "category" in df.columns else lambda x: 0,
                }
            )
            .reset_index()
        )

        # Flatten column names
        user_stats.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in user_stats.columns
        ]
        user_stats = user_stats.rename(columns={f"{user_col}_": user_col})

        # Merge back to original dataframe
        df = df.merge(user_stats, on=user_col, how="left", suffixes=("", "_user_agg"))

        # Deviation from user's normal behavior
        if "amount" in df.columns:
            df["amount_zscore"] = (df["amount"] - df["amount_mean"]) / (
                df["amount_std"] + 1e-8
            )
            df["amount_deviation"] = np.abs(df["amount_zscore"])
            df["is_amount_outlier"] = (df["amount_deviation"] > 2).astype(int)

        # Merchant diversity
        df["merchant_diversity"] = df["merchant_id_nunique"] / df["amount_count"]

        # Category diversity
        if "category" in df.columns:
            df["category_diversity"] = df["category_nunique"] / df["amount_count"]

        return df

    def create_network_features(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        merchant_col: str = "merchant_id",
    ) -> pd.DataFrame:
        """Network/Graph özellikleri"""

        df = df.copy()

        # Create bipartite graph (users-merchants)
        G = nx.Graph()

        # Add edges
        for _, row in df.iterrows():
            G.add_edge(f"user_{row[user_col]}", f"merchant_{row[merchant_col]}")

        # Calculate network features
        network_features = {}

        # Degree centrality
        degree_centrality = nx.degree_centrality(G)

        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(G)

        # Clustering coefficient
        clustering = nx.clustering(G)

        # PageRank
        pagerank = nx.pagerank(G)

        # Map back to dataframe
        df["user_degree_centrality"] = df[user_col].map(
            lambda x: degree_centrality.get(f"user_{x}", 0)
        )
        df["merchant_degree_centrality"] = df[merchant_col].map(
            lambda x: degree_centrality.get(f"merchant_{x}", 0)
        )

        df["user_betweenness"] = df[user_col].map(
            lambda x: betweenness_centrality.get(f"user_{x}", 0)
        )
        df["merchant_betweenness"] = df[merchant_col].map(
            lambda x: betweenness_centrality.get(f"merchant_{x}", 0)
        )

        df["user_clustering"] = df[user_col].map(
            lambda x: clustering.get(f"user_{x}", 0)
        )
        df["merchant_clustering"] = df[merchant_col].map(
            lambda x: clustering.get(f"merchant_{x}", 0)
        )

        df["user_pagerank"] = df[user_col].map(lambda x: pagerank.get(f"user_{x}", 0))
        df["merchant_pagerank"] = df[merchant_col].map(
            lambda x: pagerank.get(f"merchant_{x}", 0)
        )

        return df

    def create_statistical_features(
        self, df: pd.DataFrame, amount_col: str = "amount"
    ) -> pd.DataFrame:
        """İstatistiksel özellikler"""

        df = df.copy()

        if amount_col not in df.columns:
            return df

        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f"{amount_col}_rolling_mean_{window}"] = (
                df[amount_col].rolling(window).mean()
            )
            df[f"{amount_col}_rolling_std_{window}"] = (
                df[amount_col].rolling(window).std()
            )
            df[f"{amount_col}_rolling_min_{window}"] = (
                df[amount_col].rolling(window).min()
            )
            df[f"{amount_col}_rolling_max_{window}"] = (
                df[amount_col].rolling(window).max()
            )
            df[f"{amount_col}_rolling_median_{window}"] = (
                df[amount_col].rolling(window).median()
            )

            # Percentiles
            df[f"{amount_col}_rolling_q25_{window}"] = (
                df[amount_col].rolling(window).quantile(0.25)
            )
            df[f"{amount_col}_rolling_q75_{window}"] = (
                df[amount_col].rolling(window).quantile(0.75)
            )

            # Deviation from rolling mean
            df[f"{amount_col}_deviation_from_rolling_mean_{window}"] = (
                df[amount_col] - df[f"{amount_col}_rolling_mean_{window}"]
            ) / (df[f"{amount_col}_rolling_std_{window}"] + 1e-8)

        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f"{amount_col}_ema_{alpha}"] = df[amount_col].ewm(alpha=alpha).mean()
            df[f"{amount_col}_ema_deviation_{alpha}"] = (
                df[amount_col] - df[f"{amount_col}_ema_{alpha}"]
            ) / df[f"{amount_col}_ema_{alpha}"]

        # Trend features
        df[f"{amount_col}_trend_5"] = (
            df[amount_col]
            .rolling(5)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0)
        )
        df[f"{amount_col}_trend_10"] = (
            df[amount_col]
            .rolling(10)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0)
        )

        return df

    def create_clustering_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Clustering-based features"""

        df = df.copy()

        # Select numeric features for clustering
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)

        if len(numeric_features.columns) == 0:
            return df

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features)

        # K-Means clustering
        for n_clusters in [5, 10, 20]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df[f"kmeans_cluster_{n_clusters}"] = kmeans.fit_predict(features_scaled)

            # Distance to cluster center
            distances = kmeans.transform(features_scaled)
            df[f"kmeans_distance_{n_clusters}"] = np.min(distances, axis=1)
            df[f"kmeans_distance_ratio_{n_clusters}"] = df[
                f"kmeans_distance_{n_clusters}"
            ] / np.mean(distances, axis=1)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        df["dbscan_cluster"] = dbscan.fit_predict(features_scaled)
        df["is_dbscan_outlier"] = (df["dbscan_cluster"] == -1).astype(int)

        return df

    def create_dimensionality_reduction_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> pd.DataFrame:
        """Dimensionality reduction features"""

        df = df.copy()

        # Select numeric features
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)

        if len(numeric_features.columns) < 2:
            return df

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features)

        # PCA
        pca = PCA(n_components=min(10, len(numeric_features.columns)))
        pca_features = pca.fit_transform(features_scaled)

        for i in range(pca_features.shape[1]):
            df[f"pca_component_{i}"] = pca_features[:, i]

        df["pca_explained_variance_ratio"] = np.sum(pca.explained_variance_ratio_)

        # ICA
        ica = FastICA(
            n_components=min(5, len(numeric_features.columns)), random_state=42
        )
        ica_features = ica.fit_transform(features_scaled)

        for i in range(ica_features.shape[1]):
            df[f"ica_component_{i}"] = ica_features[:, i]

        # SVD
        svd = TruncatedSVD(
            n_components=min(5, len(numeric_features.columns)), random_state=42
        )
        svd_features = svd.fit_transform(features_scaled)

        for i in range(svd_features.shape[1]):
            df[f"svd_component_{i}"] = svd_features[:, i]

        return df

    def create_interaction_features(
        self, df: pd.DataFrame, feature_cols: List[str], max_interactions: int = 20
    ) -> pd.DataFrame:
        """Feature interactions"""

        df = df.copy()

        # Select numeric features
        numeric_features = [
            col
            for col in feature_cols
            if col in df.columns and df[col].dtype in ["int64", "float64"]
        ]

        interaction_count = 0

        # Pairwise interactions
        for i, col1 in enumerate(numeric_features):
            for col2 in numeric_features[i + 1 :]:
                if interaction_count >= max_interactions:
                    break

                # Multiplication
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

                # Division (with safety)
                df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)

                # Difference
                df[f"{col1}_diff_{col2}"] = df[col1] - df[col2]

                # Ratio
                df[f"{col1}_ratio_{col2}"] = df[col1] / (df[col1] + df[col2] + 1e-8)

                interaction_count += 4

        return df

    def create_frequency_encoding(
        self, df: pd.DataFrame, categorical_cols: List[str]
    ) -> pd.DataFrame:
        """Frequency encoding for categorical variables"""

        df = df.copy()

        for col in categorical_cols:
            if col in df.columns:
                # Frequency encoding
                freq_map = df[col].value_counts().to_dict()
                df[f"{col}_frequency"] = df[col].map(freq_map)

                # Normalized frequency
                df[f"{col}_frequency_norm"] = df[f"{col}_frequency"] / len(df)

                # Rank encoding
                rank_map = (
                    df[col]
                    .value_counts()
                    .rank(method="dense", ascending=False)
                    .to_dict()
                )
                df[f"{col}_rank"] = df[col].map(rank_map)

        return df

    def create_target_encoding(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        target_col: str,
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """Target encoding with cross-validation"""

        df = df.copy()

        if target_col not in df.columns:
            return df

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for col in categorical_cols:
            if col in df.columns:
                df[f"{col}_target_encoded"] = 0.0

                for train_idx, val_idx in kf.split(df):
                    # Calculate target mean for each category in training set
                    target_mean = df.iloc[train_idx].groupby(col)[target_col].mean()

                    # Apply to validation set
                    df.loc[val_idx, f"{col}_target_encoded"] = df.loc[val_idx, col].map(
                        target_mean
                    )

                # Fill missing values with global mean
                global_mean = df[target_col].mean()
                df[f"{col}_target_encoded"].fillna(global_mean, inplace=True)

        return df

    def select_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        method: str = "mutual_info",
        k: int = 50,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Feature selection"""

        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col]

        if method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        # Create new dataframe with selected features
        df_selected = pd.DataFrame(
            X_selected, columns=selected_features, index=df.index
        )
        df_selected[target_col] = y

        return df_selected, selected_features

    def engineer_all_features(
        self,
        df: pd.DataFrame,
        user_col: str = "user_id",
        merchant_col: str = "merchant_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        target_col: str = "is_fraud",
    ) -> pd.DataFrame:
        """Tüm feature engineering işlemlerini uygula"""

        print("🔧 Starting comprehensive feature engineering...")

        # 1. Temporal features
        print("⏰ Creating temporal features...")
        df = self.create_temporal_features(df, timestamp_col)

        # 2. Velocity features
        print("🚀 Creating velocity features...")
        df = self.create_velocity_features(df, user_col, timestamp_col)

        # 3. Behavioral features
        print("👤 Creating behavioral features...")
        df = self.create_behavioral_features(df, user_col)

        # 4. Statistical features
        print("📊 Creating statistical features...")
        df = self.create_statistical_features(df, amount_col)

        # 5. Network features
        print("🕸️ Creating network features...")
        df = self.create_network_features(df, user_col, merchant_col)

        # 6. Frequency encoding
        print("🔢 Creating frequency encodings...")
        categorical_cols = (
            [merchant_col, "category"] if "category" in df.columns else [merchant_col]
        )
        df = self.create_frequency_encoding(df, categorical_cols)

        # 7. Clustering features
        print("🎯 Creating clustering features...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        df = self.create_clustering_features(
            df, numeric_cols[:20]
        )  # Limit for performance

        # 8. Dimensionality reduction
        print("📉 Creating dimensionality reduction features...")
        df = self.create_dimensionality_reduction_features(df, numeric_cols[:15])

        # 9. Interaction features
        print("🔗 Creating interaction features...")
        important_cols = [
            amount_col,
            "hour",
            "day_of_week",
            "amount_count",
            "merchant_id_nunique",
        ]
        available_cols = [col for col in important_cols if col in df.columns]
        df = self.create_interaction_features(df, available_cols, max_interactions=10)

        print(f"✅ Feature engineering complete! Created {len(df.columns)} features")

        return df


# Usage example
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame(
        {
            "user_id": [f"user_{i%100}" for i in range(n_samples)],
            "merchant_id": [f"merchant_{i%50}" for i in range(n_samples)],
            "amount": np.random.lognormal(4, 1, n_samples),
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1H"),
            "category": np.random.choice(
                ["grocery", "electronics", "gas", "restaurant"], n_samples
            ),
            "is_fraud": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        }
    )

    # Apply feature engineering
    engineer = AdvancedFeatureEngineer()
    enhanced_data = engineer.engineer_all_features(sample_data)

    print(f"Original features: {len(sample_data.columns)}")
    print(f"Enhanced features: {len(enhanced_data.columns)}")
    print(f"Feature names: {list(enhanced_data.columns[:10])}...")
