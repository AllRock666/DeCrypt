import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx

from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

class AnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=config.contamination,
            random_state=config.random_state
        )
        self.kmeans: KMeans = None
        
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[list, dict]:
        """
        Identify anomalous addresses using Isolation Forest.
        """
        logger.info("Detecting anomalies using Isolation Forest...")
        
        # Ensure numeric
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        if numeric_features.empty:
            raise ValueError("No numeric features available for model training.")
            
        # Scale
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        # Predict
        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        suspicious_indices = np.where(anomaly_scores == -1)[0]
        suspicious_addresses = features.index[suspicious_indices].tolist()
        
        # Risk scores (0 to 100)
        raw_scores = self.isolation_forest.score_samples(X_scaled)
        # Normalize assuming normal samples have scores closer to 0, anomalies closer to -1
        risk_scores = ((1 - raw_scores) * 50).clip(0, 100)
        
        risk_metrics = {
            'address': features.index.tolist(),
            'risk_score': risk_scores.tolist()
        }
        
        logger.info(f"Detected {len(suspicious_addresses)} suspicious addresses.")
        return suspicious_addresses, risk_metrics

    def cluster_addresses(self, features: pd.DataFrame, max_clusters: int = 20) -> Dict[str, Any]:
        """
        Cluster addresses using KMeans with elbow method to find optimal k.
        """
        logger.info("Clustering addresses...")
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        n_samples = len(X_scaled)
        max_k = min(max_clusters, n_samples)
        
        if max_k <= 1:
            logger.warning("Not enough samples for clustering. Skipping.")
            return {'optimal_clusters': 1, 'address_clusters': {addr: 0 for addr in features.index}, 'cluster_characteristics': {}}
            
        inertias = []
        K_range = range(1, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init='auto')
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            
        # Basic elbow point estimation (using diffs)
        diffs = np.diff(inertias)
        if len(diffs) > 0 and np.mean(diffs) < 0:
            elbow_point = np.where(diffs > np.mean(diffs))[0]
            optimal_k = elbow_point[-1] + 2 if len(elbow_point) > 0 else self.config.n_clusters
        else:
            optimal_k = min(self.config.n_clusters, max_k)
            
        optimal_k = min(max_k, optimal_k)
        
        logger.info(f"Optimal clusters estimated at: {optimal_k}")
        
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=self.config.random_state, n_init='auto')
        clusters = self.kmeans.fit_predict(X_scaled)
        
        cluster_characteristics = self._analyze_clusters(features, clusters, optimal_k)
        
        cluster_results = {
            'address_clusters': dict(zip(features.index, clusters)),
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_centers': self.kmeans.cluster_centers_,
            'optimal_clusters': optimal_k,
            'cluster_characteristics': cluster_characteristics,
        }
        
        return cluster_results
        
    def _analyze_clusters(self, features: pd.DataFrame, clusters: np.ndarray, n_clusters: int) -> dict:
        characteristics = {}
        for cluster_id in range(n_clusters):
            mask = clusters == cluster_id
            cluster_subset = features.loc[mask]
            
            if cluster_subset.empty:
                continue
                
            chars = {
                'size': int(np.sum(mask)),
                'avg_transaction_volume': float(cluster_subset['total_volume'].mean()),
                'avg_transaction_frequency': float(cluster_subset['total_transactions'].mean()),
                'avg_in_degree': float(cluster_subset.get('in_degree_centrality', pd.Series(0)).mean()),
                'avg_out_degree': float(cluster_subset.get('out_degree_centrality', pd.Series(0)).mean()),
                'behavioral_pattern': self._identify_pattern(cluster_subset)
            }
            characteristics[cluster_id] = chars
        return characteristics
        
    def _identify_pattern(self, cluster_features: pd.DataFrame) -> str:
        patterns = []
        vol_median = cluster_features['total_volume'].median()
        
        if cluster_features['total_volume'].mean() > vol_median * 3:
            patterns.append("High-value")
        elif cluster_features['total_volume'].mean() < vol_median * 0.3:
            patterns.append("Low-value")
            
        if cluster_features.get('pagerank', pd.Series(0)).mean() > 0.05:
            patterns.append("Central node")
            
        if cluster_features['deposit_withdraw_ratio'].mean() > 0.8:
            patterns.append("Mainly deposits")
        elif cluster_features['deposit_withdraw_ratio'].mean() < -0.8:
            patterns.append("Mainly withdrawals")
            
        return ", ".join(patterns) if patterns else "Mixed activity"
