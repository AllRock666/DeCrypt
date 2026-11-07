import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import rankdata

from .config import Config, setup_logger, default_config
from .features import prepare_numeric_features

logger = setup_logger(__name__, default_config)

class AnomalyDetector:
    def __init__(self, config: Config):
        self.config = config
        self.if_scaler = StandardScaler()
        self.cluster_scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=config.contamination,
            random_state=config.random_state,
            max_features=getattr(config, 'if_max_features', 1.0)
        )
        self.kmeans: KMeans = None
        
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        Identify anomalous addresses using Isolation Forest with rank/percentile scoring.
        """
        logger.info("Detecting anomalies using Isolation Forest...")
        numeric_features = prepare_numeric_features(features)
        
        if numeric_features.empty:
            raise ValueError("No numeric features available for model training.")
            
        X_scaled = self.if_scaler.fit_transform(numeric_features)
        
        # Binary prediction based on contamination
        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        suspicious_indices = np.where(anomaly_scores == -1)[0]
        suspicious_addresses = [str(features.index[i]) for i in suspicious_indices]
        
        # Continuous raw decision function: lower means more anomalous
        raw_scores = self.isolation_forest.score_samples(X_scaled)
        
        # Rank-based percentile risk score: 0 (most normal) to 100 (most anomalous)
        # Invert raw_scores so highest rank = lowest raw_score = highest anomaly
        ranks = rankdata(-raw_scores, method='average')
        n_samples = len(ranks)
        if n_samples > 1:
            risk_scores = ((ranks - 1.0) / (n_samples - 1.0) * 100.0)
        else:
            risk_scores = np.array([50.0])
            
        risk_metrics = {
            'address': [str(idx) for idx in features.index],
            'raw_score': raw_scores.tolist(),
            'risk_score': risk_scores.tolist()
        }
        
        logger.info(f"Detected {len(suspicious_addresses)} suspicious addresses (contamination={self.config.contamination}).")
        return suspicious_addresses, risk_metrics

    def cluster_addresses(self, features: pd.DataFrame, max_clusters: int = 10) -> Dict[str, Any]:
        """
        Cluster addresses using KMeans with silhouette score optimization.
        """
        logger.info("Clustering addresses with silhouette-based k selection...")
        numeric_features = prepare_numeric_features(features)
        X_scaled = self.cluster_scaler.fit_transform(numeric_features)
        
        n_samples = len(X_scaled)
        max_k = min(max_clusters, n_samples - 1)
        
        if max_k < 2:
            logger.warning("Not enough samples for clustering. Defaulting to 1 cluster.")
            return {
                'optimal_clusters': 1,
                'address_clusters': {str(addr): 0 for addr in features.index},
                'cluster_sizes': {0: n_samples},
                'cluster_characteristics': {0: {'size': n_samples, 'behavioral_pattern': 'Single Cluster'}}
            }
            
        best_k = 2
        best_score = -1.0
        scores = {}
        
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init='auto')
            labels = km.fit_predict(X_scaled)
            try:
                score = silhouette_score(X_scaled, labels)
                scores[k] = score
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                logger.debug(f"Silhouette failed for k={k}: {e}")
                
        logger.info(f"Silhouette analysis evaluated k=2..{max_k}. Optimal k={best_k} (score={best_score:.4f}).")
        
        self.kmeans = KMeans(n_clusters=best_k, random_state=self.config.random_state, n_init='auto')
        clusters = self.kmeans.fit_predict(X_scaled)
        
        cluster_characteristics = self._analyze_clusters(features, clusters, best_k)
        
        return {
            'address_clusters': dict(zip([str(idx) for idx in features.index], clusters)),
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_centers': self.kmeans.cluster_centers_,
            'optimal_clusters': best_k,
            'silhouette_score': float(best_score),
            'cluster_characteristics': cluster_characteristics
        }
        
    def _analyze_clusters(self, features: pd.DataFrame, clusters: np.ndarray, n_clusters: int) -> dict:
        characteristics = {}
        features_reset = features.copy()
        features_reset['cluster_label'] = clusters
        
        vol_median = features_reset['total_volume'].median() if 'total_volume' in features_reset else 1.0
        
        for cluster_id in range(n_clusters):
            cluster_subset = features_reset[features_reset['cluster_label'] == cluster_id]
            if cluster_subset.empty:
                continue
                
            chars = {
                'size': int(len(cluster_subset)),
                'avg_transaction_volume': float(cluster_subset['total_volume'].mean()) if 'total_volume' in cluster_subset else 0.0,
                'avg_transaction_frequency': float(cluster_subset['total_transactions'].mean()) if 'total_transactions' in cluster_subset else 0.0,
                'avg_in_degree': float(cluster_subset['in_degree_centrality'].mean()) if 'in_degree_centrality' in cluster_subset else 0.0,
                'avg_out_degree': float(cluster_subset['out_degree_centrality'].mean()) if 'out_degree_centrality' in cluster_subset else 0.0,
                'behavioral_pattern': self._identify_pattern(cluster_subset, vol_median)
            }
            characteristics[cluster_id] = chars
        return characteristics
        
    def _identify_pattern(self, cluster_features: pd.DataFrame, global_vol_median: float) -> str:
        patterns = []
        if 'total_volume' in cluster_features:
            vol_mean = cluster_features['total_volume'].mean()
            if vol_mean > global_vol_median * 3:
                patterns.append("High-value")
            elif vol_mean < global_vol_median * 0.3:
                patterns.append("Low-value")
                
        if 'pagerank' in cluster_features and cluster_features['pagerank'].mean() > 0.05:
            patterns.append("Central node")
            
        if 'deposit_withdraw_ratio' in cluster_features:
            d_w = cluster_features['deposit_withdraw_ratio'].mean()
            if d_w > 0.6:
                patterns.append("Mainly deposits")
            elif d_w < -0.6:
                patterns.append("Mainly withdrawals")
                
        if 'pass_through_ratio' in cluster_features and cluster_features['pass_through_ratio'].mean() > 0.8:
            patterns.append("Pass-through / Mixer candidate")
            
        return ", ".join(patterns) if patterns else "Mixed activity"
