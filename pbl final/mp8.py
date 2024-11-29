import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class BitcoinFraudDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        
    # ... [previous methods remain the same until cluster_addresses] ...

    def build_transaction_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build a graph of transactions between addresses.
        Edges are weighted by transaction frequency and volume.
        """
        G = nx.Graph()
        
        # Group transactions by hash to find direct relationships
        transactions = df.groupby('hash').agg({
            'address_id': lambda x: list(x),
            'amount': 'first',
            'type': 'first'
        })
        
        # Add edges between addresses that appear in the same transaction
        for _, row in transactions.iterrows():
            addresses = row['address_id']
            if len(addresses) >= 2:  # Only consider transactions with 2 or more addresses
                amount = abs(row['amount'])
                for i in range(len(addresses)):
                    for j in range(i+1, len(addresses)):
                        addr1, addr2 = addresses[i], addresses[j]
                        if G.has_edge(addr1, addr2):
                            G[addr1][addr2]['weight'] += amount
                            G[addr1][addr2]['transactions'] += 1
                        else:
                            G.add_edge(addr1, addr2, weight=amount, transactions=1)
        
        return G

    def extract_clustering_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for clustering that combine both behavioral and transactional patterns.
        """
        # Get behavioral features
        behavioral_features = features.copy()
        
        # Build transaction graph
        G = self.build_transaction_graph(df)
        
        # Extract network features
        network_features = pd.DataFrame(index=features.index)
        
        # Calculate network metrics for each address
        network_features['degree_centrality'] = pd.Series({node: deg for node, deg in G.degree()})
        network_features['clustering_coefficient'] = pd.Series(nx.clustering(G))
        network_features['weighted_degree'] = pd.Series({
            node: sum(data['weight'] for _, _, data in G.edges(node, data=True))
            for node in G.nodes()
        })
        
        # Calculate temporal features
        temporal_features = pd.DataFrame(index=features.index)
        grouped_by_address = df.groupby('address_id')
        
        temporal_features['avg_time_between_txs'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().mean() if len(x) > 1 else 0
        )
        
        temporal_features['tx_time_variance'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().var() if len(x) > 1 else 0
        )
        
        # Combine all features
        combined_features = pd.concat([
            behavioral_features,
            network_features.fillna(0),
            temporal_features.fillna(0)
        ], axis=1)
        
        return combined_features

    def cluster_addresses(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """
        Cluster addresses based on both behavioral patterns and transaction relationships.
        """
        # Extract comprehensive clustering features
        clustering_features = self.extract_clustering_features(df, features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(clustering_features)
        
        # Use elbow method to find optimal number of clusters
        inertias = []
        K = range(1, min(20, len(clustering_features)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point using the kneedle algorithm
        diffs = np.diff(inertias)
        elbow_point = np.where(diffs > np.mean(diffs))[0][-1] + 2
        
        # Fit final clustering model
        self.kmeans = KMeans(n_clusters=elbow_point, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics
        cluster_characteristics = {}
        for cluster_id in range(elbow_point):
            cluster_mask = clusters == cluster_id
            cluster_characteristics[cluster_id] = {
                'size': np.sum(cluster_mask),
                'avg_transaction_volume': clustering_features.loc[cluster_mask, 'total_volume'].mean(),
                'avg_transaction_frequency': clustering_features.loc[cluster_mask, 'total_transactions'].mean(),
                'avg_network_connectivity': clustering_features.loc[cluster_mask, 'degree_centrality'].mean(),
                'behavioral_pattern': self._identify_cluster_pattern(
                    clustering_features.loc[cluster_mask]
                )
            }
        
        # Create clustering results
        cluster_results = {
            'address_clusters': dict(zip(features.index, clusters)),
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_centers': self.kmeans.cluster_centers_,
            'optimal_clusters': elbow_point,
            'cluster_characteristics': cluster_characteristics,
            'clustering_features': clustering_features
        }
        
        return cluster_results

    def _identify_cluster_pattern(self, cluster_features: pd.DataFrame) -> str:
        """
        Identify the dominant pattern in a cluster based on its features.
        """
        patterns = []
        
        # Check transaction patterns
        if cluster_features['total_volume'].mean() > cluster_features['total_volume'].median() * 3:
            patterns.append("High-value transactions")
        
        if cluster_features['degree_centrality'].mean() > 0.7:
            patterns.append("Highly connected")
        elif cluster_features['degree_centrality'].mean() < 0.3:
            patterns.append("Isolated")
            
        if cluster_features['clustering_coefficient'].mean() > 0.7:
            patterns.append("Tight-knit community")
            
        if cluster_features['deposit_withdraw_ratio'].mean() > 0.8:
            patterns.append("Mainly deposits")
        elif cluster_features['deposit_withdraw_ratio'].mean() < -0.8:
            patterns.append("Mainly withdrawals")
            
        if len(patterns) == 0:
            patterns.append("Mixed activity")
            
        return ", ".join(patterns)

    # ... [rest of the class methods remain the same] ...