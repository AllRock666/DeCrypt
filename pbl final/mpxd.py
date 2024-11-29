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
from datetime import datetime

warnings.filterwarnings('ignore')

class BitcoinFraudDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=10, random_state=42)

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, header=None,
                         names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
        df['type'] = df['type'].map({'deposit': 1, 'withdraw': -1})
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        address_features = pd.DataFrame()
        grouped = df.groupby('address_id')

        address_features['total_transactions'] = grouped.size()
        address_features['total_volume'] = grouped['amount'].sum().abs()
        address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
        address_features['transaction_variance'] = grouped['amount'].var()
        address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
        address_features['unique_interactions'] = grouped['hash'].nunique()
        address_features['max_transaction'] = grouped['amount'].max().abs()
        address_features['min_transaction'] = grouped['amount'].min().abs()

        address_features['first_transaction'] = grouped['timestamp'].min()
        address_features['last_transaction'] = grouped['timestamp'].max()
        address_features['activity_period_days'] = (
            (address_features['last_transaction'] - address_features['first_transaction'])
            .dt.total_seconds() / (24 * 3600)
        )

        address_features['large_transaction_ratio'] = (
            grouped['amount'].apply(lambda x: (x.abs() > x.abs().mean() * 3).mean())
        )

        return address_features.fillna(0)

    def build_transaction_graph(self, df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()

        transactions = df.groupby('hash').agg({
            'address_id': lambda x: list(x),
            'amount': 'first',
            'type': 'first',
            'timestamp': 'first'
        })

        for _, row in transactions.iterrows():
            addresses = row['address_id']
            if len(addresses) >= 2:
                amount = abs(row['amount'])
                timestamp = row['timestamp']
                for i in range(len(addresses)):
                    for j in range(i + 1, len(addresses)):
                        addr1, addr2 = addresses[i], addresses[j]
                        if G.has_edge(addr1, addr2):
                            G[addr1][addr2]['weight'] += amount
                            G[addr1][addr2]['transactions'] += 1
                            G[addr1][addr2]['last_timestamp'] = max(
                                G[addr1][addr2]['last_timestamp'],
                                timestamp
                            )
                        else:
                            G.add_edge(addr1, addr2,
                                       weight=amount,
                                       transactions=1,
                                       first_timestamp=timestamp,
                                       last_timestamp=timestamp)

        return G

    def extract_clustering_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        behavioral_features = features.copy()
        G = self.build_transaction_graph(df)

        network_features = pd.DataFrame(index=features.index)

        network_features['degree_centrality'] = pd.Series({node: deg for node, deg in G.degree()})
        network_features['clustering_coefficient'] = pd.Series(nx.clustering(G))
        network_features['weighted_degree'] = pd.Series({
            node: sum(data['weight'] for _, _, data in G.edges(node, data=True))
            for node in G.nodes()
        })

        try:
            network_features['betweenness_centrality'] = pd.Series(
                nx.betweenness_centrality(G, weight='weight')
            )
        except:
            network_features['betweenness_centrality'] = 0

        temporal_features = pd.DataFrame(index=features.index)
        grouped_by_address = df.groupby('address_id')

        temporal_features['avg_time_between_txs'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().dt.total_seconds().mean() if len(x) > 1 else 0
        )

        temporal_features['tx_time_variance'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().dt.total_seconds().var() if len(x) > 1 else 0
        )

        temporal_features['weekend_ratio'] = grouped_by_address.apply(
            lambda x: (x['timestamp'].dt.dayofweek >= 5).mean()
        )

        combined_features = pd.concat([
            behavioral_features,
            network_features.fillna(0),
            temporal_features.fillna(0)
        ], axis=1)

        return combined_features, G

    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[List[str], Dict]:
        numeric_features = features.select_dtypes(include=['int64', 'float64'])

        X_scaled = self.scaler.fit_transform(numeric_features)

        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        suspicious_indices = np.where(anomaly_scores == -1)[0]
        suspicious_addresses = features.index[suspicious_indices].tolist()

        risk_scores = self.isolation_forest.score_samples(X_scaled)
        risk_scores = ((1 - risk_scores) * 50).clip(0, 100)

        risk_metrics = {
            'address': features.index.tolist(),
            'risk_score': risk_scores.tolist()
        }

        return suspicious_addresses, risk_metrics

    def cluster_addresses(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict:
        clustering_features, transaction_graph = self.extract_clustering_features(df, features)

        X_scaled = self.scaler.fit_transform(clustering_features)

        inertias = []
        K = range(1, min(20, len(clustering_features)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        diffs = np.diff(inertias)
        elbow_point = np.where(diffs > np.mean(diffs))[0][-1] + 2

        self.kmeans = KMeans(n_clusters=elbow_point, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)

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

        cluster_results = {
            'address_clusters': dict(zip(features.index, clusters)),
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict(),
            'cluster_centers': self.kmeans.cluster_centers_,
            'optimal_clusters': elbow_point,
            'cluster_characteristics': cluster_characteristics,
            'clustering_features': clustering_features,
            'transaction_graph': transaction_graph
        }

        return cluster_results

    def _identify_cluster_pattern(self, cluster_features: pd.DataFrame) -> str:
        patterns = []

        if cluster_features['total_volume'].mean() > cluster_features['total_volume'].median() * 3:
            patterns.append("High-value transactions")
        elif cluster_features['total_volume'].mean() < cluster_features['total_volume'].median() * 0.3:
            patterns.append("Low-value transactions")

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

        if cluster_features['weekend_ratio'].mean() > 0.6:
            patterns.append("Weekend-heavy activity")

        if len(patterns) == 0:
            patterns.append("Mixed activity")

        return ", ".join(patterns)

    def visualize_patterns(self, df: pd.DataFrame,
                           suspicious_addresses: List[str],
                           risk_metrics: Dict,
                           cluster_results: Dict):
        plt.figure(figsize=(20, 16))

        plt.subplot(2, 3, 1)
        sns.histplot(risk_metrics['risk_score'], bins=50)
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')

        plt.subplot(2, 3, 2)
        sns.histplot(df['timestamp'].dt.hour, bins=24)
        plt.title('Transaction Time Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Frequency')

        plt.subplot(2, 3, 3)
        sns.histplot(df['amount'], bins=50, kde=True)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')

        plt.subplot(2, 3, 4)
        cluster_counts = pd.Series(cluster_results['address_clusters']).value_counts()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.title('Cluster Sizes')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Addresses')

        plt.subplot(2, 3, 5)
        risk_df = pd.DataFrame(risk_metrics)
        sns.boxplot(x='cluster', y='risk_score', data=risk_df)
        plt.title('Risk Score by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Risk Score')

        plt.subplot(2, 3, 6)
        suspicious_df = df[df['address_id'].isin(suspicious_addresses)]
        sns.histplot(suspicious_df['amount'], bins=50, kde=True)
        plt.title('Suspicious Transaction Amounts')
        plt.xlabel('Transaction Amount')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def generate_report(self, suspicious_addresses: List[str], cluster_results: Dict):
        print("\nFraud Detection Report:")
        print(f"Total Suspicious Addresses: {len(suspicious_addresses)}")
        print("\nSuspicious Addresses:")
        print(suspicious_addresses)

        print("\nCluster Analysis:")
        for cluster_id, characteristics in cluster_results['cluster_characteristics'].items():
            print(f"\nCluster {cluster_id} Characteristics:")
            for feature, value in characteristics.items():
                print(f"  - {feature}: {value}")
