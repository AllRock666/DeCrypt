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
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        address_features = pd.DataFrame()
        grouped = df.groupby('address_id')
        
        # Basic transaction features
        address_features['total_transactions'] = grouped.size()
        address_features['total_volume'] = grouped['amount'].sum().abs()
        address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
        address_features['transaction_variance'] = grouped['amount'].var()
        address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
        address_features['unique_interactions'] = grouped['hash'].nunique()
        address_features['max_transaction'] = grouped['amount'].max().abs()
        address_features['min_transaction'] = grouped['amount'].min().abs()
        
        # Calculate time-based features
        address_features['first_transaction'] = grouped['timestamp'].min()
        address_features['last_transaction'] = grouped['timestamp'].max()
        address_features['activity_period_days'] = (
            (address_features['last_transaction'] - address_features['first_transaction'])
            .dt.total_seconds() / (24 * 3600)
        )
        
        # Advanced features
        address_features['large_transaction_ratio'] = (
            grouped['amount'].apply(lambda x: (x.abs() > x.abs().mean() * 3).mean())
        )
        
        return address_features.fillna(0)
    
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
            'type': 'first',
            'timestamp': 'first'
        })
        
        # Add edges between addresses that appear in the same transaction
        for _, row in transactions.iterrows():
            addresses = row['address_id']
            if len(addresses) >= 2:
                amount = abs(row['amount'])
                timestamp = row['timestamp']
                for i in range(len(addresses)):
                    for j in range(i+1, len(addresses)):
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
        
        try:
            network_features['betweenness_centrality'] = pd.Series(
                nx.betweenness_centrality(G, weight='weight')
            )
        except:
            network_features['betweenness_centrality'] = 0
            
        # Calculate temporal features
        temporal_features = pd.DataFrame(index=features.index)
        grouped_by_address = df.groupby('address_id')
        
        temporal_features['avg_time_between_txs'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().mean().total_seconds() if len(x) > 1 else 0
        )
        
        temporal_features['tx_time_variance'] = grouped_by_address.apply(
            lambda x: x['timestamp'].diff().var().total_seconds() if len(x) > 1 else 0
        )
        
        # Add periodic activity features
        temporal_features['weekend_ratio'] = grouped_by_address.apply(
            lambda x: (x['timestamp'].dt.dayofweek >= 5).mean()
        )
        
        # Combine all features
        combined_features = pd.concat([
            behavioral_features,
            network_features.fillna(0),
            temporal_features.fillna(0)
        ], axis=1)
        
        return combined_features, G

    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=10, random_state=42)
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, header=None,
                         names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
        df['type'] = df['type'].map({'deposit': 1, 'withdraw': -1})
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        address_features = pd.DataFrame()
        grouped = df.groupby('address_id')
        
        # Basic transaction features
        address_features['total_transactions'] = grouped.size()
        address_features['total_volume'] = grouped['amount'].sum().abs()
        address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
        address_features['transaction_variance'] = grouped['amount'].var()
        address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
        address_features['unique_interactions'] = grouped['hash'].nunique()
        address_features['max_transaction'] = grouped['amount'].max().abs()
        address_features['min_transaction'] = grouped['amount'].min().abs()
        
        # Calculate time-based features
        address_features['first_transaction'] = grouped['timestamp'].min()
        address_features['last_transaction'] = grouped['timestamp'].max()
        # Convert time differences to numeric (days)
        address_features['activity_period_days'] = (
            (address_features['last_transaction'] - address_features['first_transaction'])
            .dt.total_seconds() / (24 * 3600)
        )
        
        # Convert datetime columns to numeric features
        address_features['days_since_first'] = (
            address_features['first_transaction'] - address_features['first_transaction'].min()
        ).dt.total_seconds() / (24 * 3600)
        
        address_features['days_since_last'] = (
            address_features['last_transaction'] - address_features['last_transaction'].min()
        ).dt.total_seconds() / (24 * 3600)
        
        # Drop the original datetime columns
        address_features = address_features.drop(['first_transaction', 'last_transaction'], axis=1)
        
        # Advanced features
        address_features['large_transaction_ratio'] = (
            grouped['amount'].apply(lambda x: (x.abs() > x.abs().mean() * 3).mean())
        )
        
        return address_features.fillna(0)

    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[List[str], Dict]:
        """
        Detect anomalies in the transaction patterns using Isolation Forest.
        Now handles numeric features only.
        """
        # Ensure all features are numeric
        numeric_features = features.select_dtypes(include=['int64', 'float64'])
        
        # Scale the numeric features
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        # Detect anomalies
        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        suspicious_indices = np.where(anomaly_scores == -1)[0]
        suspicious_addresses = features.index[suspicious_indices].tolist()
        
        # Calculate risk scores
        risk_scores = self.isolation_forest.score_samples(X_scaled)
        risk_scores = ((1 - risk_scores) * 50).clip(0, 100)
        
        risk_metrics = {
            'address': features.index.tolist(),
            'risk_score': risk_scores.tolist()
        }
        
        return suspicious_addresses, risk_metrics


    def cluster_addresses(self, df: pd.DataFrame, features: pd.DataFrame) -> Dict:
        """
        Cluster addresses based on both behavioral patterns and transaction relationships.
        """
        # Extract comprehensive clustering features
        clustering_features, transaction_graph = self.extract_clustering_features(df, features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(clustering_features)
        
        # Use elbow method to find optimal number of clusters
        inertias = []
        K = range(1, min(20, len(clustering_features)))
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
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
            'clustering_features': clustering_features,
            'transaction_graph': transaction_graph
        }
        
        return cluster_results

    def _identify_cluster_pattern(self, cluster_features: pd.DataFrame) -> str:
        """
        Identify the dominant pattern in a cluster based on its features.
        """
        patterns = []
        
        # Volume patterns
        if cluster_features['total_volume'].mean() > cluster_features['total_volume'].median() * 3:
            patterns.append("High-value transactions")
        elif cluster_features['total_volume'].mean() < cluster_features['total_volume'].median() * 0.3:
            patterns.append("Low-value transactions")
            
        # Network patterns
        if cluster_features['degree_centrality'].mean() > 0.7:
            patterns.append("Highly connected")
        elif cluster_features['degree_centrality'].mean() < 0.3:
            patterns.append("Isolated")
            
        if cluster_features['clustering_coefficient'].mean() > 0.7:
            patterns.append("Tight-knit community")
            
        # Transaction patterns
        if cluster_features['deposit_withdraw_ratio'].mean() > 0.8:
            patterns.append("Mainly deposits")
        elif cluster_features['deposit_withdraw_ratio'].mean() < -0.8:
            patterns.append("Mainly withdrawals")
            
        # Temporal patterns
        if cluster_features['weekend_ratio'].mean() > 0.6:
            patterns.append("Weekend-heavy activity")
            
        if len(patterns) == 0:
            patterns.append("Mixed activity")
            
        return ", ".join(patterns)

    def visualize_patterns(self, df: pd.DataFrame, 
                         suspicious_addresses: List[str], 
                         risk_metrics: Dict,
                         cluster_results: Dict):
        """
        Create comprehensive visualizations of the analysis results.
        """
        plt.figure(figsize=(20, 16))
        
        # 1. Risk Score Distribution
        plt.subplot(2, 3, 1)
        sns.histplot(risk_metrics['risk_score'], bins=50)
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        
        # 2. Transaction Amounts Box Plot
        plt.subplot(2, 3, 2)
        suspicious_mask = df['address_id'].isin(suspicious_addresses)
        sns.boxplot(x=suspicious_mask, y=df['amount'].abs())
        plt.title('Transaction Amounts by Address Type')
        plt.xticks([0, 1], ['Normal', 'Suspicious'])
        plt.yscale('log')
        
        # 3. Cluster Sizes
        plt.subplot(2, 3, 3)
        cluster_sizes = pd.Series(cluster_results['cluster_sizes'])
        sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values)
        plt.title('Addresses per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Addresses')
        
        # 4. Network Graph of Suspicious Clusters
        plt.subplot(2, 3, 4)
        G = cluster_results['transaction_graph']
        suspicious_subgraph = G.subgraph(suspicious_addresses)
        pos = nx.spring_layout(suspicious_subgraph)
        nx.draw(suspicious_subgraph, pos, 
                node_color='red', 
                node_size=100,
                edge_color='gray',
                alpha=0.6,
                with_labels=False)
        plt.title('Transaction Network of Suspicious Addresses')
        
        # 5. Temporal Pattern Analysis
        plt.subplot(2, 3, 5)
        df_suspicious = df[df['address_id'].isin(suspicious_addresses)]
        df_normal = df[~df['address_id'].isin(suspicious_addresses)]
        
        plt.hist(df_suspicious['timestamp'].dt.hour, bins=24, alpha=0.5, 
                density=True, label='Suspicious', color='red')
        plt.hist(df_normal['timestamp'].dt.hour, bins=24, alpha=0.5, 
                density=True, label='Normal', color='blue')
        plt.title('Transaction Time Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Density')
        plt.legend()
        
        # 6. Cluster Characteristics Heatmap
        plt.subplot(2, 3, 6)
        cluster_chars = pd.DataFrame(cluster_results['cluster_characteristics']).T
        cluster_chars = cluster_chars.drop('behavioral_pattern', axis=1)
        sns.heatmap(cluster_chars.iloc[:, :4], annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Cluster Characteristics')
        
        plt.tight_layout()
        plt.savefig('fraud_analysis_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, suspicious_addresses: List[str], 
                       risk_metrics: Dict, 
                       patterns: Dict,
                       cluster_results: Dict) -> str:
        """
        Generate a comprehensive analysis report.
        """
        report = [
            "Bitcoin Address Fraud Analysis Report",
            "=" * 50,
            f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nSummary Statistics:",
            f"- Total Addresses Analyzed: {len(risk_metrics['address'])}",
            f"- Suspicious Addresses Detected: {len(suspicious_addresses)}",
            f"- Number of Address Clusters: {cluster_results['optimal_clusters']}",
            "\nCluster Analysis:",
            "-" * 30
        ]
        
        # Add cluster characteristics
        for cluster_id, chars in cluster_results['cluster_characteristics'].items():
            report.extend([
                f"\nCluster {cluster_id}:",
                f"- Size: {chars['size']} addresses",
                f"- Average Transaction Volume: {chars['avg_transaction_volume']:.2f}",
                f"- Average Transaction Frequency: {chars['avg_transaction_frequency']:.2f}",
                f"- Network Connectivity: {chars['avg_network_connectivity']:.2f}",
                f"- Behavioral Pattern: {chars['behavioral_pattern']}"
            ])
        
        # Add high-risk addresses
        risk_df = pd.DataFrame(risk_metrics)
        high_risk = risk_df[risk_df['risk_score'] > 75]
        report.extend([
            "\nHigh-Risk Addresses:",
            "-" * 30
        ])
        
        for _, row in high_risk.iterrows():
            report.append(f"- Address: {row['address']}, Risk Score: {row['risk_score']:.2f}")
        
        # Add network analysis
        report.extend([
            "\nNetwork Analysis:",
            "-" * 30,
            f"- Number of Suspicious Address Clusters: {len(set(cluster_results['address_clusters'][addr] for addr in suspicious_addresses))}",
            f"- Average Network Density: {nx.density(cluster_results['transaction_graph']):.3f}"
        ])
        
        # Add recommendations
        report.extend([
            "\nRecommendations:",
            "-" * 30,
            "1. Monitor high-risk addresses for unusual transaction patterns",
            "2. Investigate clusters with high concentrations of suspicious addresses",
            "3. Review large volume transactions from addresses with high risk scores",
            "4. Implement additional verification for addresses showing unusual temporal patterns",
            "\nNote: This analysis is based on historical transaction data and behavioral patterns.",
            "Regular updates and human oversight are recommended for accurate fraud detection.",
            "\nEnd of Report"
        ])
        
        return "\n".join(report)

    def analyze_transactions(self, filepath: str) -> Dict:
        """
        Main method to analyze transactions and detect potential fraud.
        
        Args:
            filepath: Path to the transaction data file
            
        Returns:
            Dictionary containing analysis results
        """
        # Load and preprocess data
        df = self.load_data(filepath)
        
        # Extract features
        features = self.extract_features(df)
        
        # Detect anomalies
        suspicious_addresses, risk_metrics = self.detect_anomalies(features)
        
        # Perform clustering analysis
        cluster_results = self.cluster_addresses(df, features)
        
        # Generate visualizations
        self.visualize_patterns(df, suspicious_addresses, risk_metrics, cluster_results)
        
        # Generate report
        report = self.generate_report(suspicious_addresses, risk_metrics, features, cluster_results)
        
        # Compile results
        results = {
            'suspicious_addresses': suspicious_addresses,
            'risk_metrics': risk_metrics,
            'cluster_results': cluster_results,
            'analysis_report': report
        }
        
        return results

if __name__ == "__main__":
    detector = BitcoinFraudDetector()
    results = detector.analyze_transactions("bitcoin_transactions.csv")
    print(results['analysis_report'])