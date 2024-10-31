import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, header=None,
                         names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
        df['type'] = df['type'].map({'deposit': 1, 'withdraw': -1})
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
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
        address_features['large_transaction_ratio'] = (
            grouped['amount'].apply(lambda x: (x.abs() > x.abs().mean() * 3).mean())
        )
        return address_features.fillna(0)
    
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[List[str], Dict]:
        X_scaled = self.scaler.fit_transform(features)
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
    
    def analyze_transaction_patterns(self, df: pd.DataFrame, suspicious_addresses: List[str]) -> Dict:
        patterns = {}
        for address in suspicious_addresses:
            addr_transactions = df[df['address_id'] == address]
            patterns[address] = {
                'total_volume': addr_transactions['amount'].abs().sum(),
                'transaction_count': len(addr_transactions),
                'avg_transaction_size': addr_transactions['amount'].abs().mean(),
                'deposit_count': len(addr_transactions[addr_transactions['type'] == 1]),
                'withdraw_count': len(addr_transactions[addr_transactions['type'] == -1]),
                'large_transactions': len(addr_transactions[addr_transactions['amount'].abs() > 
                                                             addr_transactions['amount'].abs().mean() * 3]),
                'suspicious_patterns': []
            }
            if patterns[address]['deposit_count'] == 0 or patterns[address]['withdraw_count'] == 0:
                patterns[address]['suspicious_patterns'].append("One-way transactions only")
            if patterns[address]['large_transactions'] > 0:
                patterns[address]['suspicious_patterns'].append("Contains unusually large transactions")
            if patterns[address]['transaction_count'] > df.groupby('address_id').size().mean() * 2:
                patterns[address]['suspicious_patterns'].append("High transaction frequency")
        return patterns
    
    def generate_report(self, suspicious_addresses: List[str], 
                        risk_metrics: Dict, 
                        patterns: Dict) -> str:
        report = ["Bitcoin Address Fraud Analysis Report",
                  "=" * 40,
                  f"\nAnalysis Date: {pd.Timestamp.now()}",
                  f"\nTotal Addresses Analyzed: {len(risk_metrics['address'])}",
                  f"Suspicious Addresses Found: {len(suspicious_addresses)}",
                  "\nDetailed Analysis of Suspicious Addresses:",
                  "-" * 40]
        address_risks = list(zip(risk_metrics['address'], risk_metrics['risk_score']))
        address_risks.sort(key=lambda x: x[1], reverse=True)
        for address, risk_score in address_risks:
            if address in suspicious_addresses:
                addr_patterns = patterns[address]
                report.extend([
                    f"\nAddress: {address}",
                    f"Risk Score: {risk_score:.2f}/100",
                    f"Total Transaction Volume: {addr_patterns['total_volume']:.2f}",
                    f"Transaction Count: {addr_patterns['transaction_count']}",
                    f"Average Transaction Size: {addr_patterns['avg_transaction_size']:.2f}",
                    f"Deposit/Withdraw Ratio: {addr_patterns['deposit_count']}/{addr_patterns['withdraw_count']}",
                    "\nSuspicious Patterns Detected:",
                    *[f"- {pattern}" for pattern in addr_patterns['suspicious_patterns']],
                    "-" * 40
                ])
        return "\n".join(report)
    
    def visualize_patterns(self, df: pd.DataFrame, 
                           suspicious_addresses: List[str], 
                           risk_metrics: Dict):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(risk_metrics['risk_score'], bins=50)
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        suspicious_mask = df['address_id'].isin(suspicious_addresses)
        sns.boxplot(x=suspicious_mask, y=df['amount'].abs())
        plt.title('Transaction Amounts by Address Type')
        plt.xticks([0, 1], ['Normal', 'Suspicious'])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('fraud_analysis_patterns.png')
        plt.close()

def main():
    try:
        detector = BitcoinFraudDetector()
        print("Loading transaction data...")
        df = detector.load_data("bitcoin_transactions.csv")
        print("Extracting features...")
        features = detector.extract_features(df)
        print("Analyzing patterns and detecting anomalies...")
        suspicious_addresses, risk_metrics = detector.detect_anomalies(features)
        patterns = detector.analyze_transaction_patterns(df, suspicious_addresses)
        print("Generating visualizations...")
        detector.visualize_patterns(df, suspicious_addresses, risk_metrics)
        print("Generating report...")
        report = detector.generate_report(suspicious_addresses, risk_metrics, patterns)
        with open('fraud_analysis_report.txt', 'w') as f:
            f.write(report)
        print("\nAnalysis complete! Results saved to fraud_analysis_report.txt")
        print(f"Number of suspicious addresses detected: {len(suspicious_addresses)}")
        if suspicious_addresses:
            print("\nTop 5 most suspicious addresses:")
            for addr in suspicious_addresses[:5]:
                print(f"- {addr} (Risk Score: {risk_metrics['risk_score'][risk_metrics['address'].index(addr)]:.2f})")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your input data format and try again.")

if __name__ == "__main__":
    main()
