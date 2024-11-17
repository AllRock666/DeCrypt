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
        """Load and preprocess transaction data"""
        df = pd.read_csv(filepath, header=None,
                        names=['address_id', 'hash', 'type', 'amount'])
        
        # Convert types
        df['type'] = df['type'].map({'deposit': 1, 'withdraw': -1})
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant features for fraud detection"""
        # Group by address
        address_features = pd.DataFrame()
        
        # Group transactions by address
        grouped = df.groupby('address_id')
        
        # Transaction pattern features
        address_features['total_transactions'] = grouped.size()
        address_features['total_volume'] = grouped['amount'].sum().abs()
        address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
        address_features['transaction_variance'] = grouped['amount'].var()
        address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
        
        # Time-based patterns (if timestamps were available)
        # address_features['transaction_frequency'] = ...
        
        # Network features
        address_features['unique_interactions'] = grouped['hash'].nunique()
        
        # Velocity features
        address_features['max_transaction'] = grouped['amount'].max().abs()
        address_features['min_transaction'] = grouped['amount'].min().abs()
        
        # Risk indicators
        address_features['large_transaction_ratio'] = (
            grouped['amount'].apply(lambda x: (x.abs() > x.abs().mean() * 3).mean())
        )
        
        return address_features.fillna(0)
    
    def detect_anomalies(self, features: pd.DataFrame) -> Tuple[List[str], Dict]:
        """Detect suspicious addresses using anomaly detection"""
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
        
        # Get suspicious addresses
        suspicious_indices = np.where(anomaly_scores == -1)[0]
        suspicious_addresses = features.index[suspicious_indices].tolist()
        
        # Calculate risk scores (0 to 100)
        risk_scores = self.isolation_forest.score_samples(X_scaled)
        risk_scores = ((1 - risk_scores) * 50).clip(0, 100)  # Convert to 0-100 scale
        
        risk_metrics = {
            'address': features.index.tolist(),
            'risk_score': risk_scores.tolist()
        }
        
        return suspicious_addresses, risk_metrics
    
    def analyze_transaction_patterns(self, df: pd.DataFrame, suspicious_addresses: List[str]) -> Dict:
        """Analyze transaction patterns of suspicious addresses"""
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
            
            # Check for suspicious patterns
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
        """Generate detailed report of findings"""
        report = ["Bitcoin Address Fraud Analysis Report",
                 "=" * 40,
                 f"\nAnalysis Date: {pd.Timestamp.now()}",
                 f"\nTotal Addresses Analyzed: {len(risk_metrics['address'])}",
                 f"Suspicious Addresses Found: {len(suspicious_addresses)}",
                 "\nDetailed Analysis of Suspicious Addresses:",
                 "-" * 40]
        
        # Sort addresses by risk score
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
        """Create visualizations of suspicious patterns"""
        # 1. Risk Score Distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(risk_metrics['risk_score'], bins=50)
        plt.title('Risk Score Distribution')
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        
        # 2. Transaction Amounts by Address Type
        plt.subplot(1, 2, 2)
        suspicious_mask = df['address_id'].isin(suspicious_addresses)
        sns.boxplot(x=suspicious_mask, y=df['amount'].abs(), 
                   labels=['Normal', 'Suspicious'])
        plt.title('Transaction Amounts by Address Type')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('fraud_analysis_patterns.png')
        plt.close()

def main():
    try:
        # Initialize detector
        detector = BitcoinFraudDetector()
        
        # Load and process data
        print("Loading transaction data...")
        df = detector.load_data("bitcoin_transactions.csv")
        
        # Extract features
        print("Extracting features...")
        features = detector.extract_features(df)
        
        # Detect suspicious addresses
        print("Analyzing patterns and detecting anomalies...")
        suspicious_addresses, risk_metrics = detector.detect_anomalies(features)
        
        # Analyze patterns
        patterns = detector.analyze_transaction_patterns(df, suspicious_addresses)
        
        # Generate visualizations
        print("Generating visualizations...")
        detector.visualize_patterns(df, suspicious_addresses, risk_metrics)
        
        # Generate and save report
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