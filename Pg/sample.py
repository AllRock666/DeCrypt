import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, List
from datetime import datetime

class BitcoinTransactionAnalyzer:
    def __init__(self):
        self.api_base_url = 'https://api.blockcypher.com/v1/btc/main'
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def load_data(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from either CSV file or DataFrame
        
        Args:
            source: Either path to CSV file or pandas DataFrame
            
        Returns:
            pd.DataFrame: Processed transaction data
        """
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            df = source.copy()
            
        # Clean and process the data
        df.columns = ['transaction_id', 'type', 'amount']
        df['type'] = df['type'].map({'deposit': 1, 'withdraw': 0})
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (X, y) features and labels
        """
        # Create features
        features = pd.DataFrame()
        
        # Transaction amount features
        features['amount'] = df['amount'].abs()
        features['amount_log'] = np.log1p(df['amount'].abs())
        
        # Transaction patterns
        features['transaction_hour'] = pd.to_datetime(df.index).hour
        features['is_weekend'] = pd.to_datetime(df.index).weekday >= 5
        
        # Rolling statistics
        features['rolling_mean'] = df['amount'].rolling(window=3, min_periods=1).mean()
        features['rolling_std'] = df['amount'].rolling(window=3, min_periods=1).std()
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features, df['type']
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the model and calculate accuracy metrics
        
        Args:
            X: Feature DataFrame
            y: Labels
            
        Returns:
            Dict: Training results and metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return results
    
    def visualize_results(self, df: pd.DataFrame, results: Dict, output_prefix: str = 'bitcoin_analysis'):
        """
        Create visualizations of the analysis results
        
        Args:
            df: Input DataFrame
            results: Analysis results dictionary
            output_prefix: Prefix for output files
        """
        # 1. Transaction Amount Distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='amount', hue='type', bins=50)
        plt.title('Transaction Amount Distribution')
        plt.xlabel('Amount')
        plt.ylabel('Count')
        
        # 2. Feature Importance Plot
        plt.subplot(1, 2, 2)
        importance_df = pd.DataFrame.from_dict(results['feature_importance'], 
                                             orient='index', 
                                             columns=['importance'])
        importance_df.sort_values('importance', ascending=True).plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_analysis.png')
        plt.close()
        
        # 3. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['Withdraw', 'Deposit'],
                   yticklabels=['Withdraw', 'Deposit'])
        plt.title('Confusion Matrix')
        plt.savefig(f'{output_prefix}_confusion_matrix.png')
        plt.close()
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a detailed analysis report
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            str: Formatted report
        """
        report = [
            "Bitcoin Transaction Analysis Report",
            "=" * 40,
            f"Analysis Time: {datetime.now().isoformat()}",
            "\nModel Performance:",
            f"Accuracy: {results['accuracy']:.4f}",
            "\nClassification Report:",
            results['classification_report'],
            "\nFeature Importance:",
        ]
        
        for feature, importance in results['feature_importance'].items():
            report.append(f"{feature}: {importance:.4f}")
        
        return "\n".join(report)

def main():
    # Initialize analyzer
    analyzer = BitcoinTransactionAnalyzer()
    
    # Load data from CSV
    df = analyzer.load_data("bitcoin_transactions.csv")  # Replace with your CSV file path
    
    # Prepare features
    X, y = analyzer.prepare_features(df)
    
    # Train model and get results
    results = analyzer.train_model(X, y)
    
    # Create visualizations
    analyzer.visualize_results(df, results)
    
    # Generate and save report
    report = analyzer.generate_report(results)
    with open('bitcoin_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis complete! Check the generated files for results.")
    print(f"Model Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()