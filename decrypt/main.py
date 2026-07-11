import argparse
import sys
from pathlib import Path

from .config import Config, setup_logger
from .data import load_transaction_data
from .features import extract_all_features
from .models import AnomalyDetector
from .reporting import generate_visualizations, generate_pdf_report

def main():
    parser = argparse.ArgumentParser(description="DeCrypt: Bitcoin Fraud Detection")
    parser.add_argument('--data', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--outdir', type=str, default='.', help="Directory to save reports and plots")
    parser.add_argument('--contamination', type=float, default=0.1, help="Expected proportion of anomalies")
    parser.add_argument('--clusters', type=int, default=10, help="Max number of clusters for KMeans")
    
    args = parser.parse_args()
    
    config = Config(contamination=args.contamination, n_clusters=args.clusters)
    logger = setup_logger("decrypt", config)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Starting DeCrypt Analysis ===")
    
    try:
        # 1. Load Data
        df = load_transaction_data(args.data)
        
        # 2. Extract Features
        features, G = extract_all_features(df, config)
        
        # 3. Machine Learning
        detector = AnomalyDetector(config)
        suspicious_addresses, risk_metrics = detector.detect_anomalies(features)
        cluster_results = detector.cluster_addresses(features, max_clusters=config.n_clusters)
        
        # 4. Reporting
        plot_path = generate_visualizations(df, suspicious_addresses, risk_metrics, cluster_results, G, outdir)
        generate_pdf_report(suspicious_addresses, risk_metrics, cluster_results, plot_path, outdir)
        
        logger.info("=== Analysis Complete ===")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
