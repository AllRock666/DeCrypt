import argparse
import sys
from pathlib import Path

from .config import Config, setup_logger
from .data import load_transaction_data
from .features import extract_all_features
from .models import AnomalyDetector
from .reporting import generate_visualizations, generate_pdf_report

def main():
    parser = argparse.ArgumentParser(description="DeCrypt: Transaction Fraud Detection")
    parser.add_argument('--data', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--outdir', type=str, default='.', help="Directory to save reports and plots")
    parser.add_argument('--contamination', type=float, default=0.1, help="Expected proportion of anomalies")
    parser.add_argument('--clusters', type=int, default=10, help="Max number of clusters for KMeans")
    
    # Analysis & Feature Flags
    parser.add_argument('--legacy-features', action='store_true', help="Use legacy flawed feature extraction")
    parser.add_argument('--link-window-hours', type=float, default=24.0, help="Heuristic graph link window in hours")
    parser.add_argument('--link-tolerance', type=float, default=0.02, help="Heuristic graph amount match tolerance")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs for sequence transformer")
    parser.add_argument('--no-transformer', action='store_true', help="Disable sequence transformer model")
    parser.add_argument('--benchmark', action='store_true', help="Run synthetic anomaly benchmark")
    
    # Schema Mapping Arguments
    parser.add_argument('--has-header', action='store_true', help="Set this flag if the CSV has a header row")
    parser.add_argument('--col-id', type=str, default='address_id', help="Column name/index for Account or Address ID")
    parser.add_argument('--col-amount', type=str, default='amount', help="Column name/index for Transaction Amount")
    parser.add_argument('--col-time', type=str, default='timestamp', help="Column name/index for Timestamp/Date")
    parser.add_argument('--col-txid', type=str, default=None, help="Column name/index for Transaction ID (Optional)")
    parser.add_argument('--col-type', type=str, default=None, help="Column name/index for Deposit/Withdraw type (Optional)")
    parser.add_argument('--type-deposit', type=str, default='deposit', help="String representing a deposit in the type column")
    parser.add_argument('--type-withdraw', type=str, default='withdraw', help="String representing a withdrawal in the type column")
    
    args = parser.parse_args()
    
    # Configure the system based on CLI args
    config = Config(
        contamination=args.contamination, 
        n_clusters=args.clusters,
        random_state=args.seed,
        seed=args.seed,
        legacy_features=args.legacy_features,
        link_window_hours=args.link_window_hours,
        link_tolerance=args.link_tolerance,
        epochs=args.epochs,
        no_transformer=args.no_transformer,
        benchmark=args.benchmark,
        col_id=args.col_id,
        col_amount=args.col_amount,
        col_time=args.col_time,
        col_txid=args.col_txid,
        col_type=args.col_type,
        type_deposit=args.type_deposit,
        type_withdraw=args.type_withdraw,
        has_header=args.has_header
    )
    logger = setup_logger("decrypt", config)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Starting DeCrypt Analysis ===")
    
    try:
        # 1. Load Data
        df = load_transaction_data(args.data, config)
        
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
