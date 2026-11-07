import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from .config import Config, setup_logger
from .data import load_transaction_data
from .features import extract_all_features
from .models import AnomalyDetector
from .sequence import SequenceAnomalyDetector, cluster_embeddings, compute_ensemble_scores
from .benchmark import run_benchmark
from .reporting import generate_visualizations, generate_pdf_report

def main():
    parser = argparse.ArgumentParser(description="DeCrypt: Bitcoin & Transaction Fraud Detection Engine")
    parser.add_argument('--data', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--outdir', type=str, default='.', help="Directory to save reports, models, and plots")
    parser.add_argument('--contamination', type=float, default=0.1, help="Expected proportion of anomalies (Isolation Forest)")
    parser.add_argument('--clusters', type=int, default=10, help="Max number of clusters for KMeans")
    
    # Analysis & Feature Flags
    parser.add_argument('--legacy-features', action='store_true', help="Use legacy flawed feature extraction baseline")
    parser.add_argument('--include-network-features', action='store_true', help="Include heuristic network features in scoring (default: False; negative result)")
    parser.add_argument('--link-window-hours', type=float, default=24.0, help="Heuristic graph link window in hours")
    parser.add_argument('--link-tolerance', type=float, default=0.02, help="Heuristic graph amount match tolerance")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs for sequence transformer")
    parser.add_argument('--no-transformer', action='store_true', help="Disable sequence transformer model")
    parser.add_argument('--benchmark', action='store_true', help="Run synthetic anomaly benchmark across seeds")
    parser.add_argument('--profile', type=str, choices=['batch', 'streaming'], default='batch', help="Analysis profile: 'batch' (IF-led mean fusion 0.8/0.2) or 'streaming' (3-signal fusion 0.2/0.1/0.7 window-5)")
    
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
    
    if args.profile == 'streaming':
        p_if_w = 0.2
        p_tf_w = 0.8
        p_seq_agg = 'window_max'
    else:
        p_if_w = 0.8
        p_tf_w = 0.2
        p_seq_agg = 'mean'

    config = Config(
        contamination=args.contamination, 
        n_clusters=args.clusters,
        random_state=args.seed,
        seed=args.seed,
        legacy_features=args.legacy_features,
        include_network_features=args.include_network_features,
        link_window_hours=args.link_window_hours,
        link_tolerance=args.link_tolerance,
        epochs=args.epochs,
        no_transformer=args.no_transformer,
        benchmark=args.benchmark,
        profile=args.profile,
        if_weight=p_if_w,
        transformer_weight=p_tf_w,
        sequence_aggregation=p_seq_agg,
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
        
        # Optional Benchmark
        benchmark_data = None
        if config.benchmark:
            logger.info("Executing Benchmark Suite...")
            benchmark_data = run_benchmark(df, outdir, config)
            
        # 2. Extract Features (Behavioral + Heuristic Graph or Legacy)
        features, G = extract_all_features(df, config)
        
        # 3. Baseline / Tabular Machine Learning (Isolation Forest + KMeans)
        detector = AnomalyDetector(config)
        suspicious_addresses, risk_metrics = detector.detect_anomalies(features)
        cluster_results = detector.cluster_addresses(features, max_clusters=config.n_clusters)
        if_scores = pd.Series(risk_metrics['risk_score'], index=features.index)
        
        # 4. Sequence Transformer Modeling & Embedding Clustering
        explanations = {}
        embedding_clusters = None
        
        if not config.no_transformer and not config.legacy_features:
            logger.info("Running Sequence Transformer pipeline...")
            seq_detector = SequenceAnomalyDetector(config)
            seq_detector.fit(df, outdir=outdir)
            tf_scores_df, explanations, embeddings = seq_detector.score_addresses(df)
            
            # HDBSCAN and KMeans on Transformer sequence embeddings (only for addresses with >= 3 events)
            valid_mask = ~tf_scores_df['insufficient_history'].values
            valid_addrs = [str(a) for a in tf_scores_df.index[valid_mask]]
            valid_embeddings = embeddings[valid_mask]
            
            embedding_clusters = cluster_embeddings(valid_embeddings, valid_addrs, config)
            
            # Combine into Ensemble Risk Scores
            risk_scores_df = compute_ensemble_scores(if_scores, tf_scores_df, config)
        else:
            if config.legacy_features:
                logger.info("Skipping Sequence Transformer due to --legacy-features flag.")
            else:
                logger.info("Skipping Sequence Transformer due to --no-transformer flag.")
                
            risk_scores_df = pd.DataFrame(index=features.index)
            risk_scores_df['if_risk_score'] = if_scores
            risk_scores_df['transformer_surprise'] = 0.0
            risk_scores_df['transformer_risk_score'] = 0.0
            risk_scores_df['ensemble_risk_score'] = if_scores
            risk_scores_df['insufficient_history'] = True
            risk_scores_df['seq_length'] = 0
            
        # Add Cluster Identifiers
        risk_scores_df['cluster_id'] = [cluster_results['address_clusters'].get(str(a), 0) for a in risk_scores_df.index]
        if embedding_clusters:
            risk_scores_df['hdbscan_cluster_id'] = [embedding_clusters['hdbscan_clusters'].get(str(a), -1) for a in risk_scores_df.index]
            
        # 5. Export Risk Scores CSV
        csv_path = outdir / "risk_scores.csv"
        risk_scores_df.to_csv(csv_path, index_label='address')
        logger.info(f"Risk scores saved to {csv_path}")
        
        # 6. Generate Visualizations & PDF Report
        plot_path = generate_visualizations(
            df=df,
            risk_scores_df=risk_scores_df,
            cluster_results=cluster_results,
            embedding_clusters=embedding_clusters,
            G=G,
            output_dir=outdir
        )
        
        pdf_path = generate_pdf_report(
            risk_scores_df=risk_scores_df,
            explanations=explanations,
            cluster_results=cluster_results,
            embedding_clusters=embedding_clusters,
            plot_path=plot_path,
            output_dir=outdir,
            benchmark_data=benchmark_data,
            graph_edges=G.number_of_edges()
        )
        
        logger.info("=== Analysis Complete ===")
        
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
