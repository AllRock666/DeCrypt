import pytest
from pathlib import Path
import pandas as pd
from decrypt.config import Config
from decrypt.data import load_transaction_data
from decrypt.features import extract_all_features
from decrypt.models import AnomalyDetector
from decrypt.sequence import SequenceAnomalyDetector, cluster_embeddings, compute_ensemble_scores
from decrypt.reporting import generate_visualizations, generate_pdf_report

def test_full_pipeline_smoke(tmp_path):
    """
    Smoke test of the full pipeline on a 50-address sample with 2 epochs.
    """
    # Load real data and sample 50 addresses
    config = Config(epochs=2, seed=42)
    df = load_transaction_data('data/bitcoin_transactions.csv', config)
    
    unique_addrs = df['address_id'].unique()[:50]
    sample_df = df[df['address_id'].isin(unique_addrs)].copy()
    
    outdir = tmp_path / "smoke_output"
    outdir.mkdir()
    
    # 1. Feature Extraction
    features, G = extract_all_features(sample_df, config)
    assert not features.empty
    assert len(features) == 50
    
    # 2. Isolation Forest & KMeans
    detector = AnomalyDetector(config)
    suspicious, risk_metrics = detector.detect_anomalies(features)
    cluster_results = detector.cluster_addresses(features, max_clusters=5)
    assert len(risk_metrics['risk_score']) == 50
    assert cluster_results['optimal_clusters'] >= 1
    
    # 3. Sequence Transformer
    seq_detector = SequenceAnomalyDetector(config)
    fit_res = seq_detector.fit(sample_df, outdir=outdir)
    assert fit_res['trained'] == True
    
    tf_scores, explanations, embeds = seq_detector.score_addresses(sample_df)
    assert len(tf_scores) == 50
    assert embeds.shape == (50, 64)
    
    # 4. Clustering Embeddings
    embed_clusters = cluster_embeddings(embeds, [str(a) for a in features.index], config)
    assert 'hdbscan_clusters' in embed_clusters
    
    # 5. Ensemble Scores
    if_scores = pd.Series(risk_metrics['risk_score'], index=features.index)
    risk_scores_df = compute_ensemble_scores(if_scores, tf_scores, config)
    assert len(risk_scores_df) == 50
    assert 'ensemble_risk_score' in risk_scores_df.columns
    
    csv_file = outdir / "risk_scores.csv"
    risk_scores_df.to_csv(csv_file)
    assert csv_file.exists()
    
    # 6. Reporting
    plot_path = generate_visualizations(
        sample_df, risk_scores_df, cluster_results, embed_clusters, G, outdir
    )
    assert plot_path.exists()
    
    pdf_path = generate_pdf_report(
        risk_scores_df, explanations, cluster_results, embed_clusters, plot_path, outdir, None, G.number_of_edges()
    )
    assert pdf_path.exists()
