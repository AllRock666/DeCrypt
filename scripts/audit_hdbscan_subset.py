import json
from pathlib import Path
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from decrypt.config import default_config
from decrypt.sequence import SequenceAnomalyDetector

df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)

cfg = default_config
seq_det = SequenceAnomalyDetector(cfg)

# Load saved bins and model weights
with open('output/transformer_bins.json') as f:
    bins_data = json.load(f)
seq_det.amt_edges = bins_data['amt_edges']
seq_det.dt_edges = bins_data['dt_edges']
seq_det.model.load_state_dict(torch.load('output/transformer_model.pt', weights_only=True))

tf_scores_df, explanations, embeddings = seq_det.score_addresses(df)

# Filter only addresses with >= 3 events
valid_mask = ~tf_scores_df['insufficient_history'].values
valid_addrs = tf_scores_df.index[valid_mask].tolist()
valid_embeddings = embeddings[valid_mask]

print(f"Total valid addresses (>=3 events): {len(valid_addrs)}")
print(f"Valid embeddings shape: {valid_embeddings.shape}")

# PCA to 16 dims
pca = PCA(n_components=16, random_state=42)
X_pca = pca.fit_transform(valid_embeddings)

# HDBSCAN on valid subset
for min_cluster_size in [5, 7, 10, int(len(valid_addrs)*0.02)]:
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2)
    labels = hdb.fit_predict(X_pca)
    non_noise = labels != -1
    n_clusters = len(set(labels[non_noise]))
    unique, counts = np.unique(labels, return_counts=True)
    sizes = dict(zip(unique, counts))
    sil = silhouette_score(X_pca[non_noise], labels[non_noise]) if n_clusters > 1 else 0.0
    print(f"min_cluster_size={min_cluster_size}: n_clusters={n_clusters}, sil={sil:.4f}, noise={np.sum(~non_noise)}, sizes={sizes}")

# KMeans on valid subset
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km_labels = km.fit_predict(X_pca)
    km_sil = silhouette_score(X_pca, km_labels)
    unique, counts = np.unique(km_labels, return_counts=True)
    sizes = dict(zip(unique, counts))
    print(f"KMeans k={k}: sil={km_sil:.4f}, sizes={sizes}")
