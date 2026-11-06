import sys
sys.path.insert(0, '.')
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from decrypt.config import Config
from decrypt.features import extract_all_features, prepare_numeric_features
from decrypt.sequence import SequenceAnomalyDetector
from decrypt.benchmark import generate_synthetic_anomalies

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

# ---------------------------------------------------------------------
# EXPERIMENT 1: Short Pass-Through (Truncated to 3-6 events)
# ---------------------------------------------------------------------
def generate_short_passthrough(real_df, n_addrs=20, seed=42):
    rng = np.random.default_rng(seed)
    pos_amounts = real_df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    min_time = real_df['timestamp'].min()
    max_time = real_df['timestamp'].max()
    time_span = (max_time - min_time).total_seconds()
    
    records = []
    addrs = []
    for i in range(n_addrs):
        addr = f"synth_pt_short_{i:02d}"
        addrs.append(addr)
        # Length in [3, 6]
        length = rng.integers(3, 7) # 3, 4, 5, or 6
        start_sec = rng.uniform(0, max(1, time_span - 5 * 86400))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        # generate alternating deposit / withdrawal
        cur_type = 1
        cur_amt = float(rng.choice(pos_amounts))
        for ev_idx in range(length):
            if cur_type == 1:
                cur_amt = float(rng.choice(pos_amounts))
                amt = cur_amt
                records.append({
                    'address_id': addr,
                    'hash': f"h_short_{addr}_{ev_idx}",
                    'timestamp': cur_time,
                    'type': 1,
                    'amount': amt
                })
                # Next is withdrawal within 2-15m
                cur_time += pd.Timedelta(minutes=float(rng.uniform(2, 15)))
                cur_type = -1
            else:
                amt = round(cur_amt * rng.uniform(0.97, 0.99), 2)
                records.append({
                    'address_id': addr,
                    'hash': f"h_short_{addr}_{ev_idx}",
                    'timestamp': cur_time,
                    'type': -1,
                    'amount': -amt
                })
                # Next deposit in 6-24h
                cur_time += pd.Timedelta(hours=float(rng.uniform(6, 24)))
                cur_type = 1
                
    return pd.DataFrame(records), addrs

# ---------------------------------------------------------------------
# EXPERIMENT 2: Amount-Perturbation Pass-Through (0.5x, 0.8x, 1.0x, 1.5x)
# ---------------------------------------------------------------------
def generate_ratio_passthrough(real_df, ratio=1.0, n_addrs=20, seed=42):
    rng = np.random.default_rng(seed)
    pos_amounts = real_df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    min_time = real_df['timestamp'].min()
    max_time = real_df['timestamp'].max()
    time_span = (max_time - min_time).total_seconds()
    
    records = []
    addrs = []
    for i in range(n_addrs):
        addr = f"synth_pt_ratio_{ratio}_{i:02d}"
        addrs.append(addr)
        n_cycles = rng.integers(6, 12)
        start_sec = rng.uniform(0, max(1, time_span - 10 * 86400))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        for _ in range(n_cycles):
            amt = float(rng.choice(pos_amounts))
            cur_time += pd.Timedelta(hours=float(rng.uniform(6, 24)))
            records.append({
                'address_id': addr,
                'hash': f"h_r_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': 1,
                'amount': amt
            })
            cur_time += pd.Timedelta(minutes=float(rng.uniform(2, 15)))
            # Withdrawal is ratio * amt
            wd_amt = round(amt * ratio, 2)
            records.append({
                'address_id': addr,
                'hash': f"h_r_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': -1,
                'amount': -wd_amt
            })
            
    return pd.DataFrame(records), addrs

# Evaluate across 5 representative seeds to get stable means and stds
short_tf_recalls = []
short_if_recalls = []

ratio_recalls = {0.5: [], 0.8: [], 1.0: [], 1.5: []}
ratio_if_recalls = {0.5: [], 0.8: [], 1.0: [], 1.5: []}

print("Running Evaluation Across Seeds...")
for s in [42, 43, 44, 45, 46]:
    rng = np.random.default_rng(s)
    real_addresses = np.array(list(df['address_id'].unique()))
    rng.shuffle(real_addresses)
    
    split_idx = int(len(real_addresses) * 0.70)
    train_addrs = set(real_addresses[:split_idx])
    test_real_addrs = set(real_addresses[split_idx:])
    
    train_df = df[df['address_id'].isin(train_addrs)].copy()
    test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
    
    cfg = Config(epochs=15, seed=s, random_state=s, contamination=0.1, include_network_features=False)
    
    # Standard benchmark anomalies for proper top-k budget
    base_synth_df, base_synth_types = generate_synthetic_anomalies(train_df, n_per_type=20, seed=s)
    k_budget = len(base_synth_types) # 100
    
    # Fit models on train_df
    train_feats, _ = extract_all_features(train_df, cfg)
    num_train = prepare_numeric_features(train_feats)
    scaler = StandardScaler().fit(num_train)
    X_train = scaler.transform(num_train)
    if_model = IsolationForest(contamination=0.1, random_state=s).fit(X_train)
    
    tf_detector = SequenceAnomalyDetector(cfg)
    tf_detector.fit(train_df)
    
    # 1. Evaluate Short Pass-Through
    short_df, short_addrs = generate_short_passthrough(train_df, n_addrs=20, seed=s)
    # Combine test real + base synth + short synth
    eval_df_short = pd.concat([test_real_df, base_synth_df, short_df], ignore_index=True)
    all_eval_addrs = list(test_real_addrs | set(base_synth_types.keys()) | set(short_addrs))
    
    # IF scoring
    test_feats_short, _ = extract_all_features(eval_df_short, cfg)
    num_test_short = prepare_numeric_features(test_feats_short).reindex(columns=num_train.columns).fillna(0.0)
    X_test_short = scaler.transform(num_test_short.loc[all_eval_addrs])
    if_scores = -if_model.score_samples(X_test_short)
    
    # TF scoring
    tf_scores_df, _, _ = tf_detector.score_addresses(eval_df_short)
    tf_scores = np.array([float(tf_scores_df.loc[a, 'transformer_surprise']) if a in tf_scores_df.index else 0.0 for a in all_eval_addrs])
    
    # Top-K recall for short_addrs (k = 100 + 20)
    total_k = k_budget + len(short_addrs)
    top_k_if = set([all_eval_addrs[i] for i in np.argsort(-if_scores)[:total_k]])
    top_k_tf = set([all_eval_addrs[i] for i in np.argsort(-tf_scores)[:total_k]])
    
    rec_if = len(set(short_addrs) & top_k_if) / len(short_addrs)
    rec_tf = len(set(short_addrs) & top_k_tf) / len(short_addrs)
    short_if_recalls.append(rec_if)
    short_tf_recalls.append(rec_tf)
    
    # 2. Evaluate Amount Perturbation Ratios
    for r in [0.5, 0.8, 1.0, 1.5]:
        r_df, r_addrs = generate_ratio_passthrough(train_df, ratio=r, n_addrs=20, seed=s)
        eval_df_r = pd.concat([test_real_df, base_synth_df, r_df], ignore_index=True)
        all_eval_r = list(test_real_addrs | set(base_synth_types.keys()) | set(r_addrs))
        
        # TF scoring
        tf_scores_df_r, _, _ = tf_detector.score_addresses(eval_df_r)
        tf_s_r = np.array([float(tf_scores_df_r.loc[a, 'transformer_surprise']) if a in tf_scores_df_r.index else 0.0 for a in all_eval_r])
        top_k_tf_r = set([all_eval_r[i] for i in np.argsort(-tf_s_r)[:total_k]])
        rec_tf_r = len(set(r_addrs) & top_k_tf_r) / len(r_addrs)
        ratio_recalls[r].append(rec_tf_r)
        
        # IF scoring
        test_feats_r, _ = extract_all_features(eval_df_r, cfg)
        num_test_r = prepare_numeric_features(test_feats_r).reindex(columns=num_train.columns).fillna(0.0)
        X_test_r = scaler.transform(num_test_r.loc[all_eval_r])
        if_s_r = -if_model.score_samples(X_test_r)
        top_k_if_r = set([all_eval_r[i] for i in np.argsort(-if_s_r)[:total_k]])
        rec_if_r = len(set(r_addrs) & top_k_if_r) / len(r_addrs)
        ratio_if_recalls[r].append(rec_if_r)

print("\n=== EXPERIMENT 1: Short Pass-Through (Truncated to 3-6 Events) ===")
print(f"Transformer Recall: mean = {np.mean(short_tf_recalls):.3f} (std = {np.std(short_tf_recalls):.3f})")
print(f"Fixed IF Recall:    mean = {np.mean(short_if_recalls):.3f} (std = {np.std(short_if_recalls):.3f})")

print("\n=== EXPERIMENT 2: Amount-Perturbation Pass-Through (Transformer Recall) ===")
for r in [0.5, 0.8, 1.0, 1.5]:
    print(f"Ratio {r:4.1f}x: TF Recall = {np.mean(ratio_recalls[r]):.3f} (std = {np.std(ratio_recalls[r]):.3f}) | IF Recall = {np.mean(ratio_if_recalls[r]):.3f} (std = {np.std(ratio_if_recalls[r]):.3f})")
