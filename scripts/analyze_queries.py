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

from decrypt.config import Config, default_config
from decrypt.features import extract_all_features, prepare_numeric_features
from decrypt.sequence import SequenceAnomalyDetector

# 1. Load data
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)

# Sort
df = df.sort_values(['address_id', 'timestamp'])

# =====================================================================
# PART 1: Event-Count Distribution (Real vs Pass-Through)
# =====================================================================
real_counts = df.groupby('address_id').size()
print("=== Real Address Transaction Count Distribution (All 646 addresses) ===")
print(real_counts.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]))

real_counts_valid = real_counts[real_counts >= 3]
print("\n=== Real Address Transaction Count Distribution (>=3 events, 372 addresses) ===")
print(real_counts_valid.describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95]))

# Pass-through in generator v2: n_cycles in [6..11], 2 events per cycle -> 12 to 22 events
# Let's verify:
pt_counts = np.array([2 * c for c in range(6, 12)])
print(f"\nPass-Through Injections in v2: length = 2 * n_cycles (where n_cycles in [6..11]) -> {pt_counts.min()} to {pt_counts.max()} events (mean = {pt_counts.mean():.1f})")

# =====================================================================
# PART 3: The 29 Real Addresses with Deposit -> Withdrawal within 15 mins
# =====================================================================
risk_df = pd.read_csv('output/risk_scores.csv')

# Identify the 29 addresses
addrs_dep_then_wd_15m = []
for addr, group in df.groupby('address_id'):
    times = group['timestamp'].values
    types = group['type'].values
    n = len(times)
    if n > 1:
        for i in range(n - 1):
            if types[i] == 1:
                dt = (times[i+1] - times[i]) / np.timedelta64(1, 's')
                if dt <= 900.0 and types[i+1] == -1:
                    addrs_dep_then_wd_15m.append(addr)
                    break

print(f"\nIdentified {len(addrs_dep_then_wd_15m)} real addresses with dep->wd <= 15m")
sub_risk = risk_df[risk_df['address'].isin(addrs_dep_then_wd_15m)].copy()

# Compute transformer surprise percentile among all addresses with valid history
valid_surprises = risk_df[~risk_df['insufficient_history']]['transformer_surprise'].values
sub_risk['tf_surprise_percentile'] = np.nan
for idx, row in sub_risk.iterrows():
    if not row['insufficient_history']:
        pct = (valid_surprises < row['transformer_surprise']).mean() * 100.0
        sub_risk.loc[idx, 'tf_surprise_percentile'] = pct

print("\n=== Risk Scores for 29 Real Addresses ===")
print(sub_risk[['address', 'seq_length', 'insufficient_history', 'transformer_surprise', 'tf_surprise_percentile', 'if_risk_score', 'ensemble_risk_score']])

print("\n=== Summary of 29 Real Addresses ===")
print(f"Total: {len(sub_risk)}")
print(f"Insufficient history (<3 txs): {sub_risk['insufficient_history'].sum()} ({sub_risk['insufficient_history'].mean()*100:.1f}%)")
print(f"Sufficient history (>=3 txs): {(~sub_risk['insufficient_history']).sum()}")
valid_sub = sub_risk[~sub_risk['insufficient_history']]
print("\nTransformer Surprise Percentile (for addresses with >=3 events):")
print(valid_sub['tf_surprise_percentile'].describe())
print("\nEnsemble Risk Score (all 29 addresses):")
print(sub_risk['ensemble_risk_score'].describe())
print("\nIF Risk Score (all 29 addresses):")
print(sub_risk['if_risk_score'].describe())
