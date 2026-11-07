import sys
import json
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from decrypt.config import Config
from decrypt.features import extract_all_features, prepare_numeric_features
from decrypt.benchmark import generate_synthetic_anomalies, compute_bootstrap_ci

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def run_feature_ablation_study(seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51], n_per_type=20):
    cfg = Config(include_network_features=False)
    
    alternation_cols = {'direction_switch_count', 'alternation_rate', 'max_alternating_run_length'}
    timing_cols = {'cv_inter_event_time', 'modal_amount_share', 'lag1_autocorr_inter_event_time'}
    centroid_cols = {'activity_centroid'}
    all_new_cols = alternation_cols | timing_cols | centroid_cols
    
    variants = [
        ('all_28_features (default)', None, 1.0),
        ('ablate_alternation (-3 feats)', alternation_cols, 1.0),
        ('ablate_timing_regularity (-3 feats)', timing_cols, 1.0),
        ('ablate_centroid (-1 feat)', centroid_cols, 1.0),
        ('ablate_all_new (21 feats)', all_new_cols, 1.0),
        ('max_features=0.5 (all 28 feats)', None, 0.5),
        ('max_features=0.7 (all 28 feats)', None, 0.7),
        ('max_features=sqrt (5 feats)', None, 5),
    ]
    
    archetypes = ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]
    
    # results[v_name][metric] = list of values across seeds
    results = {
        v_name: {
            'auroc': [], 'ap': [], 'recall_at_k': [],
            **{f"recall_{a}": [] for a in archetypes}
        }
        for v_name, _, _ in variants
    }
    
    print(f"Starting Isolation Forest feature ablation study across {len(seeds)} seeds...")
    
    for s in seeds:
        rng = np.random.default_rng(s)
        real_addresses = np.array(list(df['address_id'].unique()))
        rng.shuffle(real_addresses)
        
        split_idx = int(len(real_addresses) * 0.70)
        train_addrs = set(real_addresses[:split_idx])
        test_real_addrs = set(real_addresses[split_idx:])
        
        train_df = df[df['address_id'].isin(train_addrs)].copy()
        test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
        
        synth_df, synth_types = generate_synthetic_anomalies(train_df, n_per_type=n_per_type, seed=s)
        synth_addrs = set(synth_types.keys())
        test_df = pd.concat([test_real_df, synth_df], ignore_index=True)
        all_test_addrs = list(test_real_addrs | synth_addrs)
        y_true = np.array([1 if a in synth_addrs else 0 for a in all_test_addrs])
        k = len(synth_addrs)
        
        # Extract full feature matrices once
        train_feats, _ = extract_all_features(train_df, cfg)
        test_feats, _ = extract_all_features(test_df, cfg)
        
        num_train_full = prepare_numeric_features(train_feats)
        num_test_full = prepare_numeric_features(test_feats).reindex(columns=num_train_full.columns).fillna(0.0)
        
        for v_name, drop_cols, mf in variants:
            # Drop specified columns if any
            if drop_cols:
                cols = [c for c in num_train_full.columns if c not in drop_cols]
                cur_num_train = num_train_full[cols]
                cur_num_test = num_test_full[cols]
            else:
                cur_num_train = num_train_full
                cur_num_test = num_test_full
                
            scaler = StandardScaler().fit(cur_num_train)
            X_train = scaler.transform(cur_num_train)
            X_test = scaler.transform(cur_num_test.loc[all_test_addrs])
            
            if_model = IsolationForest(
                contamination=0.1,
                random_state=s,
                max_features=mf
            ).fit(X_train)
            
            scores = -if_model.score_samples(X_test)
            
            auc = float(roc_auc_score(y_true, scores))
            ap = float(average_precision_score(y_true, scores))
            top_k_indices = np.argsort(-scores)[:k]
            top_k_addrs = set(all_test_addrs[i] for i in top_k_indices)
            overall_rec = len(top_k_addrs & synth_addrs) / max(k, 1)
            
            results[v_name]['auroc'].append(auc)
            results[v_name]['ap'].append(ap)
            results[v_name]['recall_at_k'].append(overall_rec)
            
            for a_type in archetypes:
                addrs_of_type = [addr for addr, t in synth_types.items() if t == a_type]
                n_type = len(addrs_of_type)
                det = len([a for a in addrs_of_type if a in top_k_addrs])
                results[v_name][f"recall_{a_type}"].append(det / max(n_type, 1))
                
        print(f"Seed {s} done.")
        
    summary = {}
    for v_name, _, _ in variants:
        summary[v_name] = {}
        for m_key in results[v_name]:
            summary[v_name][m_key] = compute_bootstrap_ci(results[v_name][m_key], n_boot=1000, ci=0.95, seed=42)
            
    return summary, results

if __name__ == '__main__':
    summary, raw = run_feature_ablation_study()
    
    print("\n" + "="*110)
    print("ISOLATION FOREST FEATURE ABLATION & STRUCTURING REGRESSION ANALYSIS (10 SEEDS)")
    print("="*110)
    
    header = f"{'Variant':<35} | {'Structuring':<14} | {'Burst':<10} | {'PassThrough':<14} | {'BotPeriodic':<14} | {'Recall@K':<12}"
    print(header)
    print("-" * 110)
    
    for v_name in summary:
        s = summary[v_name]
        struct_m = s['recall_structuring']['mean']
        struct_ci = f"[{s['recall_structuring']['ci_lower']:.2f}, {s['recall_structuring']['ci_upper']:.2f}]"
        burst_m = s['recall_burst']['mean']
        pt_m = s['recall_pass_through']['mean']
        bot_m = s['recall_bot_periodic']['mean']
        rec_m = s['recall_at_k']['mean']
        dorm_m = s['recall_dormant_spike']['mean']
        
        print(f"{v_name:<35} | {struct_m:.3f} {struct_ci:<8} | {burst_m:<10.3f} | {pt_m:<14.3f} | {bot_m:<14.3f} | {rec_m:<12.3f}")
        
    out_path = Path('output/feature_ablation_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved ablation results to {out_path}")
