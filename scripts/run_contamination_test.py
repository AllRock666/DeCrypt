import sys
import json
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from decrypt.config import Config
from decrypt.features import extract_all_features, prepare_numeric_features
from decrypt.sequence import SequenceAnomalyDetector

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def run_stratified_contamination_experiment(seeds=[42, 43, 44, 45, 46], insertion_mode='end'):
    """
    Evaluates in-situ contamination of real addresses across 3 history length strata:
      - 5-9 events (short)
      - 10-24 events (medium)
      - 25+ events (long)
      - overall (all >= 5 events)
    Compares:
      - Fixed IF
      - Transformer Mean Surprise
      - Transformer Top-k Surprise
      - Transformer Sliding-Window Max Surprise
    """
    strata_names = ['5-9', '10-24', '25+', 'overall']
    models = ['fixed_if', 'tf_mean', 'tf_topk', 'tf_window_max']
    
    # Store results: results[k][stratum][model] = list of recalls across seeds
    results = {
        k: {s: {m: [] for m in models} for s in strata_names}
        for k in [1, 2, 3]
    }
    
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        rng = np.random.default_rng(s)
        real_addresses = np.array(list(df['address_id'].unique()))
        rng.shuffle(real_addresses)
        
        split_idx = int(len(real_addresses) * 0.70)
        train_addrs = set(real_addresses[:split_idx])
        test_real_addrs = set(real_addresses[split_idx:])
        
        train_df = df[df['address_id'].isin(train_addrs)].copy()
        test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
        
        cfg = Config(include_network_features=False, epochs=15, seed=s, random_state=s, contamination=0.1)
        
        # Train baseline models on clean train data
        train_feats, _ = extract_all_features(train_df, cfg)
        num_train = prepare_numeric_features(train_feats)
        scaler = StandardScaler().fit(num_train)
        if_model = IsolationForest(contamination=0.1, random_state=s).fit(scaler.transform(num_train))
        
        tf_detector = SequenceAnomalyDetector(cfg)
        tf_detector.fit(train_df)
        
        # Determine test addresses by history length strata
        test_counts = test_real_df.groupby('address_id').size()
        
        strata_addrs = {
            '5-9': list(test_counts[(test_counts >= 5) & (test_counts <= 9)].index),
            '10-24': list(test_counts[(test_counts >= 10) & (test_counts <= 24)].index),
            '25+': list(test_counts[test_counts >= 25].index),
            'overall': list(test_counts[test_counts >= 5].index)
        }
        
        # Choose 6 addrs from 5-9, 6 from 10-24, and 5 from 25+
        chosen_by_stratum = {}
        for st, count in [('5-9', 6), ('10-24', 6), ('25+', 5)]:
            avail = strata_addrs[st]
            n_pick = min(count, len(avail))
            chosen_by_stratum[st] = list(rng.choice(avail, size=n_pick, replace=False))
            
        all_chosen = []
        for c in chosen_by_stratum.values():
            all_chosen.extend(c)
        chosen_by_stratum['overall'] = all_chosen
        
        for k in [1, 2, 3]:
            # Inject k cycles into chosen addresses
            contam_records = []
            for addr in test_real_addrs:
                addr_txs = test_real_df[test_real_df['address_id'] == addr].copy()
                if addr in all_chosen:
                    if insertion_mode == 'end':
                        last_t = addr_txs['timestamp'].max()
                        cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
                    else:
                        times = addr_txs['timestamp'].sort_values().values
                        mid_idx = len(times) // 2
                        cur_t = pd.to_datetime(times[mid_idx]) + pd.Timedelta(minutes=float(rng.uniform(5.0, 30.0)))
                        
                    new_rows = []
                    for cyc in range(k):
                        amt = float(rng.choice(pos_amounts))
                        new_rows.append({
                            'address_id': addr,
                            'hash': f"contam_dep_{addr}_{cyc}",
                            'timestamp': cur_t,
                            'type': 1,
                            'amount': amt
                        })
                        cur_t += pd.Timedelta(minutes=float(rng.uniform(2.0, 15.0)))
                        wd_amt = round(amt * rng.uniform(0.97, 0.99), 2)
                        new_rows.append({
                            'address_id': addr,
                            'hash': f"contam_wd_{addr}_{cyc}",
                            'timestamp': cur_t,
                            'type': -1,
                            'amount': -wd_amt
                        })
                        cur_t += pd.Timedelta(hours=float(rng.uniform(2.0, 8.0)))
                    
                    addr_txs = pd.concat([addr_txs, pd.DataFrame(new_rows)], ignore_index=True)
                contam_records.append(addr_txs)
                
            eval_df = pd.concat(contam_records, ignore_index=True).sort_values(['address_id', 'timestamp'])
            all_eval_addrs = list(test_real_addrs)
            
            # Score with Fixed IF
            test_feats, _ = extract_all_features(eval_df, cfg)
            num_test = prepare_numeric_features(test_feats).reindex(columns=num_train.columns).fillna(0.0)
            if_scores = -if_model.score_samples(scaler.transform(num_test.loc[all_eval_addrs]))
            score_map_if = dict(zip(all_eval_addrs, if_scores))
            
            # Score with TF with score aggregations
            tf_scores_df, _, _ = tf_detector.score_addresses(eval_df, top_k=3, rolling_window=3)
            
            score_map_tf_mean = {}
            score_map_tf_topk = {}
            score_map_tf_wmax = {}
            for a in all_eval_addrs:
                if a in tf_scores_df.index:
                    score_map_tf_mean[a] = float(tf_scores_df.loc[a, 'transformer_surprise_mean'])
                    score_map_tf_topk[a] = float(tf_scores_df.loc[a, 'transformer_surprise_topk'])
                    score_map_tf_wmax[a] = float(tf_scores_df.loc[a, 'transformer_surprise_window_max'])
                else:
                    score_map_tf_mean[a] = 0.0
                    score_map_tf_topk[a] = 0.0
                    score_map_tf_wmax[a] = 0.0
                    
            # Calculate Recall@K for each stratum
            for st in strata_names:
                pool = strata_addrs[st]
                positives = chosen_by_stratum[st]
                K = len(positives)
                if K == 0 or len(pool) == 0:
                    continue
                    
                # Rank addresses in this stratum pool
                def get_recall(score_dict):
                    sorted_pool = sorted(pool, key=lambda a: score_dict.get(a, -1e9), reverse=True)
                    top_k_set = set(sorted_pool[:K])
                    return len(set(positives) & top_k_set) / K
                    
                rec_if = get_recall(score_map_if)
                rec_mean = get_recall(score_map_tf_mean)
                rec_topk = get_recall(score_map_tf_topk)
                rec_wmax = get_recall(score_map_tf_wmax)
                
                results[k][st]['fixed_if'].append(rec_if)
                results[k][st]['tf_mean'].append(rec_mean)
                results[k][st]['tf_topk'].append(rec_topk)
                results[k][st]['tf_window_max'].append(rec_wmax)
                
        print(f"Seed {s} completed.")
        
    return results

if __name__ == '__main__':
    print("Executing Stratified Contamination Experiment across 5 seeds...")
    results = run_stratified_contamination_experiment(seeds=[42, 43, 44, 45, 46])
    
    print("\n" + "="*80)
    print("CONTAMINATION EXPERIMENT EMPIRICAL RESULTS SUMMARY")
    print("="*80)
    
    summary_data = {}
    for k in [1, 2, 3]:
        print(f"\n--- Injected Cycles k = {k} ({2*k} transactions) ---")
        summary_data[str(k)] = {}
        for st in ['5-9', '10-24', '25+', 'overall']:
            st_data = {}
            row_str = f"  Stratum [{st}]: "
            for m in ['fixed_if', 'tf_mean', 'tf_topk', 'tf_window_max']:
                vals = results[k][st][m]
                m_mean = float(np.mean(vals))
                m_std = float(np.std(vals))
                st_data[m] = {'mean': round(m_mean, 4), 'std': round(m_std, 4)}
                row_str += f"{m}={m_mean:.3f} (±{m_std:.3f}) | "
            print(row_str)
            summary_data[str(k)][st] = st_data
            
    # Update output/validation_controls.json
    val_file = Path('output/validation_controls.json')
    if val_file.exists():
        with open(val_file, 'r') as f:
            full_val = json.load(f)
    else:
        full_val = {}
        
    full_val['contamination_test'] = {
        'description': 'Injecting k alternating deposit/withdraw cycles into existing sequences of real addresses with >=5 events',
        'evaluation_mode': 'Stratum-conditioned Recall@K where K = |Positives|, evaluated across length strata (5-9, 10-24, 25+, overall)',
        'aggregations_compared': ['fixed_if', 'tf_mean', 'tf_topk', 'tf_window_max'],
        'seeds': [42, 43, 44, 45, 46],
        'results_by_cycle': summary_data
    }
    
    with open(val_file, 'w') as f:
        json.dump(full_val, f, indent=2)
        
    print(f"\nSaved updated results to {val_file}")
