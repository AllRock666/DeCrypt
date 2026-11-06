import sys
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

def run_contamination_experiment(seeds=[42, 43, 44, 45, 46], n_contaminate=20, insertion_mode='end'):
    results = {1: {'if': [], 'tf': []}, 2: {'if': [], 'tf': []}, 3: {'if': [], 'tf': []}}
    
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
    for s in seeds:
        rng = np.random.default_rng(s)
        real_addresses = np.array(list(df['address_id'].unique()))
        rng.shuffle(real_addresses)
        
        split_idx = int(len(real_addresses) * 0.70)
        train_addrs = set(real_addresses[:split_idx])
        test_real_addrs = set(real_addresses[split_idx:])
        
        train_df = df[df['address_id'].isin(train_addrs)].copy()
        test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
        
        cfg = Config(include_network_features=False, epochs=15, seed=s, random_state=s, contamination=0.1)
        
        # Train models on clean train data
        train_feats, _ = extract_all_features(train_df, cfg)
        num_train = prepare_numeric_features(train_feats)
        scaler = StandardScaler().fit(num_train)
        if_model = IsolationForest(contamination=0.1, random_state=s).fit(scaler.transform(num_train))
        
        tf_detector = SequenceAnomalyDetector(cfg)
        tf_detector.fit(train_df)
        
        # Test addresses with >= 5 events
        test_counts = test_real_df.groupby('address_id').size()
        eligible_addrs = list(test_counts[test_counts >= 5].index)
        
        # Pick n_contaminate addresses
        chosen_addrs = rng.choice(eligible_addrs, size=min(n_contaminate, len(eligible_addrs)), replace=False)
        
        for k in [1, 2, 3]:
            # Build contaminated test dataframe
            contam_records = []
            for addr in test_real_addrs:
                addr_txs = test_real_df[test_real_df['address_id'] == addr].copy()
                if addr in chosen_addrs:
                    # Inject k cycles
                    if insertion_mode == 'end':
                        last_t = addr_txs['timestamp'].max()
                        cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
                    else:
                        # Insert in middle
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
            
            # Score with IF
            test_feats, _ = extract_all_features(eval_df, cfg)
            num_test = prepare_numeric_features(test_feats).reindex(columns=num_train.columns).fillna(0.0)
            if_scores = -if_model.score_samples(scaler.transform(num_test.loc[all_eval_addrs]))
            
            # Score with TF
            tf_scores_df, _, _ = tf_detector.score_addresses(eval_df)
            tf_scores = np.array([float(tf_scores_df.loc[a, 'transformer_surprise']) if a in tf_scores_df.index else 0.0 for a in all_eval_addrs])
            
            # Top-K recall (K = len(chosen_addrs))
            k_budget = len(chosen_addrs)
            top_k_if = set([all_eval_addrs[i] for i in np.argsort(-if_scores)[:k_budget]])
            top_k_tf = set([all_eval_addrs[i] for i in np.argsort(-tf_scores)[:k_budget]])
            
            rec_if = len(set(chosen_addrs) & top_k_if) / k_budget
            rec_tf = len(set(chosen_addrs) & top_k_tf) / k_budget
            
            results[k]['if'].append(rec_if)
            results[k]['tf'].append(rec_tf)
            
        print(f"Seed {s} complete. Recalls so far:")
        for k in [1, 2, 3]:
            print(f"  k={k}: IF={np.mean(results[k]['if']):.3f}, TF={np.mean(results[k]['tf']):.3f}")
            
    return results

if __name__ == '__main__':
    print("Testing Contamination Experiment across 5 seeds...")
    res = run_contamination_experiment(seeds=[42, 43, 44, 45, 46], n_contaminate=20, insertion_mode='end')
    print("\nFINAL SUMMARY:")
    for k in [1, 2, 3]:
        print(f"k={k} cycles:")
        print(f"  Fixed IF Recall:    mean={np.mean(res[k]['if']):.3f} (std={np.std(res[k]['if']):.3f})")
        print(f"  Transformer Recall: mean={np.mean(res[k]['tf']):.3f} (std={np.std(res[k]['tf']):.3f})")
