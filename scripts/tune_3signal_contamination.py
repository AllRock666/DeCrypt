import sys
import json
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from decrypt.config import Config
from decrypt.features import extract_all_features, prepare_numeric_features
from decrypt.sequence import SequenceAnomalyDetector, extract_address_events, MAX_SEQ_LEN, MASK_DIR, MASK_AMT_BIN, MASK_DT_BIN, MASK_HOUR, MASK_DOW
import torch
import torch.nn as nn

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def tune_3signal_fusion(tuning_seeds=[101, 102, 103]):
    strata_names = ['5-9', '10-24', '25+', 'overall']
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
    # Store per-seed, per-k data: (eval_addrs, positives_by_stratum, pool_by_stratum, ranks_if, ranks_mean, ranks_win5)
    seed_data = {}
    
    for s in tuning_seeds:
        print(f"\n--- Running Tuning Seed {s} ---")
        rng = np.random.default_rng(s)
        real_addresses = np.array(list(df['address_id'].unique()))
        rng.shuffle(real_addresses)
        
        split_idx = int(len(real_addresses) * 0.70)
        train_addrs = set(real_addresses[:split_idx])
        test_real_addrs = set(real_addresses[split_idx:])
        
        train_df = df[df['address_id'].isin(train_addrs)].copy()
        test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
        
        cfg = Config(include_network_features=False, epochs=15, seed=s, random_state=s, contamination=0.1)
        
        # Train Fixed IF
        train_feats, _ = extract_all_features(train_df, cfg)
        num_train = prepare_numeric_features(train_feats)
        scaler = StandardScaler().fit(num_train)
        if_model = IsolationForest(contamination=0.1, random_state=s).fit(scaler.transform(num_train))
        
        # Train TF
        tf_detector = SequenceAnomalyDetector(cfg)
        tf_detector.fit(train_df)
        tf_detector.model.eval()
        
        test_counts = test_real_df.groupby('address_id').size()
        strata_addrs = {
            '5-9': list(test_counts[(test_counts >= 5) & (test_counts <= 9)].index),
            '10-24': list(test_counts[(test_counts >= 10) & (test_counts <= 24)].index),
            '25+': list(test_counts[test_counts >= 25].index),
            'overall': list(test_counts[test_counts >= 5].index)
        }
        
        chosen_by_stratum = {}
        for st, count in [('5-9', 6), ('10-24', 6), ('25+', 5)]:
            avail = strata_addrs[st]
            n_pick = min(count, len(avail))
            chosen_by_stratum[st] = list(rng.choice(avail, size=n_pick, replace=False))
            
        all_chosen = []
        for c in chosen_by_stratum.values():
            all_chosen.extend(c)
        chosen_by_stratum['overall'] = all_chosen
        
        seed_data[s] = {}
        
        criterion_dir = nn.CrossEntropyLoss(reduction='none')
        criterion_amt = nn.CrossEntropyLoss(reduction='none')
        criterion_dt = nn.CrossEntropyLoss(reduction='none')
        
        for k in [1, 2, 3]:
            contam_records = []
            for addr in test_real_addrs:
                addr_txs = test_real_df[test_real_df['address_id'] == addr].copy()
                if addr in all_chosen:
                    last_t = addr_txs['timestamp'].max()
                    cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
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
            
            # Score Fixed IF
            test_feats, _ = extract_all_features(eval_df, cfg)
            num_test = prepare_numeric_features(test_feats).reindex(columns=num_train.columns).fillna(0.0)
            if_scores = -if_model.score_samples(scaler.transform(num_test.loc[all_eval_addrs]))
            
            # TF scoring
            events_by_address = extract_address_events(eval_df, tf_detector.amt_edges, tf_detector.dt_edges)
            raw_mean = []
            raw_win5 = []
            
            with torch.no_grad():
                for addr in all_eval_addrs:
                    if addr not in events_by_address:
                        raw_mean.append(0.0)
                        raw_win5.append(0.0)
                        continue
                    data = events_by_address[addr]
                    length = data['length']
                    if length < cfg.min_seq_len:
                        raw_mean.append(0.0)
                        raw_win5.append(0.0)
                        continue
                        
                    window_size = min(length, MAX_SEQ_LEN)
                    stride = 64
                    starts = list(range(0, max(1, length - window_size + 1), stride))
                    if starts[-1] + window_size < length:
                        starts.append(length - window_size)
                        
                    event_nll_accum = np.zeros(length, dtype=np.float32)
                    event_counts = np.zeros(length, dtype=np.int32)
                    
                    for start_i in starts:
                        end_i = start_i + window_size
                        w_dir = data['direction'][start_i:end_i]
                        w_amt_b = data['amt_bin'][start_i:end_i]
                        w_dt_b = data['dt_bin'][start_i:end_i]
                        w_hour = data['hour'][start_i:end_i]
                        w_dow = data['dow'][start_i:end_i]
                        w_amt_c = data['amt_cont'][start_i:end_i]
                        w_dt_c = data['dt_cont'][start_i:end_i]
                        W = len(w_dir)
                        
                        b_dir = np.tile(w_dir, (W, 1))
                        b_amt_b = np.tile(w_amt_b, (W, 1))
                        b_dt_b = np.tile(w_dt_b, (W, 1))
                        b_hour = np.tile(w_hour, (W, 1))
                        b_dow = np.tile(w_dow, (W, 1))
                        b_amt_c = np.tile(w_amt_c, (W, 1))
                        b_dt_c = np.tile(w_dt_c, (W, 1))
                        
                        diag_idx = np.arange(W)
                        b_dir[diag_idx, diag_idx] = MASK_DIR
                        b_amt_b[diag_idx, diag_idx] = MASK_AMT_BIN
                        b_dt_b[diag_idx, diag_idx] = MASK_DT_BIN
                        b_hour[diag_idx, diag_idx] = MASK_HOUR
                        b_dow[diag_idx, diag_idx] = MASK_DOW
                        b_amt_c[diag_idx, diag_idx] = 0.0
                        b_dt_c[diag_idx, diag_idx] = 0.0
                        
                        tb_dir = torch.tensor(b_dir, dtype=torch.long)
                        tb_amt_b = torch.tensor(b_amt_b, dtype=torch.long)
                        tb_dt_b = torch.tensor(b_dt_b, dtype=torch.long)
                        tb_hour = torch.tensor(b_hour, dtype=torch.long)
                        tb_dow = torch.tensor(b_dow, dtype=torch.long)
                        tb_amt_c = torch.tensor(b_amt_c, dtype=torch.float32)
                        tb_dt_c = torch.tensor(b_dt_c, dtype=torch.float32)
                        
                        _, logits_d, logits_a, logits_t = tf_detector.model(
                            tb_dir, tb_amt_b, tb_dt_b, tb_hour, tb_dow, tb_amt_c, tb_dt_c
                        )
                        
                        pred_d = logits_d[diag_idx, diag_idx]
                        pred_a = logits_a[diag_idx, diag_idx]
                        pred_t = logits_t[diag_idx, diag_idx]
                        
                        true_d = torch.tensor(w_dir, dtype=torch.long)
                        true_a = torch.tensor(w_amt_b, dtype=torch.long)
                        true_t = torch.tensor(w_dt_b, dtype=torch.long)
                        
                        nll_d = criterion_dir(pred_d, true_d).cpu().numpy()
                        nll_a = criterion_amt(pred_a, true_a).cpu().numpy()
                        nll_t = criterion_dt(pred_t, true_t).cpu().numpy()
                        
                        w_nll = nll_d + nll_a + nll_t
                        event_nll_accum[start_i:end_i] += w_nll
                        event_counts[start_i:end_i] += 1
                        
                    mean_nll = event_nll_accum / np.maximum(event_counts, 1)
                    m_val = float(np.mean(mean_nll))
                    raw_mean.append(m_val)
                    if length <= 5:
                        raw_win5.append(m_val)
                    else:
                        w5 = float(np.max([np.mean(mean_nll[i : i + 5]) for i in range(length - 4)]))
                        raw_win5.append(w5)
                        
            # Normalize to percentile ranks across evaluation addresses
            N = len(all_eval_addrs)
            r_if = rankdata(if_scores) / N
            r_mean = rankdata(raw_mean) / N
            r_win5 = rankdata(raw_win5) / N
            
            addr_to_idx = {a: i for i, a in enumerate(all_eval_addrs)}
            
            seed_data[s][k] = {
                'addrs': all_eval_addrs,
                'addr_to_idx': addr_to_idx,
                'r_if': r_if,
                'r_mean': r_mean,
                'r_win5': r_win5,
                'positives': chosen_by_stratum,
                'pools': strata_addrs
            }

    print("\n--- Tuning 3-Signal Fusion Grid ---")
    # Grid search over w_if, w_mean, w_win5 with step 0.05
    weight_candidates = []
    steps = np.round(np.arange(0.0, 1.01, 0.05), 2)
    for w_if in steps:
        for w_mean in steps:
            w_win5 = round(1.0 - w_if - w_mean, 2)
            if w_win5 >= -1e-6:
                weight_candidates.append((w_if, w_mean, max(0.0, w_win5)))
                
    best_weights = None
    best_overall_recall = -1.0
    grid_records = []
    
    for w_if, w_mean, w_win5 in weight_candidates:
        recalls_by_seed_k = []
        for s in tuning_seeds:
            for k in [1, 2, 3]:
                d = seed_data[s][k]
                fused = w_if * d['r_if'] + w_mean * d['r_mean'] + w_win5 * d['r_win5']
                score_map = {addr: fused[d['addr_to_idx'][addr]] for addr in d['addrs']}
                
                # Overall recall
                pool = d['pools']['overall']
                pos = d['positives']['overall']
                K = len(pos)
                sorted_pool = sorted(pool, key=lambda a: score_map.get(a, -1e9), reverse=True)
                top_k_set = set(sorted_pool[:K])
                rec = len(set(pos) & top_k_set) / K
                recalls_by_seed_k.append(rec)
                
        mean_rec = float(np.mean(recalls_by_seed_k))
        grid_records.append({'weights': (w_if, w_mean, w_win5), 'mean_recall': mean_rec})
        
        if mean_rec > best_overall_recall:
            best_overall_recall = mean_rec
            best_weights = (w_if, w_mean, w_win5)
            
    print(f"\nOptimal 3-signal weights selected on tuning seeds [101, 102, 103]:")
    print(f"w_IF={best_weights[0]:.2f}, w_mean={best_weights[1]:.2f}, w_win5={best_weights[2]:.2f} (Average Tuning Contamination Recall={best_overall_recall:.4f})")
    
    # Also sort and print top 5
    top5 = sorted(grid_records, key=lambda x: x['mean_recall'], reverse=True)[:5]
    print("\nTop 5 weight configurations on tuning seeds:")
    for r in top5:
        print(f"  w_IF={r['weights'][0]:.2f}, w_mean={r['weights'][1]:.2f}, w_win5={r['weights'][2]:.2f} -> {r['mean_recall']:.4f}")
        
    out = {
        'tuning_seeds': tuning_seeds,
        'best_weights': {'w_if': best_weights[0], 'w_mean': best_weights[1], 'w_win5': best_weights[2]},
        'best_tuning_recall': best_overall_recall,
        'top5': top5
    }
    with open('output/tuning_3signal_contamination.json', 'w') as f:
        json.dump(out, f, indent=2)
        
    return best_weights

if __name__ == '__main__':
    tune_3signal_fusion()
