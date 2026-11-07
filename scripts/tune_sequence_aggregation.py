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
from decrypt.sequence import SequenceAnomalyDetector, extract_address_events, MAX_SEQ_LEN, MASK_DIR, MASK_AMT_BIN, MASK_DT_BIN, MASK_HOUR, MASK_DOW
import torch
import torch.nn as nn

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def run_aggregation_tuning(tuning_seeds=[101, 102, 103]):
    candidates = ['mean', 'top1', 'top2', 'top3', 'top5', 'window2', 'window3', 'window5', 'fixed_if']
    strata_names = ['5-9', '10-24', '25+', 'overall']
    
    # results[candidate][k][stratum] = list of recalls across tuning seeds
    tuning_results = {
        c: {k: {s: [] for s in strata_names} for k in [1, 2, 3]}
        for c in candidates
    }
    
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
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
        
        # Determine test addresses by history length strata
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
        
        for k in [1, 2, 3]:
            # Inject k alternating cycles
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
            score_map_if = dict(zip(all_eval_addrs, if_scores))
            
            # Extract events and compute per-event NLL for all candidates
            events_by_address = extract_address_events(eval_df, tf_detector.amt_edges, tf_detector.dt_edges)
            
            cand_scores = {c: {} for c in candidates if c != 'fixed_if'}
            
            criterion_dir = nn.CrossEntropyLoss(reduction='none')
            criterion_amt = nn.CrossEntropyLoss(reduction='none')
            criterion_dt = nn.CrossEntropyLoss(reduction='none')
            
            with torch.no_grad():
                for addr in all_eval_addrs:
                    if addr not in events_by_address:
                        for c in cand_scores:
                            cand_scores[c][addr] = 0.0
                        continue
                    data = events_by_address[addr]
                    length = data['length']
                    if length < cfg.min_seq_len:
                        for c in cand_scores:
                            cand_scores[c][addr] = 0.0
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
                    
                    # Compute aggregations
                    mean_score = float(np.mean(mean_nll))
                    cand_scores['mean'][addr] = mean_score
                    
                    # top-k
                    sorted_nll = np.sort(mean_nll)
                    cand_scores['top1'][addr] = float(sorted_nll[-1])
                    cand_scores['top2'][addr] = float(np.mean(sorted_nll[-min(2, length):]))
                    cand_scores['top3'][addr] = float(np.mean(sorted_nll[-min(3, length):]))
                    cand_scores['top5'][addr] = float(np.mean(sorted_nll[-min(5, length):]))
                    
                    # window-w
                    def get_win_max(w):
                        if length <= w:
                            return mean_score
                        return float(np.max([np.mean(mean_nll[i : i + w]) for i in range(length - w + 1)]))
                        
                    cand_scores['window2'][addr] = get_win_max(2)
                    cand_scores['window3'][addr] = get_win_max(3)
                    cand_scores['window5'][addr] = get_win_max(5)
            
            # Calculate recalls
            for st in strata_names:
                pool = strata_addrs[st]
                positives = chosen_by_stratum[st]
                K = len(positives)
                if K == 0 or len(pool) == 0:
                    continue
                    
                def eval_recall(s_dict):
                    sorted_pool = sorted(pool, key=lambda a: s_dict.get(a, -1e9), reverse=True)
                    top_k_set = set(sorted_pool[:K])
                    return len(set(positives) & top_k_set) / K
                    
                tuning_results['fixed_if'][k][st].append(eval_recall(score_map_if))
                for c in cand_scores:
                    tuning_results[c][k][st].append(eval_recall(cand_scores[c]))
                    
    return tuning_results

if __name__ == '__main__':
    res = run_aggregation_tuning(tuning_seeds=[101, 102, 103])
    
    candidates = ['fixed_if', 'mean', 'top1', 'top2', 'top3', 'top5', 'window2', 'window3', 'window5']
    strata = ['5-9', '10-24', '25+', 'overall']
    
    print("\n" + "="*90)
    print("TRANSFORMER AGGREGATION TUNING ON SEEDS [101, 102, 103] ONLY")
    print("="*90)
    
    # Calculate overall average recall across k=1,2,3 for each candidate
    grand_summary = {}
    for c in candidates:
        overall_recalls_per_k = [np.mean(res[c][k]['overall']) for k in [1, 2, 3]]
        avg_overall = float(np.mean(overall_recalls_per_k))
        grand_summary[c] = {
            'avg_overall': avg_overall,
            'k1_overall': float(np.mean(res[c][1]['overall'])),
            'k2_overall': float(np.mean(res[c][2]['overall'])),
            'k3_overall': float(np.mean(res[c][3]['overall'])),
            'by_stratum_k3': {s: float(np.mean(res[c][3][s])) for s in strata}
        }
        
    print(f"{'Candidate':<12} | {'Avg Overall':<11} | {'k=1':<8} | {'k=2':<8} | {'k=3':<8} | {'5-9 (k=3)':<10} | {'10-24 (k=3)':<11} | {'25+ (k=3)':<10}")
    print("-" * 90)
    for c in sorted(candidates, key=lambda x: grand_summary[x]['avg_overall'], reverse=True):
        gs = grand_summary[c]
        print(f"{c:<12} | {gs['avg_overall']:<11.4f} | {gs['k1_overall']:<8.4f} | {gs['k2_overall']:<8.4f} | {gs['k3_overall']:<8.4f} | {gs['by_stratum_k3']['5-9']:<10.4f} | {gs['by_stratum_k3']['10-24']:<11.4f} | {gs['by_stratum_k3']['25+']:<10.4f}")
        
    with open('output/aggregation_tuning_results.json', 'w') as f:
        json.dump({'tuning_seeds': [101, 102, 103], 'summary': grand_summary, 'raw': res}, f, indent=2)
    print("\nSaved tuning results to output/aggregation_tuning_results.json")
