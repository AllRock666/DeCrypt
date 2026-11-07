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
from decrypt.benchmark import compute_bootstrap_ci, compute_paired_bootstrap_ci
import torch
import torch.nn as nn

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def run_heldout_contamination(eval_seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]):
    models = ['fixed_if', 'tf_mean', 'tf_window5']
    strata_names = ['5-9', '10-24', '25+', 'overall']
    
    # Store per-seed metrics: runs[s][k][stratum][model] = recall
    runs = {s: {k: {st: {} for st in strata_names} for k in [1, 2, 3]} for s in eval_seeds}
    
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
    for s in eval_seeds:
        print(f"\n--- Running Evaluation Seed {s} ---")
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
            # Inject k cycles
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
            
            # Compute TF Mean and Window-5
            events_by_address = extract_address_events(eval_df, tf_detector.amt_edges, tf_detector.dt_edges)
            score_map_mean = {}
            score_map_win5 = {}
            
            criterion_dir = nn.CrossEntropyLoss(reduction='none')
            criterion_amt = nn.CrossEntropyLoss(reduction='none')
            criterion_dt = nn.CrossEntropyLoss(reduction='none')
            
            with torch.no_grad():
                for addr in all_eval_addrs:
                    if addr not in events_by_address:
                        score_map_mean[addr] = 0.0
                        score_map_win5[addr] = 0.0
                        continue
                    data = events_by_address[addr]
                    length = data['length']
                    if length < cfg.min_seq_len:
                        score_map_mean[addr] = 0.0
                        score_map_win5[addr] = 0.0
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
                    mean_val = float(np.mean(mean_nll))
                    score_map_mean[addr] = mean_val
                    
                    if length <= 5:
                        score_map_win5[addr] = mean_val
                    else:
                        score_map_win5[addr] = float(np.max([np.mean(mean_nll[i : i + 5]) for i in range(length - 4)]))
                        
            # Record recalls for this seed
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
                    
                runs[s][k][st]['fixed_if'] = eval_recall(score_map_if)
                runs[s][k][st]['tf_mean'] = eval_recall(score_map_mean)
                runs[s][k][st]['tf_window5'] = eval_recall(score_map_win5)
                
    return runs

if __name__ == '__main__':
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    runs = run_heldout_contamination(eval_seeds=seeds)
    
    strata = ['5-9', '10-24', '25+', 'overall']
    models = ['fixed_if', 'tf_mean', 'tf_window5']
    
    print("\n" + "="*120)
    print("HELDOUT CONTAMINATION TEST RESULTS (10 EVALUATION SEEDS) WITH PAIRED BOOTSTRAP CIs")
    print("="*120)
    
    report_dict = {}
    for k in [1, 2, 3]:
        print(f"\n==================== Injected Alternating Cycles k = {k} ({2*k} events) ====================")
        report_dict[str(k)] = {}
        for st in strata:
            vals_if = [runs[s][k][st]['fixed_if'] for s in seeds]
            vals_mean = [runs[s][k][st]['tf_mean'] for s in seeds]
            vals_win5 = [runs[s][k][st]['tf_window5'] for s in seeds]
            
            ci_if = compute_bootstrap_ci(vals_if)
            ci_mean = compute_bootstrap_ci(vals_mean)
            ci_win5 = compute_bootstrap_ci(vals_win5)
            
            paired_diff_mean_vs_if = compute_paired_bootstrap_ci(vals_mean, vals_if)
            paired_diff_win5_vs_if = compute_paired_bootstrap_ci(vals_win5, vals_if)
            
            report_dict[str(k)][st] = {
                'fixed_if': ci_if,
                'tf_mean': ci_mean,
                'tf_window5': ci_win5,
                'diff_mean_minus_if': paired_diff_mean_vs_if,
                'diff_win5_minus_if': paired_diff_win5_vs_if
            }
            
            print(f"Stratum [{st:<7}]:")
            print(f"  Fixed IF:    {ci_if['mean']:.3f} [{ci_if['ci_lower']:.3f}, {ci_if['ci_upper']:.3f}]")
            print(f"  TF Mean:     {ci_mean['mean']:.3f} [{ci_mean['ci_lower']:.3f}, {ci_mean['ci_upper']:.3f}] | Diff vs IF: {paired_diff_mean_vs_if['mean']:+.3f} [{paired_diff_mean_vs_if['ci_lower']:+.3f}, {paired_diff_mean_vs_if['ci_upper']:+.3f}]")
            print(f"  TF Window-5: {ci_win5['mean']:.3f} [{ci_win5['ci_lower']:.3f}, {ci_win5['ci_upper']:.3f}] | Diff vs IF: {paired_diff_win5_vs_if['mean']:+.3f} [{paired_diff_win5_vs_if['ci_lower']:+.3f}, {paired_diff_win5_vs_if['ci_upper']:+.3f}]")
            
    val_file = Path('output/validation_controls.json')
    if val_file.exists():
        with open(val_file, 'r') as f:
            full_val = json.load(f)
    else:
        full_val = {}
        
    full_val['contamination_heldout_paired_cis'] = {
        'description': 'Stratum-conditioned Recall@K for in-situ contamination (k=1,2,3 cycles) on 10 held-out evaluation seeds',
        'chosen_aggregation': 'tf_window5 (selected on tuning seeds 101-103)',
        'eval_seeds': seeds,
        'results_by_cycle': report_dict
    }
    with open(val_file, 'w') as f:
        json.dump(full_val, f, indent=2)
    print(f"\nSaved held-out contamination results to {val_file}")
