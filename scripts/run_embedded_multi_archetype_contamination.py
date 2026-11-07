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
from decrypt.benchmark import (
    compute_hierarchical_bootstrap_ci,
    compute_hierarchical_paired_ci,
    inject_alternating_cycles,
    inject_burst,
    inject_structuring,
    inject_bot_periodic
)
import torch
import torch.nn as nn

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

def run_multi_archetype_contamination_eval(eval_seeds=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51]):
    archetypes = ['burst', 'structuring', 'bot_periodic', 'alternating_k1', 'alternating_k2', 'alternating_k3']
    models = ['fixed_if', 'batch_fusion', 'streaming_fusion']
    strata_names = ['overall', '5-9', '10-24', '25+']
    
    # 3-signal weights
    w_if_3s = 0.20
    w_mean_3s = 0.10
    w_win5_3s = 0.70
    
    indicators = {arch: {st: {m: {} for m in models} for st in strata_names} for arch in archetypes}
    
    pos_amounts = df['amount'].abs().values
    pos_amounts = pos_amounts[pos_amounts > 0]
    
    criterion_dir = nn.CrossEntropyLoss(reduction='none')
    criterion_amt = nn.CrossEntropyLoss(reduction='none')
    criterion_dt = nn.CrossEntropyLoss(reduction='none')
    
    for s in eval_seeds:
        print(f"\n--- Running Evaluation Seed {s} ---")
        ss = np.random.SeedSequence(s)
        child_seeds = ss.generate_state(5)
        seed_split = int(child_seeds[0]) % (2**31 - 1)
        seed_injection = int(child_seeds[1]) % (2**31 - 1)
        seed_if = int(child_seeds[2]) % (2**31 - 1)
        seed_tf = int(child_seeds[3]) % (2**31 - 1)
        
        rng_split = np.random.default_rng(seed_split)
        real_addresses = np.sort(df['address_id'].unique())
        rng_split.shuffle(real_addresses)
        
        split_idx = int(len(real_addresses) * 0.70)
        train_addrs = set(real_addresses[:split_idx])
        test_real_addrs = set(real_addresses[split_idx:])
        
        train_df = df[df['address_id'].isin(train_addrs)].copy()
        test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
        
        cfg = Config(include_network_features=False, epochs=15, seed=seed_tf, random_state=seed_tf, contamination=0.1)
        
        # Train Fixed IF
        print("  Training Fixed IF...")
        train_feats, _ = extract_all_features(train_df, cfg)
        num_train = prepare_numeric_features(train_feats)
        scaler = StandardScaler().fit(num_train)
        if_model = IsolationForest(contamination=0.1, random_state=seed_if).fit(scaler.transform(num_train))
        
        # Train TF
        print("  Training Transformer (15 epochs)...")
        tf_detector = SequenceAnomalyDetector(cfg)
        tf_detector.fit(train_df)
        tf_detector.model.eval()
        
        test_counts = test_real_df.groupby('address_id').size()
        strata_addrs = {
            'overall': sorted(list(test_counts[test_counts >= 5].index)),
            '5-9': sorted(list(test_counts[(test_counts >= 5) & (test_counts <= 9)].index)),
            '10-24': sorted(list(test_counts[(test_counts >= 10) & (test_counts <= 24)].index)),
            '25+': sorted(list(test_counts[test_counts >= 25].index))
        }
        
        rng_inj = np.random.default_rng(seed_injection)
        chosen_by_stratum = {}
        for st, count in [('5-9', 6), ('10-24', 6), ('25+', 5)]:
            avail = strata_addrs[st]
            n_pick = min(count, len(avail))
            chosen_by_stratum[st] = sorted(list(rng_inj.choice(avail, size=n_pick, replace=False)))
            
        all_chosen = []
        for c in [chosen_by_stratum['5-9'], chosen_by_stratum['10-24'], chosen_by_stratum['25+']]:
            all_chosen.extend(c)
        chosen_by_stratum['overall'] = sorted(all_chosen)
        
        all_eval_addrs = sorted(list(test_real_addrs))
        N = len(all_eval_addrs)
        
        for arch in archetypes:
            # Construct injected eval dataset
            contam_records = []
            for addr in all_eval_addrs:
                addr_txs = test_real_df[test_real_df['address_id'] == addr].copy()
                if addr in all_chosen:
                    if arch == 'burst':
                        addr_txs = inject_burst(addr_txs, pos_amounts=pos_amounts, rng=rng_inj)
                    elif arch == 'structuring':
                        addr_txs = inject_structuring(addr_txs, pos_amounts=pos_amounts, rng=rng_inj)
                    elif arch == 'bot_periodic':
                        addr_txs = inject_bot_periodic(addr_txs, pos_amounts=pos_amounts, rng=rng_inj)
                    elif arch == 'alternating_k1':
                        addr_txs = inject_alternating_cycles(addr_txs, k=1, pos_amounts=pos_amounts, rng=rng_inj)
                    elif arch == 'alternating_k2':
                        addr_txs = inject_alternating_cycles(addr_txs, k=2, pos_amounts=pos_amounts, rng=rng_inj)
                    elif arch == 'alternating_k3':
                        addr_txs = inject_alternating_cycles(addr_txs, k=3, pos_amounts=pos_amounts, rng=rng_inj)
                contam_records.append(addr_txs)
                
            eval_df = pd.concat(contam_records, ignore_index=True).sort_values(['address_id', 'timestamp'])
            
            # Score Fixed IF
            test_feats, _ = extract_all_features(eval_df, cfg)
            num_test = prepare_numeric_features(test_feats).reindex(columns=num_train.columns).fillna(0.0)
            if_scores = -if_model.score_samples(scaler.transform(num_test.loc[all_eval_addrs]))
            
            # Score TF
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
                        
            # Percentile ranks
            r_if = rankdata(if_scores) / N
            r_mean = rankdata(raw_mean) / N
            r_win5 = rankdata(raw_win5) / N
            
            fused_batch = 0.8 * r_if + 0.2 * r_mean
            fused_3s = w_if_3s * r_if + w_mean_3s * r_mean + w_win5_3s * r_win5
            
            score_dicts = {
                'fixed_if': dict(zip(all_eval_addrs, r_if)),
                'batch_fusion': dict(zip(all_eval_addrs, fused_batch)),
                'streaming_fusion': dict(zip(all_eval_addrs, fused_3s))
            }
            
            for st in strata_names:
                pool = strata_addrs[st]
                positives = chosen_by_stratum[st]
                K = len(positives)
                if K == 0 or len(pool) == 0:
                    continue
                    
                def get_address_indicators(s_dict):
                    sorted_pool = sorted(pool, key=lambda a: s_dict.get(a, -1e9), reverse=True)
                    top_k_set = set(sorted_pool[:K])
                    return [1 if a in top_k_set else 0 for a in positives]
                    
                for m_name in models:
                    indicators[arch][st][m_name][s] = get_address_indicators(score_dicts[m_name])

    return indicators

if __name__ == '__main__':
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    indicators = run_multi_archetype_contamination_eval(eval_seeds=seeds)
    
    archetypes = ['burst', 'structuring', 'bot_periodic', 'alternating_k1', 'alternating_k2', 'alternating_k3']
    strata = ['overall', '5-9', '10-24', '25+']
    models = ['fixed_if', 'batch_fusion', 'streaming_fusion']
    
    print("\n" + "="*120)
    print("EMBEDDED MULTI-ARCHETYPE CONTAMINATION BENCHMARK (HIERARCHICAL SEED & ADDRESS BOOTSTRAP CIs)")
    print("Models: Fixed IF vs Batch Fusion (0.8/0.2 Mean) vs Streaming Fusion (3-Signal 0.2/0.1/0.7)")
    print("="*120)
    
    report_dict = {}
    
    for arch in archetypes:
        print(f"\n==================== Archetype: {arch.upper()} ====================")
        report_dict[arch] = {}
        
        # Table Header
        print(f"| Stratum | Fixed IF Recall@K | Batch Fusion Recall@K | Diff (Batch vs IF) | Streaming Fusion Recall@K | Diff (Streaming vs IF) |")
        print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
        
        for st in strata:
            ind_if = indicators[arch][st]['fixed_if']
            ind_batch = indicators[arch][st]['batch_fusion']
            ind_stream = indicators[arch][st]['streaming_fusion']
            
            ci_if = compute_hierarchical_bootstrap_ci(ind_if)
            ci_batch = compute_hierarchical_bootstrap_ci(ind_batch)
            ci_stream = compute_hierarchical_bootstrap_ci(ind_stream)
            
            diff_batch = compute_hierarchical_paired_ci(ind_batch, ind_if)
            diff_stream = compute_hierarchical_paired_ci(ind_stream, ind_if)
            
            report_dict[arch][st] = {
                'fixed_if': ci_if,
                'batch_fusion': ci_batch,
                'streaming_fusion': ci_stream,
                'diff_batch_vs_if': diff_batch,
                'diff_stream_vs_if': diff_stream
            }
            
            print(
                f"| **{st}** | "
                f"{ci_if['mean']:.3f} [{ci_if['ci_lower']:.3f}, {ci_if['ci_upper']:.3f}] | "
                f"{ci_batch['mean']:.3f} [{ci_batch['ci_lower']:.3f}, {ci_batch['ci_upper']:.3f}] | "
                f"{diff_batch['mean']:+.3f} [{diff_batch['ci_lower']:+.3f}, {diff_batch['ci_upper']:+.3f}] | "
                f"{ci_stream['mean']:.3f} [{ci_stream['ci_lower']:.3f}, {ci_stream['ci_upper']:.3f}] | "
                f"{diff_stream['mean']:+.3f} [{diff_stream['ci_lower']:+.3f}, {diff_stream['ci_upper']:+.3f}] |"
            )
            
    val_file = Path('output/validation_controls.json')
    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            full_val = json.load(f)
    else:
        full_val = {}
        
    full_val['embedded_multi_archetype_contamination'] = {
        'description': 'In-situ contamination evaluation for embedded archetypes (burst, structuring, bot-periodic, alternating k=1..3) on addresses with >=5 events across 10 evaluation seeds (42-51)',
        'bootstrap_method': 'hierarchical_cluster_bootstrap (Level 1: seeds with replacement; Level 2: address indicators within seed with replacement)',
        'eval_seeds': seeds,
        'models': {
            'fixed_if': 'Isolation Forest on 23 tabular features',
            'batch_fusion': '0.80 * IF + 0.20 * Transformer Mean Surprise',
            'streaming_fusion': '0.20 * IF + 0.10 * Transformer Mean + 0.70 * Transformer Window-5'
        },
        'results_by_archetype': report_dict
    }
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(full_val, f, indent=2)
    print(f"\nSaved embedded multi-archetype contamination results with hierarchical CIs to {val_file}")
