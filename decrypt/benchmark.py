import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from .config import Config, setup_logger, default_config
from .features import extract_all_features, extract_legacy_features, prepare_numeric_features, build_heuristic_flow_graph
from .sequence import SequenceAnomalyDetector

logger = setup_logger(__name__, default_config)

def compute_bootstrap_ci(data: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """
    Computes mean and 95% bootstrap confidence interval.
    """
    arr = np.array(data)
    mean_val = float(np.mean(arr))
    if len(arr) <= 1 or np.std(arr) == 0:
        return {'mean': mean_val, 'ci_lower': mean_val, 'ci_upper': mean_val, 'std': 0.0}
        
    rng = np.random.default_rng(seed)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {
        'mean': mean_val,
        'ci_lower': lower,
        'ci_upper': upper,
        'std': float(np.std(arr))
    }

def compute_paired_bootstrap_ci(a: List[float], b: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> Dict[str, float]:
    """
    Computes mean and 95% bootstrap confidence interval of the paired difference (a - b).
    """
    diffs = [float(x - y) for x, y in zip(a, b)]
    return compute_bootstrap_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)

def compute_hierarchical_bootstrap_ci(
    seed_addr_outcomes: Dict[Any, List[int]],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes mean and 95% bootstrap confidence interval using hierarchical cluster bootstrap.
    Resamples seeds (clusters) with replacement, and within each drawn seed, resamples
    individual address detection indicators with replacement.
    Prevents variance collapse when per-seed summary averages coincide.
    """
    rng = np.random.default_rng(seed)
    seeds = list(seed_addr_outcomes.keys())
    all_vals = np.concatenate([seed_addr_outcomes[s] for s in seeds]) if seeds else np.array([])
    if len(all_vals) == 0:
        return {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'std': 0.0}
    mean_val = float(np.mean(all_vals))
    
    boot_means = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_vals = []
        for s in sampled_seeds:
            v = seed_addr_outcomes[s]
            if len(v) > 0:
                boot_vals.append(rng.choice(v, size=len(v), replace=True))
        if boot_vals:
            boot_means.append(np.mean(np.concatenate(boot_vals)))
        else:
            boot_means.append(mean_val)
            
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {
        'mean': mean_val,
        'ci_lower': lower,
        'ci_upper': upper,
        'std': float(np.std(boot_means))
    }

def compute_hierarchical_paired_ci(
    outcomes_a: Dict[Any, List[int]],
    outcomes_b: Dict[Any, List[int]],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes mean and 95% bootstrap confidence interval of paired difference (a - b)
    using hierarchical cluster bootstrap (level 1 = seeds, level 2 = positive address indicators).
    """
    rng = np.random.default_rng(seed)
    seeds = list(outcomes_a.keys())
    diffs = {}
    for s in seeds:
        a_arr = np.array(outcomes_a[s])
        b_arr = np.array(outcomes_b[s])
        diffs[s] = a_arr - b_arr
        
    all_diffs = np.concatenate([diffs[s] for s in seeds]) if seeds else np.array([])
    if len(all_diffs) == 0:
        return {'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'std': 0.0}
    mean_val = float(np.mean(all_diffs))
    
    boot_means = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot_diffs = []
        for s in sampled_seeds:
            d = diffs[s]
            if len(d) > 0:
                boot_diffs.append(rng.choice(d, size=len(d), replace=True))
        if boot_diffs:
            boot_means.append(np.mean(np.concatenate(boot_diffs)))
        else:
            boot_means.append(mean_val)
            
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return {
        'mean': mean_val,
        'ci_lower': lower,
        'ci_upper': upper,
        'std': float(np.std(boot_means))
    }

def run_graph_sanity_test(df: pd.DataFrame, config: Config, n_runs: int = 20, seed: int = 42) -> Dict[str, Any]:
    """
    Evaluates the heuristic flow graph against null permutation models:
    20 runs with shuffled timestamps and 20 runs with shuffled amounts.
    """
    logger.info("Running Heuristic Flow Graph Permutation Sanity Test (20 runs each)...")
    real_G = build_heuristic_flow_graph(df, config)
    real_edges = real_G.number_of_edges()
    
    rng = np.random.default_rng(seed)
    
    # 1. Shuffled timestamps
    time_edges = []
    for _ in range(n_runs):
        df_t = df.copy()
        df_t['timestamp'] = rng.permutation(df['timestamp'].values)
        G_t = build_heuristic_flow_graph(df_t, config)
        time_edges.append(G_t.number_of_edges())
        
    # 2. Shuffled amounts (preserving type/direction)
    amt_edges = []
    w_idx = df[df['type'] == -1].index
    d_idx = df[df['type'] == 1].index
    for _ in range(n_runs):
        df_a = df.copy()
        df_a.loc[w_idx, 'amount'] = rng.permutation(df_a.loc[w_idx, 'amount'].values)
        df_a.loc[d_idx, 'amount'] = rng.permutation(df_a.loc[d_idx, 'amount'].values)
        G_a = build_heuristic_flow_graph(df_a, config)
        amt_edges.append(G_a.number_of_edges())
        
    mean_time = float(np.mean(time_edges))
    mean_amt = float(np.mean(amt_edges))
    
    # Check if real edges is clearly above shuffled distribution
    is_above_time = real_edges > (mean_time + 2 * np.std(time_edges))
    is_above_amt = real_edges > (mean_amt + 2 * np.std(amt_edges))
    is_high_confidence = is_above_time and is_above_amt
    
    sanity_results = {
        'real_edges': real_edges,
        'timestamp_shuffled': {
            'mean': mean_time,
            'std': float(np.std(time_edges)),
            'min': int(np.min(time_edges)),
            'max': int(np.max(time_edges)),
            'runs': time_edges
        },
        'amount_shuffled': {
            'mean': mean_amt,
            'std': float(np.std(amt_edges)),
            'min': int(np.min(amt_edges)),
            'max': int(np.max(amt_edges)),
            'runs': amt_edges
        },
        'is_high_confidence': bool(is_high_confidence),
        'label': "high-confidence" if is_high_confidence else "low-confidence"
    }
    
    logger.info(
        f"Graph Sanity: Real={real_edges} | Shuffled Time={mean_time:.1f} (min={np.min(time_edges)}, max={np.max(time_edges)}) | "
        f"Shuffled Amt={mean_amt:.1f} (min={np.min(amt_edges)}, max={np.max(amt_edges)}) -> Status: {sanity_results['label'].upper()}"
    )
    return sanity_results

def generate_synthetic_anomalies(
    real_df: pd.DataFrame,
    n_per_type: int = 20,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generates synthetic anomalies sampled strictly from empirical data distributions.
    """
    rng = np.random.default_rng(seed)
    
    real_amounts = real_df['amount'].abs().values
    pos_amounts = real_amounts[real_amounts > 0]
    min_time = real_df['timestamp'].min()
    max_time = real_df['timestamp'].max()
    time_span = (max_time - min_time).total_seconds()
    
    records = []
    anomaly_types = {}
    
    # Archetype A: Structuring ("Smurfing")
    # Multiple deposits just below a round threshold, followed by 1 full withdrawal
    # Max threshold capped at 250 so combined withdrawal stays strictly within empirical range (max 4000 <= 4579)
    for i in range(n_per_type):
        addr = f"synth_structuring_{i:02d}"
        anomaly_types[addr] = "structuring"
        threshold = rng.choice([20.0, 50.0, 100.0, 250.0])
        n_deposits = rng.integers(8, 16)
        
        start_sec = rng.uniform(0, max(1, time_span - 15 * 86400))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        total_dep = 0.0
        for _ in range(n_deposits):
            amt = round(rng.uniform(0.85, 0.98) * threshold, 2)
            total_dep += amt
            cur_time += pd.Timedelta(hours=float(rng.uniform(4, 24)))
            records.append({
                'address_id': addr,
                'hash': f"h_struct_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': 1,
                'amount': amt
            })
            
        cur_time += pd.Timedelta(hours=float(rng.uniform(1, 6)))
        records.append({
            'address_id': addr,
            'hash': f"h_struct_{addr}_{len(records)}",
            'timestamp': cur_time,
            'type': -1,
            'amount': -round(total_dep, 2)
        })

    # Archetype B: Burst
    # 30-50 transactions within minutes (high frequency burst)
    for i in range(n_per_type):
        addr = f"synth_burst_{i:02d}"
        anomaly_types[addr] = "burst"
        n_txs = rng.integers(30, 50)
        
        start_sec = rng.uniform(0, max(1, time_span - 3600))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        for _ in range(n_txs):
            delta = float(rng.uniform(5, 30))
            cur_time += pd.Timedelta(seconds=delta)
            amt = float(rng.choice(pos_amounts))
            direction = int(rng.choice([1, -1]))
            records.append({
                'address_id': addr,
                'hash': f"h_burst_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': direction,
                'amount': amt if direction == 1 else -amt
            })

    # Archetype C: Pass-through / Round-trip Mixer (with delay variants: 15m, 1h, 6h, 24h)
    delay_tags = ['15m', '1h', '6h', '24h']
    for i in range(n_per_type):
        delay_tag = delay_tags[i % len(delay_tags)]
        addr = f"synth_passthrough_{delay_tag}_{i:02d}"
        anomaly_types[addr] = "pass_through"
        n_cycles = rng.integers(6, 12)
        
        start_sec = rng.uniform(0, max(1, time_span - 10 * 86400))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        for _ in range(n_cycles):
            amt = float(rng.choice(pos_amounts))
            cur_time += pd.Timedelta(hours=float(rng.uniform(6, 24)))
            records.append({
                'address_id': addr,
                'hash': f"h_pass_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': 1,
                'amount': amt
            })
            
            # Withdrawal delay based on variant
            if delay_tag == '15m':
                cur_time += pd.Timedelta(minutes=float(rng.uniform(2, 15)))
            elif delay_tag == '1h':
                cur_time += pd.Timedelta(minutes=float(rng.uniform(50, 70)))
            elif delay_tag == '6h':
                cur_time += pd.Timedelta(hours=float(rng.uniform(5.0, 7.0)))
            elif delay_tag == '24h':
                cur_time += pd.Timedelta(hours=float(rng.uniform(22.0, 26.0)))
                
            wd_amt = round(amt * rng.uniform(0.97, 0.99), 2)
            records.append({
                'address_id': addr,
                'hash': f"h_pass_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': -1,
                'amount': -wd_amt
            })

    # Archetype D: Bot-Periodic
    # Identical interval and identical amounts
    for i in range(n_per_type):
        addr = f"synth_bot_{i:02d}"
        anomaly_types[addr] = "bot_periodic"
        n_txs = rng.integers(20, 35)
        fixed_dt = float(rng.choice([300.0, 600.0, 1800.0, 3600.0]))
        fixed_amt = float(rng.choice([5.0, 10.0, 25.0, 50.0]))
        
        start_sec = rng.uniform(0, max(1, time_span - n_txs * fixed_dt))
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        for _ in range(n_txs):
            cur_time += pd.Timedelta(seconds=fixed_dt)
            direction = int(rng.choice([1, -1]))
            records.append({
                'address_id': addr,
                'hash': f"h_bot_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': direction,
                'amount': fixed_amt if direction == 1 else -fixed_amt
            })

    # Archetype E: Dormant-then-Spike
    # Inactivity for months, followed by sudden spike in volume
    for i in range(n_per_type):
        addr = f"synth_dormant_{i:02d}"
        anomaly_types[addr] = "dormant_spike"
        start_sec = rng.uniform(0, 30 * 86400)
        cur_time = min_time + pd.Timedelta(seconds=start_sec)
        
        # 1 initial deposit
        records.append({
            'address_id': addr,
            'hash': f"h_dorm_{addr}_{len(records)}",
            'timestamp': cur_time,
            'type': 1,
            'amount': 10.0
        })
        
        # Long dormancy (150-250 days)
        dormant_days = rng.uniform(150, 250)
        cur_time += pd.Timedelta(days=float(dormant_days))
        
        # Violent spike in 12 hours
        n_spike = rng.integers(15, 25)
        for _ in range(n_spike):
            cur_time += pd.Timedelta(minutes=float(rng.uniform(10, 45)))
            amt = float(rng.choice(pos_amounts) * rng.uniform(2.0, 5.0))
            direction = int(rng.choice([1, -1]))
            records.append({
                'address_id': addr,
                'hash': f"h_dorm_{addr}_{len(records)}",
                'timestamp': cur_time,
                'type': direction,
                'amount': amt if direction == 1 else -amt
            })

    anomaly_df = pd.DataFrame(records)
    return anomaly_df, anomaly_types

def inject_alternating_cycles(
    addr_df,
    k: int,
    pos_amounts = None,
    rng = None,
    seed: int = 42
):
    """
    Injects k alternating deposit-withdrawal cycles into an existing address sequence.
    Each cycle consists of:
    - 1 deposit of an empirical positive amount
    - 1 withdrawal delayed by 2-15 minutes of ~97-99% of that amount
    Followed by an inter-cycle gap of 2-8 hours.
    Preserves original address events and appends newly generated cycle events.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if len(addr_df) == 0:
        return addr_df.copy()
    if pos_amounts is None or len(pos_amounts) == 0:
        pos_amounts = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        
    addr = addr_df['address_id'].iloc[0]
    last_t = addr_df['timestamp'].max()
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
    
    res = pd.concat([addr_df, pd.DataFrame(new_rows)], ignore_index=True)
    return res.sort_values('timestamp').reset_index(drop=True)

def inject_burst(
    addr_df,
    pos_amounts = None,
    rng = None,
    seed: int = 42,
    n_txs: int = None
):
    """
    Injects a high-frequency burst into an existing real address history.
    Default 20-30 rapid transactions with short intervals (5-30s).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if len(addr_df) == 0:
        return addr_df.copy()
    if pos_amounts is None or len(pos_amounts) == 0:
        pos_amounts = np.array([1.0, 5.0, 10.0, 20.0, 50.0])
        
    addr = addr_df['address_id'].iloc[0]
    last_t = addr_df['timestamp'].max()
    cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
    
    if n_txs is None:
        n_txs = int(rng.integers(20, 31))
        
    new_rows = []
    for i in range(n_txs):
        delta = float(rng.uniform(5.0, 30.0))
        cur_t += pd.Timedelta(seconds=delta)
        amt = float(rng.choice(pos_amounts))
        direction = int(rng.choice([1, -1]))
        new_rows.append({
            'address_id': addr,
            'hash': f"contam_burst_{addr}_{i}",
            'timestamp': cur_t,
            'type': direction,
            'amount': amt if direction == 1 else -amt
        })
        
    res = pd.concat([addr_df, pd.DataFrame(new_rows)], ignore_index=True)
    return res.sort_values('timestamp').reset_index(drop=True)

def inject_structuring(
    addr_df,
    pos_amounts = None,
    rng = None,
    seed: int = 42,
    threshold: float = None,
    n_deposits: int = None
):
    """
    Injects structuring / smurfing transactions into an existing real address history:
    8-12 repeated deposits just below a round threshold (e.g. 10.0, 50.0, 100.0),
    spaced by 4-24 hours, followed by a consolidated withdrawal.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if len(addr_df) == 0:
        return addr_df.copy()
        
    addr = addr_df['address_id'].iloc[0]
    last_t = addr_df['timestamp'].max()
    cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
    
    if threshold is None:
        threshold = float(rng.choice([10.0, 50.0, 100.0]))
    if n_deposits is None:
        n_deposits = int(rng.integers(8, 13))
        
    total_dep = 0.0
    new_rows = []
    for i in range(n_deposits):
        amt = round(float(rng.uniform(0.85, 0.98)) * threshold, 2)
        total_dep += amt
        cur_t += pd.Timedelta(hours=float(rng.uniform(4.0, 24.0)))
        new_rows.append({
            'address_id': addr,
            'hash': f"contam_struct_{addr}_{i}",
            'timestamp': cur_t,
            'type': 1,
            'amount': amt
        })
        
    cur_t += pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
    new_rows.append({
        'address_id': addr,
        'hash': f"contam_struct_wd_{addr}",
        'timestamp': cur_t,
        'type': -1,
        'amount': -round(total_dep, 2)
    })
    
    res = pd.concat([addr_df, pd.DataFrame(new_rows)], ignore_index=True)
    return res.sort_values('timestamp').reset_index(drop=True)

def inject_bot_periodic(
    addr_df,
    pos_amounts = None,
    rng = None,
    seed: int = 42,
    fixed_dt: float = None,
    fixed_amt: float = None,
    n_txs: int = None
):
    """
    Injects bot-periodic transactions into an existing real address history:
    15-25 transactions with identical interval (fixed_dt) and identical amount (fixed_amt).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if len(addr_df) == 0:
        return addr_df.copy()
        
    addr = addr_df['address_id'].iloc[0]
    last_t = addr_df['timestamp'].max()
    cur_t = last_t + pd.Timedelta(hours=float(rng.uniform(1.0, 6.0)))
    
    if fixed_dt is None:
        fixed_dt = float(rng.choice([300.0, 600.0, 1800.0, 3600.0]))
    if fixed_amt is None:
        fixed_amt = float(rng.choice([5.0, 10.0, 25.0, 50.0]))
    if n_txs is None:
        n_txs = int(rng.integers(15, 26))
        
    new_rows = []
    for i in range(n_txs):
        cur_t += pd.Timedelta(seconds=fixed_dt)
        direction = int(rng.choice([1, -1]))
        new_rows.append({
            'address_id': addr,
            'hash': f"contam_bot_{addr}_{i}",
            'timestamp': cur_t,
            'type': direction,
            'amount': fixed_amt if direction == 1 else -fixed_amt
        })
        
    res = pd.concat([addr_df, pd.DataFrame(new_rows)], ignore_index=True)
    return res.sort_values('timestamp').reset_index(drop=True)

def run_single_seed_benchmark(
    df: pd.DataFrame,
    seed: int,
    config: Config,
    n_per_type: int = 20,
    w_if: float = 0.8,
    w_tf: float = 0.2
) -> Dict[str, Any]:
    """
    Evaluates Legacy IF, Fixed IF, Transformer, Mean-Rank, Max-Rank, 0.8/0.2 Mean,
    0.8/0.2 Window-5, 3-Signal (0.2/0.1/0.7), and Tuned Weighted Fusion.
    Uses SeedSequence to derive independent, isolated RNG seeds for:
      - data split (seed_split)
      - synthetic anomaly injection (seed_injection)
      - Isolation Forest models (seed_if)
      - Transformer training (seed_tf)
      - heuristic graph sanity testing (seed_graph)
    """
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.generate_state(5)
    seed_split = int(child_seeds[0]) % (2**31 - 1)
    seed_injection = int(child_seeds[1]) % (2**31 - 1)
    seed_if = int(child_seeds[2]) % (2**31 - 1)
    seed_tf = int(child_seeds[3]) % (2**31 - 1)
    seed_graph = int(child_seeds[4]) % (2**31 - 1)
    
    # 1. Deterministic Data Split
    rng_split = np.random.default_rng(seed_split)
    real_addresses = np.sort(df['address_id'].unique())
    rng_split.shuffle(real_addresses)
    
    split_idx = int(len(real_addresses) * 0.70)
    train_addrs = set(real_addresses[:split_idx])
    test_real_addrs = set(real_addresses[split_idx:])
    
    train_df = df[df['address_id'].isin(train_addrs)].copy()
    test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
    
    # 2. Synthetic Anomalies with dedicated injection seed
    synth_df, synth_types = generate_synthetic_anomalies(train_df, n_per_type=n_per_type, seed=seed_injection)
    synth_addrs = set(synth_types.keys())
    
    test_df = pd.concat([test_real_df, synth_df], ignore_index=True)
    all_test_addrs = sorted(list(test_real_addrs | synth_addrs))
    y_true = np.array([1 if a in synth_addrs else 0 for a in all_test_addrs])
    
    # 3. Legacy IF Pipeline
    legacy_cfg = Config(legacy_features=True, seed=seed_if, random_state=seed_if, contamination=config.contamination)
    train_legacy_feats, _ = extract_legacy_features(train_df, legacy_cfg)
    test_legacy_feats, _ = extract_legacy_features(test_df, legacy_cfg)
    
    num_train_leg = train_legacy_feats.select_dtypes(include='number')
    num_test_leg = test_legacy_feats.select_dtypes(include='number').reindex(columns=num_train_leg.columns).fillna(0.0)
    
    scaler_leg = StandardScaler()
    X_train_leg = scaler_leg.fit_transform(num_train_leg)
    X_test_leg = scaler_leg.transform(num_test_leg.loc[all_test_addrs])
    
    if_legacy = IsolationForest(contamination=config.contamination, random_state=seed_if)
    if_legacy.fit(X_train_leg)
    scores_legacy = -if_legacy.score_samples(X_test_leg)
    
    # 4. Fixed-Features IF Pipeline (Without Network Features by default)
    fixed_cfg = Config(
        include_network_features=False,
        seed=seed_if,
        random_state=seed_if,
        contamination=config.contamination,
        if_max_features=config.if_max_features
    )
    train_fixed_feats, G_train = extract_all_features(train_df, fixed_cfg)
    test_fixed_feats, G_test = extract_all_features(test_df, fixed_cfg)
    
    num_train_fixed = prepare_numeric_features(train_fixed_feats)
    num_test_fixed = prepare_numeric_features(test_fixed_feats).reindex(columns=num_train_fixed.columns).fillna(0.0)
    
    scaler_fixed = StandardScaler()
    X_train_fixed = scaler_fixed.fit_transform(num_train_fixed)
    X_test_fixed = scaler_fixed.transform(num_test_fixed.loc[all_test_addrs])
    
    if_fixed = IsolationForest(contamination=config.contamination, max_features=config.if_max_features, random_state=seed_if)
    if_fixed.fit(X_train_fixed)
    scores_fixed_if = -if_fixed.score_samples(X_test_fixed)
    
    # 4b. Fixed-Features IF WITH Network Features (for Negative Result Ablation)
    cfg_with_net = Config(
        include_network_features=True,
        seed=seed_if,
        random_state=seed_if,
        contamination=config.contamination,
        link_window_hours=config.link_window_hours,
        link_tolerance=config.link_tolerance
    )
    train_wnet_feats, _ = extract_all_features(train_df, cfg_with_net)
    test_wnet_feats, _ = extract_all_features(test_df, cfg_with_net)
    num_train_wnet = prepare_numeric_features(train_wnet_feats)
    num_test_wnet = prepare_numeric_features(test_wnet_feats).reindex(columns=num_train_wnet.columns).fillna(0.0)
    scaler_wnet = StandardScaler()
    X_train_wnet = scaler_wnet.fit_transform(num_train_wnet)
    X_test_wnet = scaler_wnet.transform(num_test_wnet.loc[all_test_addrs])
    if_wnet = IsolationForest(contamination=config.contamination, random_state=seed_if)
    if_wnet.fit(X_train_wnet)
    scores_fixed_if_with_net = -if_wnet.score_samples(X_test_wnet)
    
    # 5. Transformer Sequence Surprise Pipeline
    cfg_tf = Config(
        epochs=config.epochs,
        seed=seed_tf,
        random_state=seed_tf,
        contamination=config.contamination,
        min_seq_len=config.min_seq_len
    )
    tf_detector = SequenceAnomalyDetector(cfg_tf)
    tf_detector.fit(train_df)
    tf_scores_df, _, _ = tf_detector.score_addresses(test_df)
    
    # Primary surprise (window-5 max)
    scores_tf_win5 = []
    for a in all_test_addrs:
        if a in tf_scores_df.index and 'transformer_surprise_window_max' in tf_scores_df.columns:
            scores_tf_win5.append(float(tf_scores_df.loc[a, 'transformer_surprise_window_max']))
        elif a in tf_scores_df.index:
            scores_tf_win5.append(float(tf_scores_df.loc[a, 'transformer_surprise']))
        else:
            scores_tf_win5.append(0.0)
    scores_tf_win5 = np.array(scores_tf_win5)
    
    # Mean surprise
    scores_tf_mean = []
    for a in all_test_addrs:
        if a in tf_scores_df.index and 'transformer_surprise_mean' in tf_scores_df.columns:
            scores_tf_mean.append(float(tf_scores_df.loc[a, 'transformer_surprise_mean']))
        elif a in tf_scores_df.index:
            scores_tf_mean.append(float(tf_scores_df.loc[a, 'transformer_surprise']))
        else:
            scores_tf_mean.append(0.0)
    scores_tf_mean = np.array(scores_tf_mean)
    
    # Ranks for Fusion
    ranks_fixed = rankdata(scores_fixed_if) / len(scores_fixed_if)
    ranks_tf_win5 = rankdata(scores_tf_win5) / len(scores_tf_win5)
    ranks_tf_mean = rankdata(scores_tf_mean) / len(scores_tf_mean)
    
    # Ensembles
    scores_mean_rank = 0.5 * ranks_fixed + 0.5 * ranks_tf_mean
    scores_max_rank = np.maximum(ranks_fixed, ranks_tf_mean)
    scores_08_02_mean = 0.8 * ranks_fixed + 0.2 * ranks_tf_mean
    scores_08_02_win5 = 0.8 * ranks_fixed + 0.2 * ranks_tf_win5
    scores_3signal = 0.20 * ranks_fixed + 0.10 * ranks_tf_mean + 0.70 * ranks_tf_win5
    scores_tuned = w_if * ranks_fixed + w_tf * ranks_tf_win5
    tuned_label = f'Ensemble (Tuned {w_if:.1f}/{w_tf:.1f})'
    
    models = {
        'Legacy IF': scores_legacy,
        'Fixed IF': scores_fixed_if,
        'Fixed IF (with Network Features)': scores_fixed_if_with_net,
        'Transformer': scores_tf_win5,
        'Transformer (Mean)': scores_tf_mean,
        'Transformer (Window-5)': scores_tf_win5,
        'Ensemble (Mean-Rank)': scores_mean_rank,
        'Ensemble (Max-Rank)': scores_max_rank,
        'Ensemble (Current 0.8/0.2 Mean)': scores_08_02_mean,
        'Ensemble (0.8/0.2 Window-5)': scores_08_02_win5,
        'Ensemble (3-Signal 0.2/0.1/0.7)': scores_3signal,
        tuned_label: scores_tuned
    }
    
    results = {}
    k = len(synth_addrs)
    
    for name, s in models.items():
        auroc = float(roc_auc_score(y_true, s))
        ap = float(average_precision_score(y_true, s))
        
        top_k_indices = np.argsort(-s)[:k]
        top_k_addrs = set(all_test_addrs[i] for i in top_k_indices)
        
        overall_recall_at_k = len(top_k_addrs & synth_addrs) / max(k, 1)
        
        type_recalls = {}
        for a_type in ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]:
            addrs_of_type = [addr for addr, t in synth_types.items() if t == a_type]
            n_type = len(addrs_of_type)
            detected = len([a for a in addrs_of_type if a in top_k_addrs])
            type_recalls[a_type] = detected / max(n_type, 1)
            
        results[name] = {
            'auroc': auroc,
            'ap': ap,
            'recall_at_k': overall_recall_at_k,
            'type_recalls': type_recalls,
            'scores': s.tolist()
        }
        
    # Granular pass-through delay variants recall
    pt_delay_recalls = {}
    for d_tag in ['15m', '1h', '6h', '24h']:
        addrs_d = [a for a in synth_addrs if f"synth_passthrough_{d_tag}" in a]
        n_d = len(addrs_d)
        pt_delay_recalls[d_tag] = {}
        for m_name in ['Fixed IF', 'Transformer', tuned_label]:
            s = models[m_name]
            top_k_indices = np.argsort(-s)[:k]
            top_k_addrs = set(all_test_addrs[i] for i in top_k_indices)
            det = len([a for a in addrs_d if a in top_k_addrs])
            pt_delay_recalls[d_tag][m_name] = det / max(n_d, 1)
            
    return {
        'seed': seed,
        'metrics': results,
        'tuned_label': tuned_label,
        'pt_delay_recalls': pt_delay_recalls,
        'y_true': y_true.tolist(),
        'n_anomalies': len(synth_addrs),
        'n_normal': len(test_real_addrs)
    }

def tune_fusion_weights(
    df: pd.DataFrame,
    config: Config,
    tuning_seeds: List[int] = [101, 102, 103],
    n_per_type: int = 20,
    weight_grid: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Selects optimal fusion weights (w_if, w_tf) using separate tuning seeds (101, 102, 103)
    to prevent tuning on test benchmark seeds.
    """
    logger.info(f"Tuning fusion weights across seeds {tuning_seeds} over grid {weight_grid}...")
    grid_metrics = {w: {'auroc': [], 'ap': [], 'recall_at_k': []} for w in weight_grid}
    
    for s in tuning_seeds:
        cfg = Config(
            epochs=config.epochs,
            seed=s,
            random_state=s,
            contamination=config.contamination
        )
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
        
        # Fit Fixed IF
        train_feats, _ = extract_all_features(train_df, cfg)
        test_feats, _ = extract_all_features(test_df, cfg)
        num_train = prepare_numeric_features(train_feats)
        num_test = prepare_numeric_features(test_feats).reindex(columns=num_train.columns).fillna(0.0)
        scaler = StandardScaler().fit(num_train)
        if_model = IsolationForest(contamination=config.contamination, random_state=s).fit(scaler.transform(num_train))
        scores_if = -if_model.score_samples(scaler.transform(num_test.loc[all_test_addrs]))
        
        # Fit TF
        tf_detector = SequenceAnomalyDetector(cfg)
        tf_detector.fit(train_df)
        tf_scores_df, _, _ = tf_detector.score_addresses(test_df)
        scores_tf = np.array([float(tf_scores_df.loc[a, 'transformer_surprise']) if a in tf_scores_df.index else 0.0 for a in all_test_addrs])
        
        ranks_if = rankdata(scores_if) / len(scores_if)
        ranks_tf = rankdata(scores_tf) / len(scores_tf)
        k = len(synth_addrs)
        
        for w in weight_grid:
            w_if = round(w, 2)
            w_tf = round(1.0 - w, 2)
            fused = w_if * ranks_if + w_tf * ranks_tf
            auc = float(roc_auc_score(y_true, fused))
            ap = float(average_precision_score(y_true, fused))
            top_k_indices = np.argsort(-fused)[:k]
            top_k_addrs = set(all_test_addrs[i] for i in top_k_indices)
            rec = len(top_k_addrs & synth_addrs) / max(k, 1)
            
            grid_metrics[w]['auroc'].append(auc)
            grid_metrics[w]['ap'].append(ap)
            grid_metrics[w]['recall_at_k'].append(rec)
            
    best_w = 0.8
    best_score = -1.0
    tuning_summary = {}
    for w in weight_grid:
        mean_auc = float(np.mean(grid_metrics[w]['auroc']))
        mean_ap = float(np.mean(grid_metrics[w]['ap']))
        mean_rec = float(np.mean(grid_metrics[w]['recall_at_k']))
        tuning_summary[str(w)] = {'auroc': mean_auc, 'ap': mean_ap, 'recall_at_k': mean_rec}
        if mean_auc > best_score:
            best_score = mean_auc
            best_w = w
            
    best_w_if = round(best_w, 2)
    best_w_tf = round(1.0 - best_w_if, 2)
    logger.info(f"Selected optimal fusion weights from tuning seeds {tuning_seeds}: w_IF={best_w_if}, w_TF={best_w_tf} (tuning AUROC={best_score:.4f})")
    return best_w_if, best_w_tf, tuning_summary

def run_benchmark(
    df: pd.DataFrame,
    outdir: Path,
    config: Config,
    seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    tuning_seeds: List[int] = [101, 102, 103],
    n_per_type: int = 20
) -> Dict[str, Any]:
    """
    Executes benchmark across 10 held-out evaluation seeds (20 per archetype = 100 anomalies/seed),
    selects fusion weights on separate tuning seeds (101-103),
    computes 95% bootstrap confidence intervals for all cells, and runs graph sanity test.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run Graph Sanity Permutation Test
    sanity_results = run_graph_sanity_test(df, config, n_runs=20, seed=config.seed)
    
    # 2. Tune fusion weights on tuning seeds
    best_w_if, best_w_tf, tuning_summary = tune_fusion_weights(
        df, config, tuning_seeds=tuning_seeds, n_per_type=n_per_type
    )
    tuned_label = f'Ensemble (Tuned {best_w_if:.1f}/{best_w_tf:.1f})'
    
    logger.info(f"=== Starting Synthetic Anomaly Benchmark ({len(seeds)} held-out seeds, {n_per_type*5} anomalies/seed) ===")
    logger.warning("BENCHMARK CAVEAT: Synthetic anomalies are proxy models. Real addresses may contain true anomalies.")
    
    seed_runs = []
    for s in seeds:
        logger.info(f"Running benchmark seed {s}...")
        cfg = Config(
            epochs=config.epochs,
            seed=s,
            random_state=s,
            contamination=config.contamination,
            link_window_hours=config.link_window_hours,
            link_tolerance=config.link_tolerance
        )
        run_res = run_single_seed_benchmark(
            df, seed=s, config=cfg, n_per_type=n_per_type, w_if=best_w_if, w_tf=best_w_tf
        )
        seed_runs.append(run_res)
        
    # Aggregate statistics with 95% Bootstrap Confidence Intervals
    model_names = [
        'Legacy IF', 'Fixed IF', 'Transformer', 
        'Ensemble (Mean-Rank)', 'Ensemble (Max-Rank)',
        'Ensemble (Current 0.8/0.2 Mean)', tuned_label
    ]
    summary = {}
    archetypes = ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]
    
    for name in model_names:
        aurocs = [r['metrics'][name]['auroc'] for r in seed_runs]
        aps = [r['metrics'][name]['ap'] for r in seed_runs]
        recalls = [r['metrics'][name]['recall_at_k'] for r in seed_runs]
        
        type_agg = {}
        for t in archetypes:
            t_vals = [r['metrics'][name]['type_recalls'][t] for r in seed_runs]
            type_agg[t] = compute_bootstrap_ci(t_vals, n_boot=1000, ci=0.95, seed=42)
            
        summary[name] = {
            'auroc': compute_bootstrap_ci(aurocs, n_boot=1000, ci=0.95, seed=42),
            'ap': compute_bootstrap_ci(aps, n_boot=1000, ci=0.95, seed=42),
            'recall_at_k': compute_bootstrap_ci(recalls, n_boot=1000, ci=0.95, seed=42),
            'archetype_recalls': type_agg
        }
        
    # Paired Bootstrap Confidence Intervals of the Difference (Tuned Fusion vs Fixed IF)
    paired_diffs = {
        'auroc': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['auroc'] for r in seed_runs],
            [r['metrics']['Fixed IF']['auroc'] for r in seed_runs]
        ),
        'ap': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['ap'] for r in seed_runs],
            [r['metrics']['Fixed IF']['ap'] for r in seed_runs]
        ),
        'recall_at_k': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['recall_at_k'] for r in seed_runs],
            [r['metrics']['Fixed IF']['recall_at_k'] for r in seed_runs]
        )
    }
    
    # Paired Bootstrap Confidence Intervals of the Difference (Tuned Fusion vs Previous Fusion)
    paired_diffs_vs_prev = {
        'auroc': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['auroc'] for r in seed_runs],
            [r['metrics']['Ensemble (Current 0.8/0.2 Mean)']['auroc'] for r in seed_runs]
        ),
        'ap': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['ap'] for r in seed_runs],
            [r['metrics']['Ensemble (Current 0.8/0.2 Mean)']['ap'] for r in seed_runs]
        ),
        'recall_at_k': compute_paired_bootstrap_ci(
            [r['metrics'][tuned_label]['recall_at_k'] for r in seed_runs],
            [r['metrics']['Ensemble (Current 0.8/0.2 Mean)']['recall_at_k'] for r in seed_runs]
        )
    }
    
    # Pass-Through Delay Variants Summary (Transformer vs Fixed IF vs Tuned Ensemble)
    pt_delay_summary = {}
    for d_tag in ['15m', '1h', '6h', '24h']:
        pt_delay_summary[d_tag] = {}
        for m_name in ['Transformer', 'Fixed IF', tuned_label]:
            vals = [r['pt_delay_recalls'][d_tag][m_name] for r in seed_runs]
            pt_delay_summary[d_tag][m_name] = compute_bootstrap_ci(vals, n_boot=1000, ci=0.95, seed=42)
            
    # Network Features Ablation (With vs Without Network Features)
    net_ablation = {
        'fixed_if_without_net': {
            'auroc': compute_bootstrap_ci([r['metrics']['Fixed IF']['auroc'] for r in seed_runs]),
            'ap': compute_bootstrap_ci([r['metrics']['Fixed IF']['ap'] for r in seed_runs]),
            'recall_at_k': compute_bootstrap_ci([r['metrics']['Fixed IF']['recall_at_k'] for r in seed_runs])
        },
        'fixed_if_with_net': {
            'auroc': compute_bootstrap_ci([r['metrics']['Fixed IF (with Network Features)']['auroc'] for r in seed_runs]),
            'ap': compute_bootstrap_ci([r['metrics']['Fixed IF (with Network Features)']['ap'] for r in seed_runs]),
            'recall_at_k': compute_bootstrap_ci([r['metrics']['Fixed IF (with Network Features)']['recall_at_k'] for r in seed_runs])
        },
        'paired_diff_with_minus_without': {
            'auroc': compute_paired_bootstrap_ci(
                [r['metrics']['Fixed IF (with Network Features)']['auroc'] for r in seed_runs],
                [r['metrics']['Fixed IF']['auroc'] for r in seed_runs]
            ),
            'ap': compute_paired_bootstrap_ci(
                [r['metrics']['Fixed IF (with Network Features)']['ap'] for r in seed_runs],
                [r['metrics']['Fixed IF']['ap'] for r in seed_runs]
            ),
            'recall_at_k': compute_paired_bootstrap_ci(
                [r['metrics']['Fixed IF (with Network Features)']['recall_at_k'] for r in seed_runs],
                [r['metrics']['Fixed IF']['recall_at_k'] for r in seed_runs]
            )
        }
    }
        
    # Generate Plots from representative Seed 42
    rep_run = seed_runs[0]
    y_true_rep = np.array(rep_run['y_true'])
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(15, 6))
    colors = {
        'Legacy IF': '#e74c3c',
        'Fixed IF': '#3498db',
        'Transformer': '#2ecc71',
        'Ensemble (Mean-Rank)': '#9b59b6',
        'Ensemble (Max-Rank)': '#f39c12',
        tuned_label: '#1abc9c'
    }
    
    for name in model_names:
        s = np.array(rep_run['metrics'][name]['scores'])
        fpr, tpr, _ = roc_curve(y_true_rep, s)
        prec, rec, _ = precision_recall_curve(y_true_rep, s)
        
        m_auroc = summary[name]['auroc']['mean']
        m_ap = summary[name]['ap']['mean']
        
        ax_roc.plot(fpr, tpr, label=f"{name} ({m_auroc:.3f})", color=colors.get(name, '#1abc9c'), lw=2)
        ax_pr.plot(rec, prec, label=f"{name} ({m_ap:.3f})", color=colors.get(name, '#1abc9c'), lw=2)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax_roc.set_title("ROC Curves (10 Seeds, 100 Anomalies/Seed, Gen v2.0)", fontsize=12, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc='lower right', frameon=True, fontsize=9)
    ax_roc.grid(True, alpha=0.3)
    
    ax_pr.set_title("Precision-Recall Curves", fontsize=12, fontweight='bold')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc='lower left', frameon=True, fontsize=9)
    ax_pr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = outdir / "benchmark_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved benchmark curve plots to {plot_path}")
    
    full_output = {
        'generator_version': 'v2.0',
        'seeds': seeds,
        'tuning_seeds': tuning_seeds,
        'tuning_summary': tuning_summary,
        'tuned_weights': {'w_if': best_w_if, 'w_tf': best_w_tf},
        'tuned_label': tuned_label,
        'n_anomalies_per_seed': n_per_type * 5,
        'summary': summary,
        'paired_differences_tuned_vs_fixed_if': paired_diffs,
        'paired_differences_tuned_vs_previous_fusion': paired_diffs_vs_prev,
        'pass_through_delay_summary': pt_delay_summary,
        'network_ablation': net_ablation,
        'graph_sanity': sanity_results,
        'caveat': "Real addresses may contain true anomalies; benchmark results against synthetic archetypes are relative.",
        'plot_file': str(plot_path.name)
    }
    
    json_path = outdir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved benchmark results JSON to {json_path}")
    
    print_benchmark_markdown_table(summary, paired_diffs, pt_delay_summary, net_ablation, paired_diffs_vs_prev, tuned_label)
    return full_output

def print_benchmark_markdown_table(
    summary: Dict[str, Any],
    paired_diffs: Optional[Dict[str, Any]] = None,
    pt_delays: Optional[Dict[str, Any]] = None,
    net_ablation: Optional[Dict[str, Any]] = None,
    paired_diffs_vs_prev: Optional[Dict[str, Any]] = None,
    tuned_label: str = 'Ensemble (Tuned)'
) -> str:
    lines = []
    lines.append("\n### Benchmark Results (10 Seeds, 20/archetype, 95% Bootstrap CI, Generator v2.0)\n")
    lines.append("| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, m in summary.items():
        auroc_s = f"{m['auroc']['mean']:.3f} [{m['auroc']['ci_lower']:.3f}, {m['auroc']['ci_upper']:.3f}]"
        ap_s = f"{m['ap']['mean']:.3f} [{m['ap']['ci_lower']:.3f}, {m['ap']['ci_upper']:.3f}]"
        rec_s = f"{m['recall_at_k']['mean']:.3f} [{m['recall_at_k']['ci_lower']:.3f}, {m['recall_at_k']['ci_upper']:.3f}]"
        
        arch = m['archetype_recalls']
        st_s = f"{arch['structuring']['mean']:.2f} [{arch['structuring']['ci_lower']:.2f}, {arch['structuring']['ci_upper']:.2f}]"
        bu_s = f"{arch['burst']['mean']:.2f} [{arch['burst']['ci_lower']:.2f}, {arch['burst']['ci_upper']:.2f}]"
        pt_s = f"{arch['pass_through']['mean']:.2f} [{arch['pass_through']['ci_lower']:.2f}, {arch['pass_through']['ci_upper']:.2f}]"
        bp_s = f"{arch['bot_periodic']['mean']:.2f} [{arch['bot_periodic']['ci_lower']:.2f}, {arch['bot_periodic']['ci_upper']:.2f}]"
        ds_s = f"{arch['dormant_spike']['mean']:.2f} [{arch['dormant_spike']['ci_lower']:.2f}, {arch['dormant_spike']['ci_upper']:.2f}]"
        
        lines.append(f"| **{name}** | {auroc_s} | {ap_s} | {rec_s} | {st_s} | {bu_s} | {pt_s} | {bp_s} | {ds_s} |")
        
    if paired_diffs:
        lines.append(f"\n### Paired Bootstrap CIs of the Difference: {tuned_label} vs Fixed IF\n")
        lines.append("| Metric | Paired Mean Difference | 95% Bootstrap CI | Interpretation |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for metric, label in [('auroc', 'AUROC'), ('ap', 'Average Precision (AP)'), ('recall_at_k', 'Recall@K')]:
            d = paired_diffs[metric]
            interp = "95% bootstrap CI excludes zero" if d['ci_lower'] > 0 else ("Borderline (touches zero)" if d['ci_lower'] == 0 or d['ci_upper'] == 0 else "Overlap with zero")
            lines.append(f"| **Δ {label}** | {d['mean']:+.4f} | [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}] | {interp} |")
            
    if paired_diffs_vs_prev:
        lines.append(f"\n### Paired Bootstrap CIs of the Difference: {tuned_label} vs Previous Ensemble (0.8/0.2 Mean)\n")
        lines.append("| Metric | Paired Mean Difference | 95% Bootstrap CI | Interpretation |")
        lines.append("| :--- | :---: | :---: | :--- |")
        for metric, label in [('auroc', 'AUROC'), ('ap', 'Average Precision (AP)'), ('recall_at_k', 'Recall@K')]:
            d = paired_diffs_vs_prev[metric]
            interp = "95% bootstrap CI excludes zero" if d['ci_lower'] > 0 else ("Borderline (touches zero)" if d['ci_lower'] == 0 or d['ci_upper'] == 0 else "Overlap with zero")
            lines.append(f"| **Δ {label}** | {d['mean']:+.4f} | [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}] | {interp} |")
                         
    if pt_delays:
        lines.append("\n### Pass-Through Recall by Delay Variant (Transformer vs Fixed IF vs Tuned Fusion)\n")
        lines.append("| Delay Variant | Transformer Recall [95% CI] | Fixed IF Recall [95% CI] | Tuned Ensemble Recall [95% CI] |")
        lines.append("| :--- | :---: | :---: | :---: |")
        for d_tag, d_label in [('15m', '15 minutes (2-15m)'), ('1h', '1 hour (50-70m)'), ('6h', '6 hours (5-7h)'), ('24h', '24 hours (22-26h)')]:
            tf_v = pt_delays[d_tag]['Transformer']
            if_v = pt_delays[d_tag]['Fixed IF']
            tuned_key = [k for k in pt_delays[d_tag] if 'Tuned' in k][0]
            en_v = pt_delays[d_tag][tuned_key]
            tf_s = f"{tf_v['mean']:.2f} [{tf_v['ci_lower']:.2f}, {tf_v['ci_upper']:.2f}]"
            if_s = f"{if_v['mean']:.2f} [{if_v['ci_lower']:.2f}, {if_v['ci_upper']:.2f}]"
            en_s = f"{en_v['mean']:.2f} [{en_v['ci_lower']:.2f}, {en_v['ci_upper']:.2f}]"
            lines.append(f"| **{d_label}** | {tf_s} | {if_s} | {en_s} |")
            
    if net_ablation:
        lines.append("\n### Network Feature Ablation: Fixed IF Without vs With Heuristic Flow Graph\n")
        lines.append("| Configuration | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] |")
        lines.append("| :--- | :---: | :---: | :---: |")
        w_out = net_ablation['fixed_if_without_net']
        w_in = net_ablation['fixed_if_with_net']
        diff = net_ablation['paired_diff_with_minus_without']
        lines.append(f"| **Without Network Features (Default)** | {w_out['auroc']['mean']:.3f} [{w_out['auroc']['ci_lower']:.3f}, {w_out['auroc']['ci_upper']:.3f}] | "
                     f"{w_out['ap']['mean']:.3f} [{w_out['ap']['ci_lower']:.3f}, {w_out['ap']['ci_upper']:.3f}] | "
                     f"{w_out['recall_at_k']['mean']:.3f} [{w_out['recall_at_k']['ci_lower']:.3f}, {w_out['recall_at_k']['ci_upper']:.3f}] |")
        lines.append(f"| **With Network Features (Ablation)** | {w_in['auroc']['mean']:.3f} [{w_in['auroc']['ci_lower']:.3f}, {w_in['auroc']['ci_upper']:.3f}] | "
                     f"{w_in['ap']['mean']:.3f} [{w_in['ap']['ci_lower']:.3f}, {w_in['ap']['ci_upper']:.3f}] | "
                     f"{w_in['recall_at_k']['mean']:.3f} [{w_in['recall_at_k']['ci_lower']:.3f}, {w_in['recall_at_k']['ci_upper']:.3f}] |")
        lines.append(f"| **Paired Difference (With - Without)** | {diff['auroc']['mean']:+.4f} [{diff['auroc']['ci_lower']:+.4f}, {diff['auroc']['ci_upper']:+.4f}] | "
                     f"{diff['ap']['mean']:+.4f} [{diff['ap']['ci_lower']:+.4f}, {diff['ap']['ci_upper']:+.4f}] | "
                     f"{diff['recall_at_k']['mean']:+.4f} [{diff['recall_at_k']['ci_lower']:+.4f}, {diff['recall_at_k']['ci_upper']:+.4f}] |")
                     
    table_str = "\n".join(lines)
    logger.info(table_str)
    return table_str
