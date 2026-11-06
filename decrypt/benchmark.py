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
from .features import extract_all_features, extract_legacy_features, prepare_numeric_features
from .sequence import SequenceAnomalyDetector, compute_ensemble_scores

logger = setup_logger(__name__, default_config)

def generate_synthetic_anomalies(
    real_df: pd.DataFrame,
    n_per_type: int = 10,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Generates synthetic anomalies based on empirical data distributions.
    Returns:
        anomaly_df: DataFrame of synthetic transactions
        anomaly_types: Dict mapping synthetic address_id to anomaly archetype
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
    for i in range(n_per_type):
        addr = f"synth_structuring_{i:02d}"
        anomaly_types[addr] = "structuring"
        threshold = rng.choice([20.0, 50.0, 100.0, 500.0])
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

    # Archetype C: Pass-through / Round-trip Mixer
    # Rapid deposit X followed shortly by withdrawal ~X
    for i in range(n_per_type):
        addr = f"synth_passthrough_{i:02d}"
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
            # Rapid withdrawal minus fee
            cur_time += pd.Timedelta(minutes=float(rng.uniform(2, 15)))
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

def run_single_seed_benchmark(
    df: pd.DataFrame,
    seed: int,
    config: Config,
    n_per_type: int = 10
) -> Dict[str, Any]:
    """
    Evaluates Legacy IF, Fixed IF, Transformer, and Ensemble on 1 seed split.
    """
    rng = np.random.default_rng(seed)
    real_addresses = np.array(list(df['address_id'].unique()))
    rng.shuffle(real_addresses)
    
    split_idx = int(len(real_addresses) * 0.70)
    train_addrs = set(real_addresses[:split_idx])
    test_real_addrs = set(real_addresses[split_idx:])
    
    train_df = df[df['address_id'].isin(train_addrs)].copy()
    test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
    
    # Generate synthetic anomalies
    synth_df, synth_types = generate_synthetic_anomalies(train_df, n_per_type=n_per_type, seed=seed)
    synth_addrs = set(synth_types.keys())
    
    # Combine test set: real held-out (0) + synthetic anomalies (1)
    test_df = pd.concat([test_real_df, synth_df], ignore_index=True)
    all_test_addrs = list(test_real_addrs | synth_addrs)
    
    y_true = np.array([1 if a in synth_addrs else 0 for a in all_test_addrs])
    addr_to_idx = {a: i for i, a in enumerate(all_test_addrs)}
    
    # 1. Legacy IF Pipeline
    legacy_cfg = Config(legacy_features=True, seed=seed, contamination=config.contamination)
    train_legacy_feats, _ = extract_legacy_features(train_df, legacy_cfg)
    test_legacy_feats, _ = extract_legacy_features(test_df, legacy_cfg)
    
    num_train_leg = train_legacy_feats.select_dtypes(include='number')
    num_test_leg = test_legacy_feats.select_dtypes(include='number').reindex(columns=num_train_leg.columns).fillna(0.0)
    
    scaler_leg = StandardScaler()
    X_train_leg = scaler_leg.fit_transform(num_train_leg)
    X_test_leg = scaler_leg.transform(num_test_leg.loc[all_test_addrs])
    
    if_legacy = IsolationForest(contamination=config.contamination, random_state=seed)
    if_legacy.fit(X_train_leg)
    scores_legacy = -if_legacy.score_samples(X_test_leg)
    
    # 2. Fixed-Features IF Pipeline
    train_fixed_feats, G_train = extract_all_features(train_df, config)
    test_fixed_feats, G_test = extract_all_features(test_df, config)
    
    num_train_fixed = prepare_numeric_features(train_fixed_feats)
    num_test_fixed = prepare_numeric_features(test_fixed_feats).reindex(columns=num_train_fixed.columns).fillna(0.0)
    
    scaler_fixed = StandardScaler()
    X_train_fixed = scaler_fixed.fit_transform(num_train_fixed)
    X_test_fixed = scaler_fixed.transform(num_test_fixed.loc[all_test_addrs])
    
    if_fixed = IsolationForest(contamination=config.contamination, random_state=seed)
    if_fixed.fit(X_train_fixed)
    scores_fixed_if = -if_fixed.score_samples(X_test_fixed)
    
    # 3. Transformer Sequence Surprise Pipeline
    tf_detector = SequenceAnomalyDetector(config)
    tf_detector.fit(train_df)
    tf_scores_df, _, _ = tf_detector.score_addresses(test_df)
    
    scores_tf = []
    for a in all_test_addrs:
        if a in tf_scores_df.index:
            scores_tf.append(float(tf_scores_df.loc[a, 'transformer_surprise']))
        else:
            scores_tf.append(0.0)
    scores_tf = np.array(scores_tf)
    
    # 4. Ensemble
    ranks_fixed = rankdata(scores_fixed_if) / len(scores_fixed_if)
    ranks_tf = rankdata(scores_tf) / len(scores_tf)
    scores_ensemble = 0.5 * ranks_fixed + 0.5 * ranks_tf
    
    models = {
        'Legacy IF': scores_legacy,
        'Fixed IF': scores_fixed_if,
        'Transformer': scores_tf,
        'Ensemble': scores_ensemble
    }
    
    results = {}
    k = len(synth_addrs)  # Recall@K where K = total synthetic anomalies
    
    for name, s in models.items():
        auroc = float(roc_auc_score(y_true, s))
        ap = float(average_precision_score(y_true, s))
        
        # Top-k indices
        top_k_indices = np.argsort(-s)[:k]
        top_k_addrs = set(all_test_addrs[i] for i in top_k_indices)
        
        overall_recall_at_k = len(top_k_addrs & synth_addrs) / max(k, 1)
        
        # Recall per anomaly archetype
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
        
    return {
        'seed': seed,
        'metrics': results,
        'y_true': y_true.tolist(),
        'n_anomalies': len(synth_addrs),
        'n_normal': len(test_real_addrs)
    }

def run_benchmark(
    df: pd.DataFrame,
    outdir: Path,
    config: Config,
    seeds: List[int] = [42, 43, 44, 45, 46],
    n_per_type: int = 10
) -> Dict[str, Any]:
    """
    Executes benchmark across 5 seeds, computes mean +/- std, generates plots and JSON.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=== Starting Synthetic Anomaly Benchmark ({len(seeds)} seeds, {n_per_type*5} anomalies/seed) ===")
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
        run_res = run_single_seed_benchmark(df, seed=s, config=cfg, n_per_type=n_per_type)
        seed_runs.append(run_res)
        
    # Aggregate statistics
    model_names = ['Legacy IF', 'Fixed IF', 'Transformer', 'Ensemble']
    summary = {}
    archetypes = ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]
    
    for name in model_names:
        aurocs = [r['metrics'][name]['auroc'] for r in seed_runs]
        aps = [r['metrics'][name]['ap'] for r in seed_runs]
        recalls = [r['metrics'][name]['recall_at_k'] for r in seed_runs]
        
        type_agg = {}
        for t in archetypes:
            t_vals = [r['metrics'][name]['type_recalls'][t] for r in seed_runs]
            type_agg[t] = {
                'mean': float(np.mean(t_vals)),
                'std': float(np.std(t_vals))
            }
            
        summary[name] = {
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_std': float(np.std(aurocs)),
            'ap_mean': float(np.mean(aps)),
            'ap_std': float(np.std(aps)),
            'recall_at_k_mean': float(np.mean(recalls)),
            'recall_at_k_std': float(np.std(recalls)),
            'archetype_recalls': type_agg
        }
        
    # Generate Plots from representative Seed 42
    rep_run = seed_runs[0]
    y_true_rep = np.array(rep_run['y_true'])
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'Legacy IF': '#e74c3c', 'Fixed IF': '#3498db', 'Transformer': '#2ecc71', 'Ensemble': '#9b59b6'}
    
    for name in model_names:
        s = np.array(rep_run['metrics'][name]['scores'])
        fpr, tpr, _ = roc_curve(y_true_rep, s)
        prec, rec, _ = precision_recall_curve(y_true_rep, s)
        
        m_auroc = summary[name]['auroc_mean']
        m_ap = summary[name]['ap_mean']
        
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={m_auroc:.3f})", color=colors[name], lw=2)
        ax_pr.plot(rec, prec, label=f"{name} (AP={m_ap:.3f})", color=colors[name], lw=2)
        
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax_roc.set_title("ROC Curves (Synthetic Anomaly Benchmark)", fontsize=13, fontweight='bold')
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate", fontsize=11)
    ax_roc.legend(loc='lower right', frameon=True)
    ax_roc.grid(True, alpha=0.3)
    
    ax_pr.set_title("Precision-Recall Curves", fontsize=13, fontweight='bold')
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.legend(loc='lower left', frameon=True)
    ax_pr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = outdir / "benchmark_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved benchmark curve plots to {plot_path}")
    
    # Format results output
    full_output = {
        'seeds': seeds,
        'summary': summary,
        'caveat': "Real addresses may contain true anomalies; benchmark results against synthetic archetypes are relative.",
        'plot_file': str(plot_path.name)
    }
    
    json_path = outdir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Saved benchmark results JSON to {json_path}")
    
    # Print formatted markdown table
    print_benchmark_markdown_table(summary)
    return full_output

def print_benchmark_markdown_table(summary: Dict[str, Any]) -> str:
    lines = []
    lines.append("\n### Benchmark Results (5 Seeds, Mean +/- Std)\n")
    lines.append("| Model | AUROC | Average Precision (AP) | Recall@K (Overall) | Structuring | Burst | Pass-Through | Bot-Periodic | Dormant-Spike |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, m in summary.items():
        auroc_s = f"{m['auroc_mean']:.3f} +/- {m['auroc_std']:.3f}"
        ap_s = f"{m['ap_mean']:.3f} +/- {m['ap_std']:.3f}"
        rec_s = f"{m['recall_at_k_mean']:.3f} +/- {m['recall_at_k_std']:.3f}"
        
        arch = m['archetype_recalls']
        st_s = f"{arch['structuring']['mean']:.2f}"
        bu_s = f"{arch['burst']['mean']:.2f}"
        pt_s = f"{arch['pass_through']['mean']:.2f}"
        bp_s = f"{arch['bot_periodic']['mean']:.2f}"
        ds_s = f"{arch['dormant_spike']['mean']:.2f}"
        
        lines.append(f"| **{name}** | {auroc_s} | {ap_s} | {rec_s} | {st_s} | {bu_s} | {pt_s} | {bp_s} | {ds_s} |")
    table_str = "\n".join(lines)
    logger.info(table_str)
    return table_str
