import pytest
import pandas as pd
import numpy as np
from decrypt.config import Config
from decrypt.benchmark import (
    generate_synthetic_anomalies,
    run_single_seed_benchmark,
    compute_hierarchical_paired_ci,
    compute_hierarchical_bootstrap_ci
)

def test_synthetic_anomaly_generation():
    timestamps = pd.date_range('2023-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'address_id': [f'addr_{i%10}' for i in range(100)],
        'hash': [f'hash_{i}' for i in range(100)],
        'timestamp': timestamps,
        'type': [1 if i % 2 == 0 else -1 for i in range(100)],
        'amount': [50.0 if i % 2 == 0 else -50.0 for i in range(100)]
    })
    
    synth_df, synth_types = generate_synthetic_anomalies(df, n_per_type=2, seed=42)
    assert len(synth_types) == 10  # 5 types * 2
    assert set(synth_types.values()) == {"structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"}
    assert not synth_df.empty
    assert 'address_id' in synth_df.columns
    assert 'amount' in synth_df.columns

def test_hierarchical_paired_bootstrap_nonzero_width():
    """
    Verifies that hierarchical bootstrap produces nonzero CI width even when
    per-seed summary means are identical (e.g. 1/17 across all 10 seeds).
    """
    # 10 seeds, each with 17 address outcomes (1 positive diff, 16 zero diffs)
    outcomes_a = {s: [1] * 15 + [0] * 2 for s in range(10)}
    outcomes_b = {s: [1] * 14 + [0] * 3 for s in range(10)}
    
    ci_res = compute_hierarchical_paired_ci(outcomes_a, outcomes_b, n_boot=500, ci=0.95, seed=42)
    assert ci_res['ci_upper'] > ci_res['ci_lower'], f"CI width should be non-zero, got {ci_res}"
    assert ci_res['std'] > 0.0
    assert ci_res['ci_lower'] <= ci_res['mean'] <= ci_res['ci_upper']
    assert abs(ci_res['mean'] - (1.0 / 17.0)) < 1e-4

    # Also test individual hierarchical bootstrap
    ind_ci = compute_hierarchical_bootstrap_ci(outcomes_b, n_boot=500, ci=0.95, seed=42)
    assert ind_ci['ci_upper'] > ind_ci['ci_lower']
    assert ind_ci['std'] > 0.0
    assert ind_ci['ci_lower'] <= ind_ci['mean'] <= ind_ci['ci_upper']

def test_benchmark_reproducibility_same_seed():
    """
    Verifies that two runs of run_single_seed_benchmark with the same master seed
    produce 100% identical benchmark metrics across all models due to isolated RNGs.
    """
    timestamps = pd.date_range('2023-01-01', periods=200, freq='h')
    df = pd.DataFrame({
        'address_id': [f'addr_{i%20}' for i in range(200)],
        'hash': [f'hash_{i}' for i in range(200)],
        'timestamp': timestamps,
        'type': [1 if i % 2 == 0 else -1 for i in range(200)],
        'amount': [float((i % 10) + 1) * 10.0 if i % 2 == 0 else -float((i % 10) + 1) * 10.0 for i in range(200)]
    })
    
    cfg = Config(epochs=1, seed=42, random_state=42)
    res1 = run_single_seed_benchmark(df, seed=42, config=cfg, n_per_type=2)
    res2 = run_single_seed_benchmark(df, seed=42, config=cfg, n_per_type=2)
    
    for m in res1['metrics']:
        auc1 = res1['metrics'][m]['auroc']
        auc2 = res2['metrics'][m]['auroc']
        ap1 = res1['metrics'][m]['ap']
        ap2 = res2['metrics'][m]['ap']
        rec1 = res1['metrics'][m]['recall_at_k']
        rec2 = res2['metrics'][m]['recall_at_k']
        assert auc1 == pytest.approx(auc2, abs=1e-9), f"AUROC mismatch in model {m}: {auc1} vs {auc2}"
        assert ap1 == pytest.approx(ap2, abs=1e-9), f"AP mismatch in model {m}: {ap1} vs {ap2}"
        assert rec1 == pytest.approx(rec2, abs=1e-9), f"Recall@K mismatch in model {m}: {rec1} vs {rec2}"
