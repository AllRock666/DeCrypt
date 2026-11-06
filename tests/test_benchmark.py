import pytest
import pandas as pd
from decrypt.config import Config
from decrypt.benchmark import generate_synthetic_anomalies, run_single_seed_benchmark

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
