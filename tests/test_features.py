import pytest
import pandas as pd
import numpy as np
from decrypt.config import Config
from decrypt.features import (
    extract_behavioral_features,
    extract_legacy_features,
    build_heuristic_flow_graph,
    extract_network_features,
    prepare_numeric_features
)

def test_feature_volume_and_net_flow():
    """
    Test that deposit 100 and withdraw 100 yields total_volume=200, net_flow=0.
    """
    config = Config()
    timestamps = pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 12:00:00'])
    df = pd.DataFrame({
        'address_id': ['addr1', 'addr1'],
        'hash': ['tx1', 'tx2'],
        'timestamp': timestamps,
        'type': [1, -1],
        'amount': [100.0, -100.0]
    })
    
    # New features
    features = extract_behavioral_features(df, config)
    assert 'addr1' in features.index
    row = features.loc['addr1']
    assert row['total_volume'] == 200.0
    assert row['net_flow'] == 0.0
    assert row['total_deposited'] == 100.0
    assert row['total_withdrawn'] == 100.0
    assert row['n_deposits'] == 1
    assert row['n_withdrawals'] == 1
    assert row['pass_through_ratio'] == 1.0

def test_legacy_features_reproduce_flaw():
    """
    Verify that legacy features produce the flawed net volume (0.0).
    """
    config = Config(legacy_features=True)
    timestamps = pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 12:00:00'])
    df = pd.DataFrame({
        'address_id': ['addr1', 'addr1'],
        'hash': ['tx1', 'tx2'],
        'timestamp': timestamps,
        'type': [1, -1],
        'amount': [100.0, -100.0]
    })
    
    legacy_feats, _ = extract_legacy_features(df, config)
    assert legacy_feats.loc['addr1', 'total_volume'] == 0.0

def test_heuristic_flow_graph():
    """
    Test heuristic flow graph edge creation within link_window_hours and tolerance.
    """
    config = Config(link_window_hours=24.0, link_tolerance=0.02, min_edges=1)
    base_time = pd.to_datetime('2023-01-01 10:00:00')
    
    df = pd.DataFrame({
        'address_id': ['A', 'B', 'C', 'D'],
        'hash': ['tx1', 'tx2', 'tx3', 'tx4'],
        'timestamp': [
            base_time,                                # A: withdrawal 50
            base_time + pd.Timedelta(hours=2),        # B: deposit 50 (valid match)
            base_time + pd.Timedelta(hours=2),        # C: deposit 100 (mismatch amt)
            base_time + pd.Timedelta(hours=48)        # D: deposit 50 (outside window)
        ],
        'type': [-1, 1, 1, 1],
        'amount': [-50.0, 50.0, 100.0, 50.0]
    })
    
    G = build_heuristic_flow_graph(df, config)
    assert G.has_edge('A', 'B')
    assert not G.has_edge('A', 'C')
    assert not G.has_edge('A', 'D')
    assert G['A']['B']['weight'] == 50.0

def test_prepare_numeric_features():
    df = pd.DataFrame({
        'num1': [1.0, 2.0, 3.0],
        'zero_var': [5.0, 5.0, 5.0],
        'str_col': ['a', 'b', 'c']
    })
    clean = prepare_numeric_features(df)
    assert 'num1' in clean.columns
    assert 'zero_var' not in clean.columns
    assert 'str_col' not in clean.columns

def test_direction_alternation_features():
    """
    Test direction_switch_count, alternation_rate, and max_alternating_run_length.
    """
    config = Config()
    timestamps = pd.date_range('2023-01-01', periods=5, freq='1h')
    # addr1: [1, -1, 1, -1, 1] -> 4 switches, rate 1.0, run length 5
    # addr2: [1, 1, -1, 1, 1]   -> switches at idx 2 (-1) and idx 3 (1): 2 switches, rate 2/4 = 0.5, max run 3
    # addr3: [1, 1, 1, 1, 1]   -> 0 switches, rate 0.0, max run 0
    df = pd.DataFrame({
        'address_id': ['addr1']*5 + ['addr2']*5 + ['addr3']*5,
        'hash': [f'tx_{i}' for i in range(15)],
        'timestamp': list(timestamps)*3,
        'type': [1, -1, 1, -1, 1] + [1, 1, -1, 1, 1] + [1, 1, 1, 1, 1],
        'amount': [10.0, -10.0, 10.0, -10.0, 10.0] + [10.0, 10.0, -10.0, 10.0, 10.0] + [10.0]*5
    })
    feats = extract_behavioral_features(df, config)
    
    assert feats.loc['addr1', 'direction_switch_count'] == 4
    assert pytest.approx(feats.loc['addr1', 'alternation_rate']) == 1.0
    assert feats.loc['addr1', 'max_alternating_run_length'] == 5
    
    assert feats.loc['addr2', 'direction_switch_count'] == 2
    assert pytest.approx(feats.loc['addr2', 'alternation_rate']) == 0.5
    assert feats.loc['addr2', 'max_alternating_run_length'] == 3
    
    assert feats.loc['addr3', 'direction_switch_count'] == 0
    assert feats.loc['addr3', 'alternation_rate'] == 0.0
    assert feats.loc['addr3', 'max_alternating_run_length'] == 0
