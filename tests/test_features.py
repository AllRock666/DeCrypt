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
