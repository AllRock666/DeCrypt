import pytest
import numpy as np
import pandas as pd
import torch
from decrypt.config import Config
from decrypt.sequence import (
    EventTransformerModel,
    compute_bin_edges,
    extract_address_events,
    SequenceAnomalyDetector,
    compute_ensemble_scores
)

def test_bin_edges_and_events():
    df = pd.DataFrame({
        'address_id': ['A', 'A', 'A', 'B', 'B'],
        'hash': ['t1', 't2', 't3', 't4', 't5'],
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-01 12:00:00',
            '2023-01-01 13:00:00',
            '2023-01-01 14:00:00'
        ]),
        'type': [1, -1, 1, 1, -1],
        'amount': [10.0, -10.0, 50.0, 100.0, -50.0]
    })
    
    amt_edges, dt_edges = compute_bin_edges(df)
    assert len(amt_edges) == 31
    assert len(dt_edges) == 31
    
    events = extract_address_events(df, amt_edges, dt_edges)
    assert 'A' in events
    assert 'B' in events
    assert events['A']['length'] == 3
    assert events['B']['length'] == 2
    assert len(events['A']['direction']) == 3
    assert len(events['A']['amt_bin']) == 3

def test_model_forward_shapes():
    model = EventTransformerModel(d_model=64, n_heads=4, num_layers=2)
    B, L = 2, 10
    
    dir_t = torch.randint(0, 3, (B, L))
    amt_b_t = torch.randint(0, 33, (B, L))
    dt_b_t = torch.randint(0, 33, (B, L))
    hour_t = torch.randint(0, 25, (B, L))
    dow_t = torch.randint(0, 8, (B, L))
    amt_c_t = torch.randn(B, L)
    dt_c_t = torch.randn(B, L)
    
    hidden, l_dir, l_amt, l_dt = model(
        dir_t, amt_b_t, dt_b_t, hour_t, dow_t, amt_c_t, dt_c_t
    )
    
    assert hidden.shape == (B, L, 64)
    assert l_dir.shape == (B, L, 2)
    assert l_amt.shape == (B, L, 32)
    assert l_dt.shape == (B, L, 32)

def test_ensemble_scoring():
    config = Config(if_weight=0.5, transformer_weight=0.5)
    if_scores = pd.Series({'A': 80.0, 'B': 40.0, 'C': 10.0})
    tf_df = pd.DataFrame({
        'transformer_surprise': [5.0, 2.0, 0.0],
        'insufficient_history': [False, False, True],
        'seq_length': [10, 5, 1]
    }, index=['A', 'B', 'C'])
    
    res = compute_ensemble_scores(if_scores, tf_df, config)
    assert 'ensemble_risk_score' in res.columns
    # C has insufficient history, so its score is purely from IF
    assert res.loc['C', 'insufficient_history'] == True
    assert len(res) == 3
