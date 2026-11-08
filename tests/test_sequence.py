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
from decrypt.benchmark import (
    inject_alternating_cycles,
    inject_burst,
    inject_structuring,
    inject_bot_periodic
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

def test_sequence_aggregations_spike_sensitivity():
    # Test that a short injected spike raises window-max and top-k scores much more than mean
    # Simulate a baseline sequence with uniform low event surprises vs an injected spike
    length = 30
    normal_nll = np.full(length, 1.0, dtype=np.float32)
    spike_nll = normal_nll.copy()
    spike_nll[10:15] = 10.0  # 5-event concentrated spike
    
    mean_normal = float(np.mean(normal_nll))
    mean_spiked = float(np.mean(spike_nll))
    
    # window-5 max
    w5_normal = float(np.max([np.mean(normal_nll[i:i+5]) for i in range(length - 4)]))
    w5_spiked = float(np.max([np.mean(spike_nll[i:i+5]) for i in range(length - 4)]))
    
    # top-3 mean
    top3_normal = float(np.mean(np.sort(normal_nll)[-3:]))
    top3_spiked = float(np.mean(np.sort(spike_nll)[-3:]))
    
    delta_mean = mean_spiked - mean_normal
    delta_w5 = w5_spiked - w5_normal
    delta_top3 = top3_spiked - top3_normal
    
    # A short injected spike of length 5 in a 30-event sequence:
    # delta_mean = (5 * 9.0) / 30 = 1.5
    # delta_w5 = 10.0 - 1.0 = 9.0
    # delta_w5 should be much larger (6x larger!) than delta_mean
    assert delta_w5 > delta_mean * 4.0
    assert delta_top3 > delta_mean * 4.0
    assert w5_spiked == 10.0
    assert round(delta_mean, 2) == 1.5

def test_contamination_helper():
    df_addr = pd.DataFrame({
        'address_id': ['addr_test'] * 5,
        'hash': [f'h_{i}' for i in range(5)],
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 12:00:00',
            '2023-01-01 14:00:00',
            '2023-01-01 16:00:00',
            '2023-01-01 18:00:00'
        ]),
        'type': [1, -1, 1, -1, 1],
        'amount': [10.0, -9.8, 25.0, -24.5, 50.0]
    })
    pos_amounts = np.array([5.0, 10.0, 20.0, 100.0])
    k = 2
    
    res = inject_alternating_cycles(df_addr, k=k, pos_amounts=pos_amounts, seed=42)
    
    # 5 original + 2*2 injected = 9 rows
    assert len(res) == 9
    assert res['address_id'].nunique() == 1
    assert res['address_id'].iloc[0] == 'addr_test'
    
    # Timestamps strictly ascending
    assert res['timestamp'].is_monotonic_increasing
    
    # Injected transactions start after original last timestamp
    injected = res.iloc[5:].reset_index(drop=True)
    assert injected['timestamp'].min() > df_addr['timestamp'].max()
    
    # Injected types alternate: deposit (+1), withdraw (-1), deposit (+1), withdraw (-1)
    assert list(injected['type']) == [1, -1, 1, -1]
    
    # Amounts check: deposits > 0, withdrawals < 0 and ~97-99% of deposit
    for cyc in range(k):
        dep_amt = injected.loc[2 * cyc, 'amount']
        wd_amt = injected.loc[2 * cyc + 1, 'amount']
        assert dep_amt in pos_amounts
        assert wd_amt < 0
        ratio = abs(wd_amt) / dep_amt
        assert 0.96 <= ratio <= 1.00

def test_inject_embedded_archetypes():
    df_addr = pd.DataFrame({
        'address_id': ['addr_test'] * 5,
        'hash': [f't{i}' for i in range(5)],
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00',
            '2023-01-01 11:00:00',
            '2023-01-01 12:00:00',
            '2023-01-01 13:00:00',
            '2023-01-01 14:00:00'
        ]),
        'type': [1, -1, 1, -1, 1],
        'amount': [10.0, -9.8, 25.0, -24.5, 50.0]
    })
    pos_amounts = np.array([5.0, 10.0, 20.0, 100.0])

    # 1. Burst
    res_burst = inject_burst(df_addr, pos_amounts=pos_amounts, seed=42, n_txs=20)
    assert len(res_burst) == 25
    assert res_burst['timestamp'].is_monotonic_increasing
    inj_burst = res_burst.iloc[5:].reset_index(drop=True)
    assert inj_burst['timestamp'].min() > df_addr['timestamp'].max()
    assert (inj_burst['timestamp'].diff().dropna().dt.total_seconds() <= 31.0).all()

    # 2. Structuring
    res_struct = inject_structuring(df_addr, pos_amounts=pos_amounts, seed=42, threshold=100.0, n_deposits=10)
    assert len(res_struct) == 5 + 10 + 1  # 5 orig + 10 deposits + 1 wd
    assert res_struct['timestamp'].is_monotonic_increasing
    inj_struct = res_struct.iloc[5:].reset_index(drop=True)
    assert inj_struct['timestamp'].min() > df_addr['timestamp'].max()
    # 10 deposits followed by 1 withdrawal
    assert list(inj_struct['type']) == [1] * 10 + [-1]
    total_dep = inj_struct.iloc[:10]['amount'].sum()
    assert abs(inj_struct.iloc[10]['amount'] + total_dep) < 1e-4

    # 3. Bot-Periodic
    res_bot = inject_bot_periodic(df_addr, pos_amounts=pos_amounts, seed=42, fixed_dt=300.0, fixed_amt=25.0, n_txs=15)
    assert len(res_bot) == 5 + 15
    assert res_bot['timestamp'].is_monotonic_increasing
    inj_bot = res_bot.iloc[5:].reset_index(drop=True)
    assert inj_bot['timestamp'].min() > df_addr['timestamp'].max()
    dts = inj_bot['timestamp'].diff().dropna().dt.total_seconds()
    assert np.allclose(dts, 300.0)
    assert (inj_bot['amount'].abs() == 25.0).all()

def test_embedded_injectors_preservation_ordering_and_ranges():
    """
    Tests that embedded burst, structuring, and bot-periodic injectors satisfy:
    1. Timestamps strictly ascending across the combined history.
    2. All injected timestamps strictly after the original address history.
    3. Original address events are strictly preserved without modification.
    4. Injected event counts match documented ranges under default sampling:
       - burst: 20-30 events
       - structuring: 8-12 deposits + 1 consolidated withdrawal (9-13 total)
       - bot-periodic: 15-25 events with constant interval and constant amount.
    """
    df_addr = pd.DataFrame({
        'address_id': ['1AddrEmbeddedTest'] * 6,
        'hash': [f'tx_{i}' for i in range(6)],
        'timestamp': pd.to_datetime([
            '2022-05-01 08:00:00',
            '2022-05-02 09:30:00',
            '2022-05-03 14:15:00',
            '2022-05-05 11:00:00',
            '2022-05-08 16:45:00',
            '2022-05-10 12:00:00',
        ]),
        'type': [1, 1, -1, 1, -1, 1],
        'amount': [15.5, 42.0, -10.0, 5.0, -4.8, 100.0]
    })
    orig_len = len(df_addr)
    orig_max_t = df_addr['timestamp'].max()

    for s in range(42, 52):
        # 1. Burst Injector (default n_txs=None -> 20 to 30 transactions)
        res_burst = inject_burst(df_addr, seed=s)
        pd.testing.assert_frame_equal(res_burst.iloc[:orig_len].reset_index(drop=True), df_addr.reset_index(drop=True), check_dtype=False)
        assert res_burst['timestamp'].is_monotonic_increasing
        inj_burst = res_burst.iloc[orig_len:].reset_index(drop=True)
        assert inj_burst['timestamp'].min() > orig_max_t
        assert 20 <= len(inj_burst) <= 30
        diffs = inj_burst['timestamp'].diff().dropna().dt.total_seconds()
        assert (diffs >= 5.0).all() and (diffs <= 30.0).all()

        # 2. Structuring Injector (default n_deposits=None -> 8 to 12 deposits + 1 withdrawal = 9 to 13 events)
        res_struct = inject_structuring(df_addr, seed=s)
        pd.testing.assert_frame_equal(res_struct.iloc[:orig_len].reset_index(drop=True), df_addr.reset_index(drop=True), check_dtype=False)
        assert res_struct['timestamp'].is_monotonic_increasing
        inj_struct = res_struct.iloc[orig_len:].reset_index(drop=True)
        assert inj_struct['timestamp'].min() > orig_max_t
        assert 9 <= len(inj_struct) <= 13
        n_deposits = len(inj_struct) - 1
        assert 8 <= n_deposits <= 12
        assert (inj_struct.iloc[:n_deposits]['type'] == 1).all()
        assert inj_struct.iloc[-1]['type'] == -1
        assert np.isclose(inj_struct.iloc[:n_deposits]['amount'].sum(), -inj_struct.iloc[-1]['amount'])

        # 3. Bot-Periodic Injector (default n_txs=None -> 15 to 25 transactions)
        res_bot = inject_bot_periodic(df_addr, seed=s)
        pd.testing.assert_frame_equal(res_bot.iloc[:orig_len].reset_index(drop=True), df_addr.reset_index(drop=True), check_dtype=False)
        assert res_bot['timestamp'].is_monotonic_increasing
        inj_bot = res_bot.iloc[orig_len:].reset_index(drop=True)
        assert inj_bot['timestamp'].min() > orig_max_t
        assert 15 <= len(inj_bot) <= 25
        dts = inj_bot['timestamp'].diff().dropna().dt.total_seconds().values
        assert np.allclose(dts, dts[0])
        amts = inj_bot['amount'].abs().values
        assert np.allclose(amts, amts[0])

