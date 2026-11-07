import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple, Optional
from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

def extract_behavioral_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Extract vectorized statistical, behavioral, and temporal features from transaction history.
    """
    logger.info("Extracting vectorized behavioral and temporal features...")
    df_sorted = df.sort_values(['address_id', 'timestamp']).copy()
    
    # Precompute per-transaction helper columns
    df_sorted['abs_amount'] = df_sorted['amount'].abs()
    df_sorted['is_deposit'] = (df_sorted['type'] == 1).astype(float)
    df_sorted['is_withdraw'] = (df_sorted['type'] == -1).astype(float)
    df_sorted['dep_amount'] = np.where(df_sorted['type'] == 1, df_sorted['abs_amount'], 0.0)
    df_sorted['wd_amount'] = np.where(df_sorted['type'] == -1, df_sorted['abs_amount'], 0.0)
    df_sorted['is_weekend'] = (df_sorted['timestamp'].dt.dayofweek >= 5).astype(float)
    
    # Delta t between consecutive transactions of the same address
    df_sorted['delta_t'] = df_sorted.groupby('address_id')['timestamp'].diff().dt.total_seconds()
    
    # Large transaction threshold per address
    mean_per_addr = df_sorted.groupby('address_id')['abs_amount'].transform('mean')
    df_sorted['is_large'] = (df_sorted['abs_amount'] > (mean_per_addr * config.large_tx_multiplier)).astype(float)
    
    # Group by address
    grouped = df_sorted.groupby('address_id')
    
    features = pd.DataFrame(index=grouped.indices.keys())
    features.index.name = 'address_id'
    
    # 1. Volume & counts
    features['total_transactions'] = grouped.size()
    features['total_volume'] = grouped['abs_amount'].sum()
    features['net_flow'] = grouped['amount'].sum()
    features['avg_transaction_size'] = grouped['abs_amount'].mean()
    features['max_transaction'] = grouped['abs_amount'].max()
    features['min_transaction'] = grouped['abs_amount'].min()
    features['transaction_variance'] = grouped['abs_amount'].var().fillna(0.0)
    
    # 2. Deposit vs. Withdrawal flow
    features['total_deposited'] = grouped['dep_amount'].sum()
    features['total_withdrawn'] = grouped['wd_amount'].sum()
    features['n_deposits'] = grouped['is_deposit'].sum()
    features['n_withdrawals'] = grouped['is_withdraw'].sum()
    features['deposit_withdraw_ratio'] = grouped['type'].mean()
    
    # Pass-through ratio: min(dep, wd) / max(dep, wd)
    dep = features['total_deposited']
    wd = features['total_withdrawn']
    max_flow = np.maximum(dep, wd)
    features['pass_through_ratio'] = np.where(max_flow > 0, np.minimum(dep, wd) / max_flow, 0.0)
    
    # 3. Timestamps & Recency (strictly entity-relative)
    first_tx = grouped['timestamp'].min()
    last_tx = grouped['timestamp'].max()
    
    span_seconds = (last_tx - first_tx).dt.total_seconds()
    features['activity_period_days'] = span_seconds / 86400.0
    
    # Entity-relative activity centroid: mean time offset from first_tx normalized by active span [0.0, 1.0]
    first_tx_per_row = df_sorted.groupby('address_id')['timestamp'].transform('min')
    df_sorted['offset_sec'] = (df_sorted['timestamp'] - first_tx_per_row).dt.total_seconds()
    mean_offset = df_sorted.groupby('address_id')['offset_sec'].mean()
    features['activity_centroid'] = pd.Series(
        np.where(span_seconds > 0, mean_offset / span_seconds, 0.5),
        index=features.index
    ).fillna(0.5)
    
    # 4. Temporal dynamics & Burstiness
    delta_stats = grouped['delta_t'].agg(['mean', 'std', 'var'])
    mu = delta_stats['mean']
    sigma = delta_stats['std'].fillna(0.0)
    denom = sigma + mu
    burstiness = np.where(denom > 0, (sigma - mu) / denom, 0.0)
    features['burstiness'] = np.clip(burstiness, -1.0, 1.0)
    features['avg_time_between_txs'] = mu.fillna(0.0)
    features['tx_time_variance'] = delta_stats['var'].fillna(0.0)
    
    # Coefficient of Variation (CV) of inter-event times: sigma / mu (approaches 0 for bot-periodic)
    features['cv_inter_event_time'] = np.where(mu > 0, sigma / mu, 0.0)
    
    # Lag-1 autocorrelation of inter-event times
    def lag1_autocorr(sub):
        s = sub['delta_t'].dropna()
        if len(s) < 3:
            return 0.0
        v = s.var()
        if v == 0 or np.isnan(v):
            return 0.0
        ac = s.autocorr(lag=1)
        return 0.0 if np.isnan(ac) else float(ac)
        
    features['lag1_autocorr_inter_event_time'] = (
        df_sorted.groupby('address_id')
        .apply(lag1_autocorr, include_groups=False)
        .reindex(features.index)
        .fillna(0.0)
    )
    
    # Modal-amount share (proportion of txs equaling the most common amount)
    amt_counts = df_sorted.groupby(['address_id', 'abs_amount']).size()
    max_amt_cnt = amt_counts.groupby('address_id').max()
    features['modal_amount_share'] = (max_amt_cnt / features['total_transactions']).reindex(features.index).fillna(0.0)
    
    # 5. Amount Shannon Entropy across 10 global quantile bins
    non_zero = df_sorted['abs_amount'][df_sorted['abs_amount'] > 0]
    n_unique = non_zero.nunique()
    if n_unique > 1:
        n_bins = min(10, n_unique)
        try:
            q_cuts = pd.qcut(df_sorted['abs_amount'], q=n_bins, labels=False, duplicates='drop')
            df_sorted['amt_bin'] = q_cuts
            bin_counts = df_sorted.groupby(['address_id', 'amt_bin'], observed=False).size().unstack(fill_value=0)
            bin_probs = bin_counts.div(bin_counts.sum(axis=1), axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                log2_p = np.where(bin_probs > 0, np.log2(bin_probs), 0.0)
                entropy_series = -(bin_probs * log2_p).sum(axis=1)
            features['amount_entropy'] = entropy_series.reindex(features.index).fillna(0.0)
        except Exception as e:
            logger.warning(f"Could not compute quantile amount entropy ({e}), defaulting to 0.")
            features['amount_entropy'] = 0.0
    else:
        features['amount_entropy'] = 0.0
        
    # 6. Behavioral ratios
    features['large_transaction_ratio'] = grouped['is_large'].mean().fillna(0.0)
    features['weekend_ratio'] = grouped['is_weekend'].mean().fillna(0.0)
    
    # 7. Direction alternation dynamics
    df_sorted['prev_type'] = df_sorted.groupby('address_id')['type'].shift(1)
    df_sorted['is_switch'] = (df_sorted['prev_type'].notna() & (df_sorted['type'] != df_sorted['prev_type'])).astype(int)
    
    direction_switch_count = df_sorted.groupby('address_id')['is_switch'].sum()
    features['direction_switch_count'] = direction_switch_count.reindex(features.index).fillna(0.0)
    
    n_txs = features['total_transactions']
    features['alternation_rate'] = np.where(
        n_txs > 1,
        features['direction_switch_count'] / np.maximum(1.0, n_txs - 1.0),
        0.0
    )
    
    def calc_max_alternating_run(types):
        if len(types) <= 1:
            return 0
        switches = (types[1:] != types[:-1])
        max_run = 0
        curr_run = 0
        for s in switches:
            if s:
                curr_run += 1
                if curr_run > max_run:
                    max_run = curr_run
            else:
                curr_run = 0
        return max_run + 1 if max_run > 0 else 0

    features['max_alternating_run_length'] = (
        df_sorted.groupby('address_id')['type']
        .apply(lambda s: calc_max_alternating_run(s.values))
        .reindex(features.index)
        .fillna(0.0)
    )
    
    if 'hash' in df.columns:
        features['unique_interactions'] = grouped['hash'].nunique()
    else:
        features['unique_interactions'] = 0
        
    return features.fillna(0.0)

def extract_legacy_features(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Reproduces the OLD feature extraction logic with all original flaws
    for benchmarking and verification.
    """
    logger.info("Extracting LEGACY (old baseline) features...")
    address_features = pd.DataFrame()
    grouped = df.groupby('address_id')
    
    # Flawed volume: net sum abs()
    address_features['total_transactions'] = grouped.size()
    address_features['total_volume'] = grouped['amount'].sum().abs()
    address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
    address_features['transaction_variance'] = grouped['amount'].var()
    address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
    if 'hash' in df.columns:
        address_features['unique_interactions'] = grouped['hash'].nunique()
    else:
        address_features['unique_interactions'] = 0
    address_features['max_transaction'] = grouped['amount'].max().abs()
    address_features['min_transaction'] = grouped['amount'].min().abs()
    
    first_tx = grouped['timestamp'].min()
    last_tx = grouped['timestamp'].max()
    
    address_features['activity_period_days'] = (
        (last_tx - first_tx).dt.total_seconds() / (24 * 3600)
    )
    global_min_time = df['timestamp'].min()
    address_features['days_since_first'] = (
        (first_tx - global_min_time).dt.total_seconds() / (24 * 3600)
    )
    # Flawed recency: days since start instead of until end
    address_features['days_since_last'] = (
        (last_tx - global_min_time).dt.total_seconds() / (24 * 3600)
    )
    
    address_features['large_transaction_ratio'] = grouped['amount'].apply(
        lambda x: (x.abs() > x.abs().mean() * config.large_tx_multiplier).mean()
    )
    address_features['avg_time_between_txs'] = grouped.apply(
        lambda x: x.sort_values('timestamp')['timestamp'].diff().dt.total_seconds().mean() if len(x) > 1 else 0
    )
    address_features['tx_time_variance'] = grouped.apply(
        lambda x: x.sort_values('timestamp')['timestamp'].diff().dt.total_seconds().var() if len(x) > 1 else 0
    )
    address_features['weekend_ratio'] = grouped.apply(
        lambda x: (x['timestamp'].dt.dayofweek >= 5).mean()
    )
    
    # Old empty graph
    G = nx.DiGraph()
    logger.info("Legacy feature extraction complete (15 features, empty graph).")
    return address_features.fillna(0.0), G

def build_heuristic_flow_graph(df: pd.DataFrame, config: Config) -> nx.DiGraph:
    """
    Build a heuristic flow graph connecting withdrawals to deposits.
    An edge A -> B is created if address A has a withdrawal and address B has a deposit
    within link_window_hours, with matching amounts within link_tolerance.
    """
    logger.info(
        f"Building HEURISTIC flow graph (window={config.link_window_hours}h, "
        f"tolerance={config.link_tolerance*100:.1f}%, min_edges={config.min_edges})..."
    )
    logger.info("NOTE: Flow graph is heuristic as the dataset lacks direct input-output transaction linkages.")
    G = nx.DiGraph()
    
    withdrawals = df[df['type'] == -1].sort_values('timestamp')
    deposits = df[df['type'] == 1].sort_values('timestamp')
    
    if withdrawals.empty or deposits.empty:
        logger.warning("Dataset missing withdrawals or deposits; cannot build heuristic flow graph.")
        return G
        
    dep_times = deposits['timestamp'].values
    dep_amts = deposits['amount'].abs().values
    dep_addrs = deposits['address_id'].values
    win_ns = np.timedelta64(int(config.link_window_hours * 3600), 's')
    
    edges = {}
    for _, w in withdrawals.iterrows():
        w_t = np.datetime64(w['timestamp'])
        w_amt = abs(w['amount'])
        w_addr = w['address_id']
        
        idx_start = np.searchsorted(dep_times, w_t, side='left')
        idx_end = np.searchsorted(dep_times, w_t + win_ns, side='right')
        
        if idx_end > idx_start:
            c_amts = dep_amts[idx_start:idx_end]
            c_addrs = dep_addrs[idx_start:idx_end]
            c_times = dep_times[idx_start:idx_end]
            
            denom = max(w_amt, 1e-9)
            rel_diff = np.abs(c_amts - w_amt) / denom
            valid_mask = (c_addrs != w_addr) & (rel_diff <= config.link_tolerance)
            
            for t_addr, t_amt, t_time in zip(c_addrs[valid_mask], c_amts[valid_mask], c_times[valid_mask]):
                edge_key = (w_addr, t_addr)
                weight = float(w_amt + t_amt) / 2.0
                if edge_key not in edges:
                    edges[edge_key] = {
                        'weight': weight,
                        'transactions': 1,
                        'first_timestamp': w['timestamp'],
                        'last_timestamp': pd.to_datetime(t_time)
                    }
                else:
                    edges[edge_key]['weight'] += weight
                    edges[edge_key]['transactions'] += 1
                    edges[edge_key]['last_timestamp'] = max(
                        edges[edge_key]['last_timestamp'],
                        pd.to_datetime(t_time)
                    )
                    
    for (u, v), attr in edges.items():
        G.add_edge(u, v, **attr)
        
    logger.info(f"Heuristic flow graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def extract_network_features(G: nx.DiGraph, addresses: pd.Index, config: Config) -> Optional[pd.DataFrame]:
    """
    Extract network centrality and structural metrics from the graph.
    If the graph has fewer than min_edges, skip network features with a warning.
    """
    if G.number_of_edges() < config.min_edges:
        logger.warning(
            f"Heuristic graph has only {G.number_of_edges()} edges (< min_edges={config.min_edges}). "
            "SKIPPING network feature extraction."
        )
        return None
        
    logger.info("Extracting network features from heuristic flow graph...")
    network_features = pd.DataFrame(index=addresses)
    
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    network_features['in_degree_centrality'] = pd.Series(in_degree)
    network_features['out_degree_centrality'] = pd.Series(out_degree)
    
    clustering = nx.clustering(G.to_undirected())
    network_features['clustering_coefficient'] = pd.Series(clustering)
    
    in_weights = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    out_weights = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}
    network_features['weighted_in_degree'] = pd.Series(in_weights)
    network_features['weighted_out_degree'] = pd.Series(out_weights)
    
    try:
        pagerank = nx.pagerank(G, weight='weight', max_iter=200)
        network_features['pagerank'] = pd.Series(pagerank)
    except Exception as e:
        logger.warning(f"PageRank computation failed: {e}. Defaulting to 0.")
        network_features['pagerank'] = 0.0
        
    return network_features.fillna(0.0)

def prepare_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Selects numeric columns and drops zero-variance columns explicitly with logging.
    """
    numeric_df = features.select_dtypes(include='number').copy()
    zero_var_cols = [c for c in numeric_df.columns if numeric_df[c].std() == 0 or np.isnan(numeric_df[c].std())]
    if zero_var_cols:
        logger.warning(f"Dropping {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")
        numeric_df = numeric_df.drop(columns=zero_var_cols)
    return numeric_df

def extract_all_features(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Coordinates extraction of all feature sets.
    Network features are excluded by default (graph is a negative result showing no evidence above shuffled nulls),
    and can be enabled via config.include_network_features.
    """
    if config.legacy_features:
        return extract_legacy_features(df, config)
        
    behavioral = extract_behavioral_features(df, config)
    G = build_heuristic_flow_graph(df, config)
    
    if config.include_network_features:
        network = extract_network_features(G, behavioral.index, config)
        if network is not None:
            combined = pd.concat([behavioral, network], axis=1).fillna(0.0)
        else:
            combined = behavioral.copy()
    else:
        logger.info("Excluding network features from scoring by default (graph is a negative result).")
        combined = behavioral.copy()
        
    logger.info(f"Total features extracted: {combined.shape[1]}")
    return combined, G
