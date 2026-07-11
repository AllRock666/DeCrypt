import pandas as pd
import numpy as np
import networkx as nx
from typing import Tuple
from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

def extract_behavioral_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Extract statistical and behavioral features from transaction history.
    """
    logger.info("Extracting behavioral and temporal features...")
    address_features = pd.DataFrame()
    grouped = df.groupby('address_id')
    
    # Basic transaction features
    address_features['total_transactions'] = grouped.size()
    address_features['total_volume'] = grouped['amount'].sum().abs()
    address_features['avg_transaction_size'] = grouped['amount'].mean().abs()
    address_features['transaction_variance'] = grouped['amount'].var()
    address_features['deposit_withdraw_ratio'] = grouped['type'].mean()
    address_features['unique_interactions'] = grouped['hash'].nunique()
    address_features['max_transaction'] = grouped['amount'].max().abs()
    address_features['min_transaction'] = grouped['amount'].min().abs()
    
    # Time-based features
    first_tx = grouped['timestamp'].min()
    last_tx = grouped['timestamp'].max()
    
    address_features['activity_period_days'] = (
        (last_tx - first_tx).dt.total_seconds() / (24 * 3600)
    )
    
    global_min_time = df['timestamp'].min()
    address_features['days_since_first'] = (
        (first_tx - global_min_time).dt.total_seconds() / (24 * 3600)
    )
    address_features['days_since_last'] = (
        (last_tx - global_min_time).dt.total_seconds() / (24 * 3600)
    )
    
    # Advanced features
    address_features['large_transaction_ratio'] = grouped['amount'].apply(
        lambda x: (x.abs() > x.abs().mean() * config.large_tx_multiplier).mean()
    )
    
    # Temporal patterns
    # Calculate average time between transactions (in seconds)
    address_features['avg_time_between_txs'] = grouped.apply(
        lambda x: x.sort_values('timestamp')['timestamp'].diff().dt.total_seconds().mean() if len(x) > 1 else 0
    )
    
    # Variance in time between transactions
    address_features['tx_time_variance'] = grouped.apply(
        lambda x: x.sort_values('timestamp')['timestamp'].diff().dt.total_seconds().var() if len(x) > 1 else 0
    )
    
    # Periodic activity (Weekend ratio)
    address_features['weekend_ratio'] = grouped.apply(
        lambda x: (x['timestamp'].dt.dayofweek >= 5).mean()
    )
    
    return address_features.fillna(0)

def build_transaction_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph of transactions.
    Withdrawals (type=-1) are sources, Deposits (type=1) are targets.
    """
    logger.info("Building directed transaction graph...")
    G = nx.DiGraph()
    
    # Group by transaction hash
    transactions = df.groupby('hash')
    
    for tx_hash, group in transactions:
        sources = group[group['type'] == -1]
        targets = group[group['type'] == 1]
        
        timestamp = group['timestamp'].iloc[0]
        
        if sources.empty or targets.empty:
            continue
            
        # Distribute amounts (approximate flow)
        for _, src in sources.iterrows():
            for _, tgt in targets.iterrows():
                src_addr = src['address_id']
                tgt_addr = tgt['address_id']
                
                # Weight by target deposit amount for approximation
                weight = abs(tgt['amount'])
                
                if G.has_edge(src_addr, tgt_addr):
                    G[src_addr][tgt_addr]['weight'] += weight
                    G[src_addr][tgt_addr]['transactions'] += 1
                    G[src_addr][tgt_addr]['last_timestamp'] = max(
                        G[src_addr][tgt_addr].get('last_timestamp', timestamp), 
                        timestamp
                    )
                else:
                    G.add_edge(src_addr, tgt_addr, 
                             weight=weight, 
                             transactions=1,
                             first_timestamp=timestamp,
                             last_timestamp=timestamp)
                             
    logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def extract_network_features(G: nx.DiGraph, addresses: pd.Index) -> pd.DataFrame:
    """
    Extract network centrality and structural metrics from the directed graph.
    """
    logger.info("Extracting network features from graph...")
    network_features = pd.DataFrame(index=addresses)
    
    # In/Out Degree Centrality
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    
    network_features['in_degree_centrality'] = pd.Series(in_degree)
    network_features['out_degree_centrality'] = pd.Series(out_degree)
    
    # Clustering coefficient (converting to undirected for approximation, or using directed clustering if available)
    clustering = nx.clustering(G.to_undirected())
    network_features['clustering_coefficient'] = pd.Series(clustering)
    
    # Weighted degrees
    in_weights = {node: sum(data['weight'] for _, _, data in G.in_edges(node, data=True)) for node in G.nodes()}
    out_weights = {node: sum(data['weight'] for _, _, data in G.out_edges(node, data=True)) for node in G.nodes()}
    
    network_features['weighted_in_degree'] = pd.Series(in_weights)
    network_features['weighted_out_degree'] = pd.Series(out_weights)
    
    # PageRank (handles directed graphs well)
    try:
        pagerank = nx.pagerank(G, weight='weight')
        network_features['pagerank'] = pd.Series(pagerank)
    except Exception as e:
        logger.warning(f"PageRank computation failed: {e}. Defaulting to 0.")
        network_features['pagerank'] = 0.0
        
    return network_features.fillna(0)

def extract_all_features(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Coordinates extraction of all feature sets.
    """
    behavioral = extract_behavioral_features(df, config)
    G = build_transaction_graph(df)
    network = extract_network_features(G, behavioral.index)
    
    combined = pd.concat([behavioral, network], axis=1).fillna(0)
    logger.info(f"Total features extracted: {combined.shape[1]}")
    return combined, G
