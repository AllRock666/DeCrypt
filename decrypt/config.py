import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model parameters
    contamination: float = 0.1
    n_clusters: int = 10
    random_state: int = 42
    seed: int = 42
    if_max_features: float = 1.0
    
    # Feature engineering parameters
    large_tx_multiplier: float = 3.0
    link_window_hours: float = 24.0
    link_tolerance: float = 0.02
    min_edges: int = 10
    legacy_features: bool = False
    include_network_features: bool = False
    
    # Sequence & Transformer parameters
    no_transformer: bool = False
    epochs: int = 20
    min_seq_len: int = 3
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_seq_len: int = 128
    transformer_weight: float = 0.2
    if_weight: float = 0.8
    sequence_aggregation: str = 'window_max'
    rolling_window: int = 5
    
    # Benchmark flag
    benchmark: bool = False
    profile: str = 'batch'  # 'batch' or 'streaming'
    
    # Schema Mapping parameters
    col_id: str = 'address_id'
    col_amount: str = 'amount'
    col_time: str = 'timestamp'
    col_txid: Optional[str] = 'hash'
    col_type: Optional[str] = 'type'
    type_deposit: str = 'deposit'
    type_withdraw: str = 'withdraw'
    has_header: bool = False
    
    # Logging configuration
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(name: str, config: Config) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(config.log_level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Default global configuration instance
default_config = Config()
