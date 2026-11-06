import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

def load_transaction_data(filepath: Union[str, Path], config: Config = None) -> pd.DataFrame:
    """
    Loads transaction data from a CSV file and maps columns dynamically.
    """
    config = config or default_config
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    logger.info(f"Loading data from {filepath}")
    
    header = 0 if config.has_header else None
    df = pd.read_csv(filepath, header=header)
    
    # Helper to resolve column names or indices
    def get_col(col_spec, default_idx=None):
        if col_spec is None:
            return default_idx if (default_idx is not None and default_idx in df.columns) else None
        if isinstance(col_spec, int):
            return col_spec if col_spec in df.columns else None
        if isinstance(col_spec, str) and col_spec.isdigit():
            idx = int(col_spec)
            return idx if idx in df.columns else None
        if col_spec in df.columns:
            return col_spec
        if not config.has_header and default_idx is not None and default_idx in df.columns:
            return default_idx
        return None

    # Resolve columns
    col_id = get_col(config.col_id, default_idx=0)
    col_txid = get_col(config.col_txid, default_idx=1)
    col_time = get_col(config.col_time, default_idx=2)
    col_type = get_col(config.col_type, default_idx=3)
    col_amount = get_col(config.col_amount, default_idx=4)
    
    if not all([col_id is not None, col_amount is not None, col_time is not None]):
        raise ValueError(f"Required columns missing. Resolved to: ID={col_id}, Amount={col_amount}, Time={col_time}")
        
    # Rename to internal standard
    rename_map = {col_id: 'address_id', col_amount: 'amount', col_time: 'timestamp'}
    if col_txid is not None: rename_map[col_txid] = 'hash'
    if col_type is not None: rename_map[col_type] = 'raw_type'
    
    df = df.rename(columns=rename_map)
    
    # Process 'amount'
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    
    # Infer or Map 'type'
    if 'raw_type' in df.columns:
        def map_type(val):
            val_str = str(val).strip().lower()
            if val_str == config.type_deposit.lower(): return 1
            if val_str == config.type_withdraw.lower(): return -1
            return 0
        df['type'] = df['raw_type'].apply(map_type)
        # Drop the raw type column
        df = df.drop(columns=['raw_type'])
    else:
        # Infer from amount sign
        df['type'] = df['amount'].apply(lambda x: 1 if x >= 0 else -1)
        
    # Ensure signed amount: deposits positive, withdrawals negative
    df['amount'] = np.where(df['type'] == -1, -df['amount'].abs(), df['amount'].abs())
    df['original_amount'] = df['amount']

    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
    
    # Drop rows where timestamp couldn't be parsed if any
    missing_timestamps = df['timestamp'].isna().sum()
    if missing_timestamps > 0:
        logger.warning(f"Dropping {missing_timestamps} rows with invalid timestamps.")
        df = df.dropna(subset=['timestamp'])
        
    logger.info(f"Successfully loaded {len(df)} transactions.")
    return df
