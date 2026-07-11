import pandas as pd
from pathlib import Path
from typing import Union
from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

def load_transaction_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads Bitcoin transaction data from a CSV file.
    Expects columns: address_id, hash, timestamp, type, amount
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Data file not found at {filepath}")
        
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(
        filepath, 
        header=None,
        names=['address_id', 'hash', 'timestamp', 'type', 'amount']
    )
    
    # Map transaction type to numeric
    df['type'] = df['type'].map({'deposit': 1, 'withdraw': -1})
    
    # Ensure amount is numeric, coercing errors to NaN and then filling with 0 or dropping
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    
    # Parse timestamp. The format in sample is DD-MM-YYYY HH:MM
    # We use dayfirst=True to handle cases like 27-07-2011 correctly.
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')
    
    # Drop rows where timestamp couldn't be parsed if any
    missing_timestamps = df['timestamp'].isna().sum()
    if missing_timestamps > 0:
        logger.warning(f"Dropping {missing_timestamps} rows with invalid timestamps.")
        df = df.dropna(subset=['timestamp'])
        
    logger.info(f"Successfully loaded {len(df)} transactions.")
    return df
