import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Model parameters
    contamination: float = 0.1
    n_clusters: int = 10
    random_state: int = 42
    
    # Feature engineering parameters
    large_tx_multiplier: float = 3.0
    
    # Logging configuration
    log_level: int = logging.INFO
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logger(name: str, config: Config) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(config.log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Default global configuration instance
default_config = Config()
