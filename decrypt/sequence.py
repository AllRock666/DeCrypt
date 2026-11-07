import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import rankdata

from .config import Config, setup_logger, default_config

logger = setup_logger(__name__, default_config)

# Token Constants
MASK_DIR = 2        # 0: deposit, 1: withdraw, 2: MASK
MASK_AMT_BIN = 32   # 0..31: bins, 32: MASK
MASK_DT_BIN = 32    # 0..31: bins, 32: MASK
MASK_HOUR = 24      # 0..23: hours, 24: MASK
MASK_DOW = 7        # 0..6: days, 7: MASK
MAX_SEQ_LEN = 128

class EventTransformerModel(nn.Module):
    """
    Compact Transformer encoder for discrete/continuous event sequences.
    Trained via BERT-style masked event modeling (15% masking).
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Categorical Embeddings (including MASK tokens)
        self.dir_emb = nn.Embedding(3, 16)
        self.amt_bin_emb = nn.Embedding(33, 16)
        self.dt_bin_emb = nn.Embedding(33, 16)
        self.hour_emb = nn.Embedding(25, 8)
        self.dow_emb = nn.Embedding(8, 8)
        
        # Continuous Feature Projections
        self.amt_cont = nn.Linear(1, 8)
        self.dt_cont = nn.Linear(1, 8)
        
        # Combined projection: 16 + 16 + 16 + 8 + 8 + 8 + 8 = 80 -> d_model (64)
        self.in_proj = nn.Linear(80, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, MAX_SEQ_LEN, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction Heads for Masked Language Modeling
        self.head_dir = nn.Linear(d_model, 2)
        self.head_amt_bin = nn.Linear(d_model, 32)
        self.head_dt_bin = nn.Linear(d_model, 32)

    def forward(self, dir_t, amt_bin_t, dt_bin_t, hour_t, dow_t, amt_cont_t, dt_cont_t, mask_padding=None):
        batch_size, seq_len = dir_t.shape
        
        d_e = self.dir_emb(dir_t)
        a_b_e = self.amt_bin_emb(amt_bin_t)
        dt_b_e = self.dt_bin_emb(dt_bin_t)
        h_e = self.hour_emb(hour_t)
        dow_e = self.dow_emb(dow_t)
        
        a_c_e = self.amt_cont(amt_cont_t.unsqueeze(-1))
        dt_c_e = self.dt_cont(dt_cont_t.unsqueeze(-1))
        
        feat_cat = torch.cat([d_e, a_b_e, dt_b_e, h_e, dow_e, a_c_e, dt_c_e], dim=-1)
        x = self.in_proj(feat_cat)
        x = x + self.pos_emb[:, :seq_len, :]
        x = self.dropout(x)
        
        # mask_padding: True indicates padding positions to ignore in attention
        hidden = self.transformer(x, src_key_padding_mask=mask_padding)
        
        logits_dir = self.head_dir(hidden)
        logits_amt = self.head_amt_bin(hidden)
        logits_dt = self.head_dt_bin(hidden)
        
        return hidden, logits_dir, logits_amt, logits_dt

def compute_bin_edges(df: pd.DataFrame) -> Tuple[List[float], List[float]]:
    """
    Computes 32 quantile bins for |amount| and log1p(delta-t).
    Returns lists of 31 interior cutoffs.
    """
    abs_amounts = df['amount'].abs().values
    non_zero_amts = abs_amounts[abs_amounts > 0]
    if len(non_zero_amts) >= 32:
        quantiles = np.linspace(0, 1, 33)[1:-1]
        amt_edges = np.quantile(non_zero_amts, quantiles).tolist()
    else:
        amt_edges = np.linspace(0.01, 100.0, 31).tolist()
        
    df_sorted = df.sort_values(['address_id', 'timestamp'])
    dt_vals = df_sorted.groupby('address_id')['timestamp'].diff().dt.total_seconds().dropna().values
    log_dt = np.log1p(np.maximum(dt_vals, 0.0))
    if len(log_dt) >= 32:
        quantiles = np.linspace(0, 1, 33)[1:-1]
        dt_edges = np.quantile(log_dt, quantiles).tolist()
    else:
        dt_edges = np.linspace(0, 15.0, 31).tolist()
        
    return amt_edges, dt_edges

def digitize_values(values: np.ndarray, edges: List[float]) -> np.ndarray:
    return np.clip(np.digitize(values, edges), 0, 31)

def extract_address_events(df: pd.DataFrame, amt_edges: List[float], dt_edges: List[float]) -> Dict[str, Dict[str, Any]]:
    """
    Preprocesses raw transaction DataFrame into per-address sorted event arrays.
    """
    df_sorted = df.sort_values(['address_id', 'timestamp']).copy()
    df_sorted['abs_amount'] = df_sorted['amount'].abs()
    df_sorted['log_amt'] = np.log1p(df_sorted['abs_amount'])
    df_sorted['delta_t'] = df_sorted.groupby('address_id')['timestamp'].diff().dt.total_seconds().fillna(0.0)
    df_sorted['log_dt'] = np.log1p(np.maximum(df_sorted['delta_t'], 0.0))
    
    df_sorted['direction'] = np.where(df_sorted['type'] == -1, 1, 0)
    df_sorted['amt_bin'] = digitize_values(df_sorted['abs_amount'].values, amt_edges)
    df_sorted['dt_bin'] = digitize_values(df_sorted['log_dt'].values, dt_edges)
    df_sorted['hour'] = df_sorted['timestamp'].dt.hour
    df_sorted['dow'] = df_sorted['timestamp'].dt.dayofweek
    
    events_by_address = {}
    for addr, grp in df_sorted.groupby('address_id'):
        events_by_address[str(addr)] = {
            'direction': grp['direction'].values.astype(np.int64),
            'amt_bin': grp['amt_bin'].values.astype(np.int64),
            'dt_bin': grp['dt_bin'].values.astype(np.int64),
            'hour': grp['hour'].values.astype(np.int64),
            'dow': grp['dow'].values.astype(np.int64),
            'amt_cont': grp['log_amt'].values.astype(np.float32),
            'dt_cont': grp['log_dt'].values.astype(np.float32),
            'timestamps': grp['timestamp'].values,
            'amounts': grp['amount'].values,
            'length': len(grp)
        }
    return events_by_address

class AddressSequenceDataset(Dataset):
    """
    PyTorch Dataset for masked event modeling training.
    """
    def __init__(self, address_keys: List[str], events_by_address: Dict[str, Dict[str, Any]], mask_prob: float = 0.15, seed: int = 42):
        self.address_keys = address_keys
        self.events_by_address = events_by_address
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.address_keys)

    def __getitem__(self, idx):
        data = self.events_by_address[self.address_keys[idx]]
        full_len = data['length']
        # Take the most recent MAX_SEQ_LEN events
        seq_len = min(full_len, MAX_SEQ_LEN)
        start_idx = full_len - seq_len
        
        dir_seq = data['direction'][start_idx:].copy()
        amt_bin_seq = data['amt_bin'][start_idx:].copy()
        dt_bin_seq = data['dt_bin'][start_idx:].copy()
        hour_seq = data['hour'][start_idx:].copy()
        dow_seq = data['dow'][start_idx:].copy()
        amt_cont_seq = data['amt_cont'][start_idx:].copy()
        dt_cont_seq = data['dt_cont'][start_idx:].copy()
        
        # Create targets for masked tokens
        target_dir = dir_seq.copy()
        target_amt_bin = amt_bin_seq.copy()
        target_dt_bin = dt_bin_seq.copy()
        
        # 15% random masking using instance RNG
        mask = self.rng.random(seq_len) < self.mask_prob
        # Ensure at least 1 token is masked if length > 1
        if seq_len > 1 and not mask.any():
            mask[self.rng.integers(0, seq_len)] = True
            
        masked_dir = dir_seq.copy()
        masked_amt_bin = amt_bin_seq.copy()
        masked_dt_bin = dt_bin_seq.copy()
        masked_hour = hour_seq.copy()
        masked_dow = dow_seq.copy()
        masked_amt_cont = amt_cont_seq.copy()
        masked_dt_cont = dt_cont_seq.copy()
        
        masked_dir[mask] = MASK_DIR
        masked_amt_bin[mask] = MASK_AMT_BIN
        masked_dt_bin[mask] = MASK_DT_BIN
        masked_hour[mask] = MASK_HOUR
        masked_dow[mask] = MASK_DOW
        masked_amt_cont[mask] = 0.0
        masked_dt_cont[mask] = 0.0
        
        # Targets for unmasked tokens set to -100 (ignored by CrossEntropyLoss)
        target_dir[~mask] = -100
        target_amt_bin[~mask] = -100
        target_dt_bin[~mask] = -100
        
        return {
            'dir': torch.tensor(masked_dir, dtype=torch.long),
            'amt_bin': torch.tensor(masked_amt_bin, dtype=torch.long),
            'dt_bin': torch.tensor(masked_dt_bin, dtype=torch.long),
            'hour': torch.tensor(masked_hour, dtype=torch.long),
            'dow': torch.tensor(masked_dow, dtype=torch.long),
            'amt_cont': torch.tensor(masked_amt_cont, dtype=torch.float32),
            'dt_cont': torch.tensor(masked_dt_cont, dtype=torch.float32),
            'target_dir': torch.tensor(target_dir, dtype=torch.long),
            'target_amt': torch.tensor(target_amt_bin, dtype=torch.long),
            'target_dt': torch.tensor(target_dt_bin, dtype=torch.long),
            'seq_len': seq_len
        }

def collate_fn(batch):
    max_len = max(item['seq_len'] for item in batch)
    b_size = len(batch)
    
    dir_pad = torch.full((b_size, max_len), fill_value=MASK_DIR, dtype=torch.long)
    amt_bin_pad = torch.full((b_size, max_len), fill_value=MASK_AMT_BIN, dtype=torch.long)
    dt_bin_pad = torch.full((b_size, max_len), fill_value=MASK_DT_BIN, dtype=torch.long)
    hour_pad = torch.full((b_size, max_len), fill_value=MASK_HOUR, dtype=torch.long)
    dow_pad = torch.full((b_size, max_len), fill_value=MASK_DOW, dtype=torch.long)
    amt_cont_pad = torch.zeros((b_size, max_len), dtype=torch.float32)
    dt_cont_pad = torch.zeros((b_size, max_len), dtype=torch.float32)
    
    target_dir_pad = torch.full((b_size, max_len), fill_value=-100, dtype=torch.long)
    target_amt_pad = torch.full((b_size, max_len), fill_value=-100, dtype=torch.long)
    target_dt_pad = torch.full((b_size, max_len), fill_value=-100, dtype=torch.long)
    padding_mask = torch.ones((b_size, max_len), dtype=torch.bool)
    
    for i, item in enumerate(batch):
        l = item['seq_len']
        dir_pad[i, :l] = item['dir']
        amt_bin_pad[i, :l] = item['amt_bin']
        dt_bin_pad[i, :l] = item['dt_bin']
        hour_pad[i, :l] = item['hour']
        dow_pad[i, :l] = item['dow']
        amt_cont_pad[i, :l] = item['amt_cont']
        dt_cont_pad[i, :l] = item['dt_cont']
        
        target_dir_pad[i, :l] = item['target_dir']
        target_amt_pad[i, :l] = item['target_amt']
        target_dt_pad[i, :l] = item['target_dt']
        padding_mask[i, :l] = False
        
    return {
        'dir': dir_pad,
        'amt_bin': amt_bin_pad,
        'dt_bin': dt_bin_pad,
        'hour': hour_pad,
        'dow': dow_pad,
        'amt_cont': amt_cont_pad,
        'dt_cont': dt_cont_pad,
        'target_dir': target_dir_pad,
        'target_amt': target_amt_pad,
        'target_dt': target_dt_pad,
        'padding_mask': padding_mask
    }

class SequenceAnomalyDetector:
    """
    Manages Transformer sequence training, NLL surprise scoring,
    explainability extraction, and HDBSCAN/KMeans embedding clustering.
    """
    def __init__(self, config: Config):
        self.config = config
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.model = EventTransformerModel(d_model=64, n_heads=4, num_layers=2, dropout=0.1)
        self.amt_edges: List[float] = []
        self.dt_edges: List[float] = []
        
    def fit(self, df: pd.DataFrame, outdir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Fits bin edges and trains Transformer on real transactions.
        """
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        logger.info("Computing quantile bin edges on training addresses...")
        self.amt_edges, self.dt_edges = compute_bin_edges(df)
        events_by_address = extract_address_events(df, self.amt_edges, self.dt_edges)
        
        # Filter addresses with at least min_seq_len events for training
        valid_addrs = sorted([k for k, v in events_by_address.items() if v['length'] >= self.config.min_seq_len])
        if len(valid_addrs) < 10:
            logger.warning("Fewer than 10 addresses with sufficient history. Skipping Transformer training.")
            return {'trained': False}
            
        rng_fit = np.random.default_rng(self.config.seed)
        rng_fit.shuffle(valid_addrs)
        val_size = max(1, int(len(valid_addrs) * 0.10))
        train_addrs = valid_addrs[val_size:]
        val_addrs = valid_addrs[:val_size]
        
        train_dataset = AddressSequenceDataset(train_addrs, events_by_address, mask_prob=0.15, seed=self.config.seed)
        val_dataset = AddressSequenceDataset(val_addrs, events_by_address, mask_prob=0.15, seed=self.config.seed + 1)
        
        g_train = torch.Generator()
        g_train.manual_seed(self.config.seed)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, generator=g_train)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        logger.info(f"Training Sequence Transformer ({len(train_addrs)} train, {len(val_addrs)} val, {self.config.epochs} epochs)...")
        self.model.train()
        
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_state = None
        
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = 0.0
            total_tokens = 0
            
            for b in train_loader:
                optimizer.zero_grad()
                _, l_dir, l_amt, l_dt = self.model(
                    b['dir'], b['amt_bin'], b['dt_bin'], b['hour'], b['dow'],
                    b['amt_cont'], b['dt_cont'], mask_padding=b['padding_mask']
                )
                
                loss_dir = criterion(l_dir.view(-1, 2), b['target_dir'].view(-1))
                loss_amt = criterion(l_amt.view(-1, 32), b['target_amt'].view(-1))
                loss_dt = criterion(l_dt.view(-1, 32), b['target_dt'].view(-1))
                
                loss = loss_dir + loss_amt + loss_dt
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                total_tokens += 1
                
            avg_train_loss = train_loss / max(total_tokens, 1)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for b in val_loader:
                    _, l_dir, l_amt, l_dt = self.model(
                        b['dir'], b['amt_bin'], b['dt_bin'], b['hour'], b['dow'],
                        b['amt_cont'], b['dt_cont'], mask_padding=b['padding_mask']
                    )
                    l_d = criterion(l_dir.view(-1, 2), b['target_dir'].view(-1))
                    l_a = criterion(l_amt.view(-1, 32), b['target_amt'].view(-1))
                    l_t = criterion(l_dt.view(-1, 32), b['target_dt'].view(-1))
                    val_loss += (l_d + l_a + l_t).item()
                    val_tokens += 1
                    
            avg_val_loss = val_loss / max(val_tokens, 1)
            logger.debug(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                    break
                    
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        logger.info(f"Transformer training complete. Final val_loss={best_val_loss:.4f}")
        
        # Serialize model and binning
        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            model_path = outdir / "transformer_model.pt"
            bins_path = outdir / "transformer_bins.json"
            torch.save(self.model.state_dict(), model_path)
            with open(bins_path, 'w') as f:
                json.dump({'amt_edges': self.amt_edges, 'dt_edges': self.dt_edges}, f, indent=2)
            logger.info(f"Saved transformer model to {model_path} and bins to {bins_path}")
            
        return {'trained': True, 'best_val_loss': float(best_val_loss)}

    def score_addresses(
        self,
        df: pd.DataFrame,
        top_k: int = 3,
        rolling_window: int = 5,
        use_aggregation: str = 'window_max'
    ) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]], np.ndarray]:
        """
        Computes leave-one-out surprise scores, top-3 explainability events,
        and address sequence embeddings for all addresses.
        Supports score aggregations:
          - transformer_surprise (primary score using chosen aggregation, default: window_max with w=5)
          - transformer_surprise_mean (mean across all events)
          - transformer_surprise_topk (mean of top-k most surprising events)
          - transformer_surprise_window_max (maximum rolling window mean surprise)
        """
        self.model.eval()
        events_by_address = extract_address_events(df, self.amt_edges, self.dt_edges)
        
        records = []
        explanations = {}
        embeddings_list = []
        address_order = []
        
        criterion_dir = nn.CrossEntropyLoss(reduction='none')
        criterion_amt = nn.CrossEntropyLoss(reduction='none')
        criterion_dt = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for addr, data in events_by_address.items():
                length = data['length']
                address_order.append(addr)
                
                # Check minimum history threshold
                if length < self.config.min_seq_len:
                    records.append({
                        'address': addr,
                        'transformer_surprise': 0.0,
                        'transformer_surprise_mean': 0.0,
                        'transformer_surprise_topk': 0.0,
                        'transformer_surprise_window_max': 0.0,
                        'insufficient_history': True,
                        'seq_length': length
                    })
                    explanations[addr] = []
                    # Dummy zero embedding
                    embeddings_list.append(np.zeros(64, dtype=np.float32))
                    continue
                    
                # Sliding window handling for long sequences
                window_size = min(length, MAX_SEQ_LEN)
                # If length <= 128, one window
                # If length > 128, extract stride-64 windows and average
                stride = 64
                starts = list(range(0, max(1, length - window_size + 1), stride))
                if starts[-1] + window_size < length:
                    starts.append(length - window_size)
                    
                event_nll_accum = np.zeros(length, dtype=np.float32)
                event_counts = np.zeros(length, dtype=np.int32)
                window_embeds = []
                
                for start_i in starts:
                    end_i = start_i + window_size
                    w_dir = data['direction'][start_i:end_i]
                    w_amt_b = data['amt_bin'][start_i:end_i]
                    w_dt_b = data['dt_bin'][start_i:end_i]
                    w_hour = data['hour'][start_i:end_i]
                    w_dow = data['dow'][start_i:end_i]
                    w_amt_c = data['amt_cont'][start_i:end_i]
                    w_dt_c = data['dt_cont'][start_i:end_i]
                    
                    W = len(w_dir)
                    
                    # Compute mean-pooled embedding on unmasked sequence
                    t_dir_u = torch.tensor(w_dir, dtype=torch.long).unsqueeze(0)
                    t_amt_b_u = torch.tensor(w_amt_b, dtype=torch.long).unsqueeze(0)
                    t_dt_b_u = torch.tensor(w_dt_b, dtype=torch.long).unsqueeze(0)
                    t_hour_u = torch.tensor(w_hour, dtype=torch.long).unsqueeze(0)
                    t_dow_u = torch.tensor(w_dow, dtype=torch.long).unsqueeze(0)
                    t_amt_c_u = torch.tensor(w_amt_c, dtype=torch.float32).unsqueeze(0)
                    t_dt_c_u = torch.tensor(w_dt_c, dtype=torch.float32).unsqueeze(0)
                    
                    hidden_u, _, _, _ = self.model(
                        t_dir_u, t_amt_b_u, t_dt_b_u, t_hour_u, t_dow_u, t_amt_c_u, t_dt_c_u
                    )
                    window_embeds.append(hidden_u.squeeze(0).mean(dim=0).cpu().numpy())
                    
                    # Batch leave-one-out masking: batch size = W
                    # Each row has position j masked
                    b_dir = np.tile(w_dir, (W, 1))
                    b_amt_b = np.tile(w_amt_b, (W, 1))
                    b_dt_b = np.tile(w_dt_b, (W, 1))
                    b_hour = np.tile(w_hour, (W, 1))
                    b_dow = np.tile(w_dow, (W, 1))
                    b_amt_c = np.tile(w_amt_c, (W, 1))
                    b_dt_c = np.tile(w_dt_c, (W, 1))
                    
                    diag_idx = np.arange(W)
                    b_dir[diag_idx, diag_idx] = MASK_DIR
                    b_amt_b[diag_idx, diag_idx] = MASK_AMT_BIN
                    b_dt_b[diag_idx, diag_idx] = MASK_DT_BIN
                    b_hour[diag_idx, diag_idx] = MASK_HOUR
                    b_dow[diag_idx, diag_idx] = MASK_DOW
                    b_amt_c[diag_idx, diag_idx] = 0.0
                    b_dt_c[diag_idx, diag_idx] = 0.0
                    
                    tb_dir = torch.tensor(b_dir, dtype=torch.long)
                    tb_amt_b = torch.tensor(b_amt_b, dtype=torch.long)
                    tb_dt_b = torch.tensor(b_dt_b, dtype=torch.long)
                    tb_hour = torch.tensor(b_hour, dtype=torch.long)
                    tb_dow = torch.tensor(b_dow, dtype=torch.long)
                    tb_amt_c = torch.tensor(b_amt_c, dtype=torch.float32)
                    tb_dt_c = torch.tensor(b_dt_c, dtype=torch.float32)
                    
                    _, logits_d, logits_a, logits_t = self.model(
                        tb_dir, tb_amt_b, tb_dt_b, tb_hour, tb_dow, tb_amt_c, tb_dt_c
                    )
                    
                    # Extract diagonal masked position predictions
                    pred_d = logits_d[diag_idx, diag_idx]  # (W, 2)
                    pred_a = logits_a[diag_idx, diag_idx]  # (W, 32)
                    pred_t = logits_t[diag_idx, diag_idx]  # (W, 32)
                    
                    true_d = torch.tensor(w_dir, dtype=torch.long)
                    true_a = torch.tensor(w_amt_b, dtype=torch.long)
                    true_t = torch.tensor(w_dt_b, dtype=torch.long)
                    
                    nll_d = criterion_dir(pred_d, true_d).cpu().numpy()
                    nll_a = criterion_amt(pred_a, true_a).cpu().numpy()
                    nll_t = criterion_dt(pred_t, true_t).cpu().numpy()
                    
                    w_nll = nll_d + nll_a + nll_t
                    event_nll_accum[start_i:end_i] += w_nll
                    event_counts[start_i:end_i] += 1
                    
                mean_nll = event_nll_accum / np.maximum(event_counts, 1)
                surprise = float(np.mean(mean_nll))
                
                # Aggregations: top-k mean and sliding-window max
                if length >= top_k:
                    topk_surprise = float(np.mean(np.sort(mean_nll)[-top_k:]))
                else:
                    topk_surprise = surprise
                    
                if length <= rolling_window:
                    window_max_surprise = surprise
                else:
                    w_means = [float(np.mean(mean_nll[i : i + rolling_window])) for i in range(length - rolling_window + 1)]
                    window_max_surprise = float(np.max(w_means))
                
                # Mean embedding across windows
                avg_embed = np.mean(window_embeds, axis=0)
                embeddings_list.append(avg_embed)
                
                # Top-3 most surprising events
                top3_idx = np.argsort(-mean_nll)[:min(3, length)]
                top3_list = []
                for idx in top3_idx:
                    ts = pd.to_datetime(data['timestamps'][idx]).strftime('%Y-%m-%d %H:%M')
                    top3_list.append({
                        'timestamp': ts,
                        'amount': float(data['amounts'][idx]),
                        'direction': 'withdraw' if data['direction'][idx] == 1 else 'deposit',
                        'nll': round(float(mean_nll[idx]), 3)
                    })
                    
                if use_aggregation == 'window_max':
                    primary_surprise = window_max_surprise
                elif use_aggregation == 'topk':
                    primary_surprise = topk_surprise
                else:
                    primary_surprise = surprise
                    
                explanations[addr] = top3_list
                records.append({
                    'address': addr,
                    'transformer_surprise': primary_surprise,
                    'transformer_surprise_mean': surprise,
                    'transformer_surprise_topk': topk_surprise,
                    'transformer_surprise_window_max': window_max_surprise,
                    'insufficient_history': False,
                    'seq_length': length
                })
                
        df_scores = pd.DataFrame(records).set_index('address')
        embeddings_matrix = np.array(embeddings_list)
        return df_scores, explanations, embeddings_matrix

def cluster_embeddings(embeddings: np.ndarray, addresses: List[str], config: Config) -> Dict[str, Any]:
    """
    Clusters address sequence embeddings using both HDBSCAN (post-PCA)
    and KMeans, evaluating and comparing both via silhouette score.
    """
    logger.info("Clustering address sequence embeddings with HDBSCAN and KMeans...")
    n_samples = len(embeddings)
    if n_samples < 5:
        return {
            'hdbscan_clusters': {addr: 0 for addr in addresses},
            'kmeans_clusters': {addr: 0 for addr in addresses},
            'hdbscan_silhouette': 0.0,
            'kmeans_silhouette': 0.0,
            'hdbscan_n_clusters': 1,
            'kmeans_k': 1,
            'preferred': 'kmeans'
        }
        
    # PCA projection to 16 dims (or fewer if n_samples < 16)
    pca_dims = min(16, n_samples - 1, embeddings.shape[1])
    pca = PCA(n_components=pca_dims, random_state=config.seed)
    X_pca = pca.fit_transform(embeddings)
    
    # 1. HDBSCAN Clustering
    hdb = HDBSCAN(min_cluster_size=max(3, int(n_samples * 0.02)), min_samples=2)
    hdb_labels = hdb.fit_predict(X_pca)
    
    # Evaluate HDBSCAN silhouette on non-noise points
    non_noise_mask = hdb_labels != -1
    n_hdb_clusters = len(set(hdb_labels[non_noise_mask]))
    if n_hdb_clusters >= 2 and np.sum(non_noise_mask) > n_hdb_clusters:
        try:
            hdb_sil = float(silhouette_score(X_pca[non_noise_mask], hdb_labels[non_noise_mask]))
        except Exception:
            hdb_sil = 0.0
    else:
        hdb_sil = 0.0
        
    # 2. KMeans Clustering
    best_k = 2
    best_km_sil = -1.0
    max_k = min(10, n_samples - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=config.seed, n_init='auto')
        km_labels = km.fit_predict(X_pca)
        try:
            s = float(silhouette_score(X_pca, km_labels))
            if s > best_km_sil:
                best_km_sil = s
                best_k = k
        except Exception:
            pass
            
    final_km = KMeans(n_clusters=best_k, random_state=config.seed, n_init='auto')
    km_labels = final_km.fit_predict(X_pca)
    
    preferred = 'hdbscan' if (hdb_sil > best_km_sil and n_hdb_clusters >= 2) else 'kmeans'
    unique_hdb, counts_hdb = np.unique(hdb_labels, return_counts=True)
    hdb_sizes = {int(k): int(v) for k, v in zip(unique_hdb, counts_hdb)}
    
    unique_km, counts_km = np.unique(km_labels, return_counts=True)
    km_sizes = {int(k): int(v) for k, v in zip(unique_km, counts_km)}
    
    logger.info(
        f"Embedding Clustering (on {len(addresses)} addresses with >=3 events): "
        f"HDBSCAN clusters={n_hdb_clusters} (sil={hdb_sil:.4f}, noise={np.sum(~non_noise_mask)}, sizes={hdb_sizes}) | "
        f"KMeans k={best_k} (sil={best_km_sil:.4f}, sizes={km_sizes}) | Preferred: {preferred}"
    )
    
    return {
        'hdbscan_clusters': dict(zip(addresses, [int(l) for l in hdb_labels])),
        'kmeans_clusters': dict(zip(addresses, [int(l) for l in km_labels])),
        'hdbscan_cluster_sizes': hdb_sizes,
        'kmeans_cluster_sizes': km_sizes,
        'hdbscan_silhouette': hdb_sil,
        'kmeans_silhouette': float(best_km_sil),
        'hdbscan_n_clusters': n_hdb_clusters,
        'kmeans_k': best_k,
        'hdbscan_noise_count': int(np.sum(~non_noise_mask)),
        'n_evaluated': len(addresses),
        'preferred': preferred
    }

def compute_ensemble_scores(
    if_risk_scores: pd.Series,
    tf_surprise_df: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    """
    Computes ensemble risk scores by rank-averaging Isolation Forest risk score
    and Transformer surprise score. Addresses with insufficient history use IF score only.
    """
    df = pd.DataFrame(index=if_risk_scores.index)
    df['if_risk_score'] = if_risk_scores
    
    # Align transformer scores
    df['transformer_surprise'] = tf_surprise_df['transformer_surprise'].reindex(df.index).fillna(0.0)
    df['insufficient_history'] = tf_surprise_df['insufficient_history'].reindex(df.index).fillna(True)
    df['seq_length'] = tf_surprise_df['seq_length'].reindex(df.index).fillna(0).astype(int)
    
    # Percentile rank transformer surprise among addresses with sufficient history
    valid_mask = ~df['insufficient_history']
    df['transformer_risk_score'] = 0.0
    
    if valid_mask.sum() > 1:
        valid_surprises = df.loc[valid_mask, 'transformer_surprise'].values
        ranks = rankdata(valid_surprises, method='average')
        tf_ranks_scaled = (ranks - 1.0) / (len(ranks) - 1.0) * 100.0
        df.loc[valid_mask, 'transformer_risk_score'] = tf_ranks_scaled
    elif valid_mask.sum() == 1:
        df.loc[valid_mask, 'transformer_risk_score'] = 50.0
        
    # Ensemble combination based on config profile
    if getattr(config, 'profile', 'batch') == 'streaming':
        # Streaming profile: 3-Signal Fusion (0.20 IF + 0.10 TF Mean + 0.70 TF Window-5)
        # Compute percentile ranks for both mean and window_max
        tf_mean_scores = tf_surprise_df['transformer_surprise_mean'].reindex(df.index).fillna(0.0) if 'transformer_surprise_mean' in tf_surprise_df.columns else df['transformer_surprise']
        tf_win5_scores = tf_surprise_df['transformer_surprise_window_max'].reindex(df.index).fillna(0.0) if 'transformer_surprise_window_max' in tf_surprise_df.columns else df['transformer_surprise']
        
        tf_mean_risk = np.zeros(len(df))
        tf_win5_risk = np.zeros(len(df))
        if valid_mask.sum() > 1:
            r_m = rankdata(tf_mean_scores.loc[valid_mask].values, method='average')
            tf_mean_risk[valid_mask] = (r_m - 1.0) / (len(r_m) - 1.0) * 100.0
            r_w = rankdata(tf_win5_scores.loc[valid_mask].values, method='average')
            tf_win5_risk[valid_mask] = (r_w - 1.0) / (len(r_w) - 1.0) * 100.0
            
        ensemble_vals = np.where(
            df['insufficient_history'],
            df['if_risk_score'],
            0.20 * df['if_risk_score'] + 0.10 * tf_mean_risk + 0.70 * tf_win5_risk
        )
    else:
        # Batch profile: IF-led mean fusion (0.80 IF + 0.20 TF Mean)
        w_if = config.if_weight
        w_tf = config.transformer_weight
        total_w = w_if + w_tf
        w_if /= total_w
        w_tf /= total_w
        
        ensemble_vals = np.where(
            df['insufficient_history'],
            df['if_risk_score'],
            w_if * df['if_risk_score'] + w_tf * df['transformer_risk_score']
        )
    
    # Final rank-based calibration to 0-100
    n = len(ensemble_vals)
    if n > 1:
        ranks = rankdata(ensemble_vals, method='average')
        df['ensemble_risk_score'] = (ranks - 1.0) / (n - 1.0) * 100.0
    else:
        df['ensemble_risk_score'] = ensemble_vals
        
    return df
