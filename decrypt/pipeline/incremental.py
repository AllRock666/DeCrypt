"""
Incremental Feature Extractor and State Manager for DeCrypt.

Maintains rolling entity statistics across incoming transactions, allowing O(1)
to O(W) feature computation for real-time streaming without requiring global
DataFrame groupby aggregations over the entire historical database.
"""

from typing import Dict, Any, List, Optional, Sequence
from decimal import Decimal
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from decrypt.domain.schema import UnifiedTransaction, TransactionType


class EntityStateManager:
    """
    In-memory / cache manager tracking entity transaction history windows.
    Defaults to capping per-entity in-memory history at 5000 events to balance
    memory usage with behavioral fidelity. Set max_history_per_entity=None for
    unbounded mode (e.g. offline parity testing).
    """

    def __init__(self, max_history_per_entity: Optional[int] = 5000):
        self.max_history = max_history_per_entity
        # Map: entity_id -> list of UnifiedTransaction sorted by timestamp
        self._entity_histories: Dict[str, List[UnifiedTransaction]] = {}

    def get_history(self, entity_id: str) -> List[UnifiedTransaction]:
        return self._entity_histories.get(entity_id, [])

    def add_transaction(self, tx: UnifiedTransaction) -> List[UnifiedTransaction]:
        """
        Appends transaction to entity history, maintaining chronological order.
        If max_history is set, enforces bounded length.
        """
        history = self._entity_histories.setdefault(tx.entity_id, [])
        # Insert maintaining time order
        if not history or tx.timestamp >= history[-1].timestamp:
            history.append(tx)
        else:
            # Insertion in middle (rare for live stream, common in backfill)
            history.append(tx)
            history.sort(key=lambda t: t.timestamp)

        # Enforce bounded window if specified
        if self.max_history and self.max_history > 0 and len(history) > self.max_history:
            self._entity_histories[tx.entity_id] = history[-self.max_history:]

        return self._entity_histories[tx.entity_id]

    def load_entity_history(self, entity_id: str, transactions: List[UnifiedTransaction]):
        """Populates history from persistent storage."""
        sorted_txs = sorted(transactions, key=lambda t: t.timestamp)
        if self.max_history and self.max_history > 0:
            self._entity_histories[entity_id] = sorted_txs[-self.max_history:]
        else:
            self._entity_histories[entity_id] = sorted_txs

    def clear(self):
        self._entity_histories.clear()


class IncrementalFeatureExtractor:
    """
    Computes canonical 23 tabular features from an entity's transaction history window.
    Strictly aligns with `decrypt.features.extract_behavioral_features`.
    """

    def __init__(
        self,
        amount_bins: Optional[Sequence[float]] = None,
        large_tx_multiplier: float = 3.0
    ):
        self.amount_bins = list(amount_bins) if amount_bins is not None else None
        self.large_tx_multiplier = large_tx_multiplier

    def compute_features(self, history: List[UnifiedTransaction]) -> Dict[str, float]:
        """
        Extracts identical 28 features as extract_behavioral_features for a single entity history.
        Strictly entity-relative, avoiding absolute calendar timestamp dependencies.
        """
        if not history:
            return self._empty_features()

        n_txs = len(history)
        amounts = np.array([float(tx.amount) for tx in history], dtype=np.float64)
        types = np.array([1 if tx.transaction_type == TransactionType.DEPOSIT else -1 for tx in history], dtype=np.int32)
        timestamps = [tx.timestamp for tx in history]

        # Signed amounts (deposits positive, withdrawals negative)
        signed_amounts = amounts * types
        abs_amounts = amounts

        # 1. Volume & counts
        total_transactions = float(n_txs)
        total_volume = float(np.sum(abs_amounts))
        net_flow = float(np.sum(signed_amounts))
        avg_transaction_size = float(np.mean(abs_amounts))
        max_transaction = float(np.max(abs_amounts))
        min_transaction = float(np.min(abs_amounts))
        transaction_variance = float(np.var(abs_amounts, ddof=1)) if n_txs > 1 else 0.0

        # 2. Deposit vs Withdrawal flow
        dep_mask = (types == 1)
        wd_mask = (types == -1)
        total_deposited = float(np.sum(abs_amounts[dep_mask])) if np.any(dep_mask) else 0.0
        total_withdrawn = float(np.sum(abs_amounts[wd_mask])) if np.any(wd_mask) else 0.0
        n_deposits = float(np.sum(dep_mask))
        n_withdrawals = float(np.sum(wd_mask))
        deposit_withdraw_ratio = float(np.mean(types))

        max_flow = max(total_deposited, total_withdrawn)
        pass_through_ratio = min(total_deposited, total_withdrawn) / max_flow if max_flow > 0 else 0.0

        # 3. Timestamps & Recency (strictly entity-relative)
        first_tx = timestamps[0]
        last_tx = timestamps[-1]
        active_span_sec = (last_tx - first_tx).total_seconds()
        activity_period_days = max(0.0, active_span_sec / 86400.0)

        # Entity-relative activity centroid: mean time offset from first_tx normalized by active span [0.0, 1.0]
        if active_span_sec > 0 and n_txs > 1:
            offsets = np.array([(t - first_tx).total_seconds() for t in timestamps])
            activity_centroid = float(np.mean(offsets) / active_span_sec)
        else:
            activity_centroid = 0.5

        # 4. Temporal dynamics & Burstiness
        if n_txs > 1:
            delta_ts = np.array([(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, n_txs)])
            mu = float(np.mean(delta_ts))
            sigma = float(np.std(delta_ts, ddof=1)) if len(delta_ts) > 1 else 0.0
            denom = sigma + mu
            burstiness = float(np.clip((sigma - mu) / denom if denom > 0 else 0.0, -1.0, 1.0))
            avg_time_between_txs = mu
            tx_time_variance = float(np.var(delta_ts, ddof=1)) if len(delta_ts) > 1 else 0.0
            cv_inter_event_time = sigma / mu if mu > 0 else 0.0

            # Lag-1 Autocorrelation
            if len(delta_ts) >= 3 and np.var(delta_ts) > 0:
                s = pd.Series(delta_ts)
                ac = s.autocorr(lag=1)
                lag1_autocorr = float(ac) if not np.isnan(ac) else 0.0
            else:
                lag1_autocorr = 0.0
        else:
            burstiness = 0.0
            avg_time_between_txs = 0.0
            tx_time_variance = 0.0
            cv_inter_event_time = 0.0
            lag1_autocorr = 0.0

        # 5. Modal-Amount Share (unrounded exact match to batch)
        vals, counts = np.unique(abs_amounts, return_counts=True)
        modal_amount_share = float(np.max(counts) / n_txs) if len(counts) > 0 else 0.0

        # 6. Amount Entropy (over fixed quantile bins if provided, else unique vals)
        if self.amount_bins is not None and len(self.amount_bins) > 1:
            binned = pd.cut(abs_amounts, bins=self.amount_bins, labels=False, include_lowest=True)
            bin_counts = pd.Series(binned).value_counts()
            probs = bin_counts / bin_counts.sum()
            log2_p = np.where(probs > 0, np.log2(probs), 0.0)
            amount_entropy = -float(np.sum(probs * log2_p))
        elif len(vals) > 1:
            probs = counts / counts.sum()
            with np.errstate(divide='ignore'):
                log2_p = np.where(probs > 0, np.log2(probs), 0.0)
                amount_entropy = float(-np.sum(probs * log2_p))
        else:
            amount_entropy = 0.0

        # 7. Behavioral Ratios
        large_thresh = avg_transaction_size * self.large_tx_multiplier
        large_transaction_ratio = float(np.mean(abs_amounts > large_thresh))
        weekend_ratio = float(np.mean([t.weekday() >= 5 for t in timestamps]))
        unique_interactions = float(len(set(tx.transaction_hash for tx in history)))

        # 8. Direction Alternation Dynamics
        if n_txs > 1:
            switches = types[1:] != types[:-1]
            direction_switch_count = float(np.sum(switches))
            alternation_rate = float(direction_switch_count / max(1.0, float(n_txs - 1)))

            # Max alternating run length
            max_run = 0
            curr_run = 0
            for s in switches:
                if s:
                    curr_run += 1
                    if curr_run > max_run:
                        max_run = curr_run
                else:
                    curr_run = 0
            max_alternating_run_length = float(max_run + 1 if max_run > 0 else 0)
        else:
            direction_switch_count = 0.0
            alternation_rate = 0.0
            max_alternating_run_length = 0.0

        return {
            "total_transactions": total_transactions,
            "total_volume": total_volume,
            "net_flow": net_flow,
            "avg_transaction_size": avg_transaction_size,
            "max_transaction": max_transaction,
            "min_transaction": min_transaction,
            "transaction_variance": transaction_variance,
            "total_deposited": total_deposited,
            "total_withdrawn": total_withdrawn,
            "n_deposits": n_deposits,
            "n_withdrawals": n_withdrawals,
            "deposit_withdraw_ratio": deposit_withdraw_ratio,
            "pass_through_ratio": pass_through_ratio,
            "activity_period_days": activity_period_days,
            "activity_centroid": activity_centroid,
            "burstiness": burstiness,
            "avg_time_between_txs": avg_time_between_txs,
            "tx_time_variance": tx_time_variance,
            "cv_inter_event_time": cv_inter_event_time,
            "lag1_autocorr_inter_event_time": lag1_autocorr,
            "modal_amount_share": modal_amount_share,
            "amount_entropy": amount_entropy,
            "large_transaction_ratio": large_transaction_ratio,
            "weekend_ratio": weekend_ratio,
            "unique_interactions": unique_interactions,
            "direction_switch_count": direction_switch_count,
            "alternation_rate": alternation_rate,
            "max_alternating_run_length": max_alternating_run_length
        }

    def _empty_features(self) -> Dict[str, float]:
        feature_names = [
            "total_transactions", "total_volume", "net_flow", "avg_transaction_size",
            "max_transaction", "min_transaction", "transaction_variance", "total_deposited",
            "total_withdrawn", "n_deposits", "n_withdrawals", "deposit_withdraw_ratio",
            "pass_through_ratio", "activity_period_days", "activity_centroid",
            "burstiness", "avg_time_between_txs", "tx_time_variance", "cv_inter_event_time",
            "lag1_autocorr_inter_event_time", "modal_amount_share", "amount_entropy",
            "large_transaction_ratio", "weekend_ratio", "unique_interactions",
            "direction_switch_count", "alternation_rate", "max_alternating_run_length"
        ]
        return {f: 0.0 for f in feature_names}
