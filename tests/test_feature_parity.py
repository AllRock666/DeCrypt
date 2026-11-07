"""
Parity test verifying that batch behavioral feature extraction
(decrypt.features.extract_behavioral_features) and incremental feature extraction
(decrypt.pipeline.incremental.IncrementalFeatureExtractor) produce identical
feature values across 20 real entities, including entities with >128 events.
"""

from datetime import timezone
import numpy as np
import pandas as pd
import pytest

from decrypt.config import Config
from decrypt.data import load_transaction_data
from decrypt.features import extract_behavioral_features
from decrypt.pipeline.incremental import IncrementalFeatureExtractor
from decrypt.domain.schema import UnifiedTransaction


def test_batch_vs_incremental_feature_parity_20_real_entities():
    """
    Loads real Bitcoin transaction data, extracts batch features across all addresses,
    and asserts numerical parity (atol=1e-4, rtol=1e-4) with IncrementalFeatureExtractor
    across 20 real entities, including entities with >128 transactions.
    """
    config = Config()
    df = load_transaction_data("data/bitcoin_transactions.csv", config)
    batch_features = extract_behavioral_features(df, config)

    # Amount quantile bins from dataset
    abs_amt = df["amount"].abs()
    _, q_bins = pd.qcut(abs_amt, q=10, retbins=True, duplicates="drop")

    # Incremental extractor (entity-relative, unbounded history mode)
    extractor = IncrementalFeatureExtractor(
        amount_bins=q_bins,
        large_tx_multiplier=config.large_tx_multiplier
    )

    # Select 20 real entities, explicitly including top 5 entities with >128 events
    counts = df["address_id"].value_counts()
    top_entities_over_128 = list(counts.head(5).index)  # 960, 655, 322, 311, 204 events
    other_entities = list(counts.iloc[10:25].index)
    selected_entities = top_entities_over_128 + other_entities
    assert len(selected_entities) == 20

    # Ensure >128 event condition is met
    for addr in top_entities_over_128:
        assert counts[addr] > 128

    mismatches = []
    comparisons_count = 0

    for addr in selected_entities:
        sub_df = df[df["address_id"] == addr].sort_values("timestamp")
        txs = [
            UnifiedTransaction.from_legacy_csv_row(
                address_id=row.address_id,
                tx_hash=getattr(row, "hash", f"h_{idx}"),
                timestamp_str=str(row.timestamp),
                tx_type_raw=row.type,
                amount_raw=row.amount
            )
            for idx, row in enumerate(sub_df.itertuples(index=False))
        ]

        inc_features = extractor.compute_features(txs)
        batch_row = batch_features.loc[addr].to_dict()

        for feat_name, inc_val in inc_features.items():
            if feat_name in batch_row:
                batch_val = batch_row[feat_name]
                comparisons_count += 1
                if not np.isclose(inc_val, batch_val, atol=1e-4, rtol=1e-4):
                    mismatches.append(
                        f"Entity {addr} (n={len(txs)}): feature '{feat_name}' differs: "
                        f"inc={inc_val:.6f} vs batch={batch_val:.6f}"
                    )

    assert len(mismatches) == 0, f"Found {len(mismatches)} feature parity mismatches:\n" + "\n".join(mismatches[:10])
    # 20 entities * ~29 features = ~580 comparisons
    assert comparisons_count >= 500
