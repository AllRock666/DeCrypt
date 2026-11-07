"""
Real-Time Model Inference, Calibration, and Explainability Engine for DeCrypt.

Provides low-latency scoring for streaming transactions, combining cached Isolation Forest
tabular deviance with Transformer sequence surprise via empirical CDF calibration.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from decrypt.domain.schema import UnifiedTransaction, TransactionType
from decrypt.config import Config, default_config
from decrypt.sequence import EventTransformerModel, SequenceAnomalyDetector
from .incremental import IncrementalFeatureExtractor

logger = logging.getLogger(__name__)


class RealTimeScorer:
    """
    Real-time inference and calibration engine for incoming transactions.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_dir: Path = Path("output"),
        w_if: float = 0.8,
        w_tf: float = 0.2
    ):
        self.config = config or default_config
        self.model_dir = Path(model_dir)
        self.w_if = w_if
        self.w_tf = w_tf
        self.feature_extractor = IncrementalFeatureExtractor(
            large_tx_multiplier=self.config.large_tx_multiplier
        )

        self.scaler: Optional[StandardScaler] = None
        self.if_model: Optional[IsolationForest] = None
        self.feature_names: List[str] = []
        self.if_reference_scores: np.ndarray = np.array([])
        self.tf_reference_surprises: np.ndarray = np.array([])

        # Training domain & source guard configuration
        self.training_domain: str = "crypto_bitcoin"
        self.training_asset: str = "BTC"
        self.model_version: str = "v2.0-realtime-ensemble"
        self.per_source_calibration: Dict[str, Dict[str, np.ndarray]] = {}

        # Sequence Transformer components
        self.tf_model: Optional[EventTransformerModel] = None
        self.amount_bins: np.ndarray = np.array([])
        self.dt_bins: np.ndarray = np.array([])
        self.device = torch.device("cpu")

        self._load_or_initialize_models()

    def register_source_calibration(
        self,
        source: str,
        if_reference_scores: np.ndarray,
        tf_reference_surprises: np.ndarray
    ):
        """
        Registers an empirical reference distribution calibrated for an alternative source.
        """
        src = str(source).lower().replace("transactionsource.", "")
        self.per_source_calibration[src] = {
            "if_ref": np.sort(np.asarray(if_reference_scores)),
            "tf_ref": np.sort(np.asarray(tf_reference_surprises))
        }
        logger.info(f"Registered fitted calibration reference distribution for source: {src}")

    def is_source_calibrated(self, source: str) -> bool:
        """
        Returns True if the source matches the model's training domain or has a fitted reference distribution.
        """
        src = str(source).lower().replace("transactionsource.", "")
        if src in [self.training_domain, "crypto_bitcoin", "csv_import", "internal"]:
            return True
        return src in self.per_source_calibration

    def _load_or_initialize_models(self):
        """
        Loads pre-trained models from output directory or fits baseline if missing.
        """
        bins_path = self.model_dir / "transformer_bins.json"
        model_path = self.model_dir / "transformer_model.pt"
        risk_csv = self.model_dir / "risk_scores.csv"

        # 1. Load Transformer bins and weights
        if bins_path.exists() and model_path.exists():
            try:
                with open(bins_path, "r") as f:
                    bins_data = json.load(f)
                    self.amount_bins = np.array(bins_data.get("amount_bins", []))
                    self.dt_bins = np.array(bins_data.get("dt_bins", []))

                self.tf_model = EventTransformerModel(
                    d_model=self.config.d_model,
                    n_heads=self.config.n_heads,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout
                )
                state_dict = torch.load(model_path, map_location=self.device)
                self.tf_model.load_state_dict(state_dict)
                self.tf_model.eval()
                logger.info(f"Loaded Sequence Transformer from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load pre-trained Transformer ({e}); initializing default.")
                self.tf_model = None

        # 2. Load reference scores for percentile calibration from risk_scores.csv if available
        if risk_csv.exists():
            try:
                df_risk = pd.read_csv(risk_csv)
                if "if_risk_score" in df_risk.columns:
                    self.if_reference_scores = np.sort(df_risk["if_risk_score"].dropna().values)
                if "transformer_surprise" in df_risk.columns:
                    self.tf_reference_surprises = np.sort(
                        df_risk[~df_risk.get("insufficient_history", False)]["transformer_surprise"].dropna().values
                    )
                logger.info(f"Loaded {len(self.if_reference_scores)} calibration reference scores.")
            except Exception as e:
                logger.warning(f"Could not load reference scores ({e}).")

        # 3. Model metadata domain tracking
        metadata_path = self.model_dir / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    self.training_domain = meta.get("training_domain", self.training_domain)
                    self.training_asset = meta.get("training_asset", self.training_asset)
            except Exception as e:
                logger.warning(f"Could not load model metadata: {e}")
        else:
            try:
                self.model_dir.mkdir(parents=True, exist_ok=True)
                with open(metadata_path, "w") as f:
                    json.dump({
                        "training_domain": self.training_domain,
                        "training_asset": self.training_asset,
                        "model_version": self.model_version,
                        "calibrated_sources": [self.training_domain, "csv_import", "internal"]
                    }, f, indent=2)
            except Exception as e:
                logger.debug(f"Could not persist model metadata: {e}")

    def fit_baseline_from_dataframe(self, df: pd.DataFrame):
        """
        Fits tabular Scaler and Isolation Forest on an empirical baseline dataset.
        """
        from decrypt.features import extract_all_features, prepare_numeric_features

        logger.info("Fitting RealTimeScorer baseline on historical dataframe...")
        feats, _ = extract_all_features(df, self.config)
        clean_num = prepare_numeric_features(feats)
        self.feature_names = list(clean_num.columns)

        self.scaler = StandardScaler().fit(clean_num)
        X_scaled = self.scaler.transform(clean_num)

        self.if_model = IsolationForest(
            contamination=self.config.contamination,
            random_state=self.config.random_state
        ).fit(X_scaled)

        raw_scores = -self.if_model.score_samples(X_scaled)
        self.if_reference_scores = np.sort(raw_scores)
        logger.info(f"Fitted IF baseline on {len(df)} transactions across {len(clean_num)} entities.")

    def calibrate_percentile(self, score: float, reference_scores: np.ndarray) -> float:
        """
        Maps a continuous score into a calibrated percentile rank [0.0, 100.0]
        against empirical reference distribution, avoiding single-item rankdata distortion.
        """
        if len(reference_scores) == 0:
            return float(np.clip(score * 10.0, 0.0, 100.0))
        pct = float(np.searchsorted(reference_scores, score, side="right") / len(reference_scores) * 100.0)
        return float(np.clip(pct, 0.0, 100.0))

    def score_entity(
        self,
        entity_id: str,
        history: List[UnifiedTransaction]
    ) -> Dict[str, Any]:
        """
        Scores an entity's complete history window in real time.
        """
        if not history:
            return {
                "entity_id": entity_id,
                "status": "empty_history",
                "ensemble_risk_score": 0.0,
                "if_risk_score": 0.0,
                "transformer_surprise": 0.0,
                "percentile_rank": 0.0,
                "insufficient_history": True,
                "top_features": [],
                "top_events": [],
                "explanation": "No transaction history recorded.",
                "disclaimer": "Anomaly scores represent statistical deviation and do not constitute verified fraud."
            }

        # Source Guard: ensure transaction source matches calibrated domain
        primary_tx = history[-1]
        primary_source = primary_tx.source.value if hasattr(primary_tx.source, "value") else str(primary_tx.source)
        if not self.is_source_calibrated(primary_source):
            return {
                "entity_id": entity_id,
                "status": "not calibrated for this source",
                "source": primary_source,
                "training_domain": self.training_domain,
                "ensemble_risk_score": None,
                "if_risk_score": None,
                "transformer_surprise": None,
                "percentile_rank": None,
                "insufficient_history": True,
                "top_features": [],
                "top_events": [],
                "explanation": (
                    f"Model artifact was trained and calibrated on {self.training_domain} ({self.training_asset}). "
                    f"Transactions from source '{primary_source}' require a fitted per-source reference distribution before scoring."
                ),
                "model_version": self.model_version,
                "disclaimer": "Uncalibrated source: domain-specific reference distribution required before anomaly scoring."
            }

        src_key = str(primary_source).lower().replace("transactionsource.", "")
        if src_key in self.per_source_calibration:
            active_if_ref = self.per_source_calibration[src_key]["if_ref"]
            active_tf_ref = self.per_source_calibration[src_key]["tf_ref"]
        else:
            active_if_ref = self.if_reference_scores
            active_tf_ref = self.tf_reference_surprises

        n_events = len(history)

        # 1. Tabular Behavioral Features
        raw_feat_dict = self.feature_extractor.compute_features(history)

        if self.scaler is None or self.if_model is None:
            # Fallback heuristic if models not yet fit
            vol = raw_feat_dict.get("total_volume", 0.0)
            if_pct = float(np.clip(np.log1p(vol) * 10.0, 5.0, 95.0))
            if_raw = vol
            top_features = [{"feature": "total_volume", "value": vol, "deviation": 1.0}]
        else:
            feat_vec = np.array([raw_feat_dict.get(col, 0.0) for col in self.feature_names], dtype=np.float64).reshape(1, -1)
            scaled_vec = self.scaler.transform(feat_vec)
            if_raw = float(-self.if_model.score_samples(scaled_vec)[0])
            if_pct = self.calibrate_percentile(if_raw, active_if_ref)

            # Feature Attribution (SHAP-style deviation from mean)
            deviations = np.abs(scaled_vec[0])
            top_indices = np.argsort(-deviations)[:3]
            top_features = [
                {
                    "feature": self.feature_names[idx],
                    "value": float(raw_feat_dict.get(self.feature_names[idx], 0.0)),
                    "z_score": float(scaled_vec[0][idx])
                }
                for idx in top_indices
            ]

        # 2. Sequence Transformer Surprise Scoring
        top_events: List[Dict[str, Any]] = []
        if n_events < self.config.min_seq_len or self.tf_model is None:
            insufficient_history = True
            tf_surprise = 0.0
            tf_pct = if_pct  # Defaults to IF
            ensemble_score = if_pct
        else:
            insufficient_history = False
            tf_surprise, top_events = self._compute_sequence_surprise(history)
            tf_pct = self.calibrate_percentile(tf_surprise, active_tf_ref)
            # Tuned Weighted Ensemble: 0.8 * IF + 0.2 * TF
            ensemble_score = float(self.w_if * if_pct + self.w_tf * tf_pct)

        ensemble_score = float(np.clip(ensemble_score, 0.0, 100.0))

        # Narrative Explanation
        explanation = self._build_narrative(
            entity_id=entity_id,
            ensemble_score=ensemble_score,
            if_score=if_pct,
            tf_surprise=tf_surprise,
            insufficient_history=insufficient_history,
            top_features=top_features,
            top_events=top_events
        )

        return {
            "entity_id": entity_id,
            "status": "success",
            "seq_length": n_events,
            "ensemble_risk_score": round(ensemble_score, 2),
            "if_risk_score": round(if_pct, 2),
            "transformer_surprise": round(tf_surprise, 3),
            "percentile_rank": round(ensemble_score, 2),
            "insufficient_history": insufficient_history,
            "top_features": top_features,
            "top_events": top_events,
            "explanation": explanation,
            "model_version": self.model_version,
            "disclaimer": "Statistical Anomaly Detection Notice: High risk scores indicate rare or extreme behavioral deviations from population baselines and warrant compliance triage; they do not constitute legal or confirmed classification of fraud."
        }

    def _compute_sequence_surprise(
        self,
        history: List[UnifiedTransaction]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluates leave-one-out event surprise over the entity history using the PyTorch Transformer.
        """
        n = min(len(history), self.config.max_seq_len)
        recent_txs = history[-n:]

        # Tokenize sequence
        directions = []
        amt_bins = []
        amt_conts = []
        dt_bins = []
        dt_conts = []
        hours = []
        dows = []

        for i, tx in enumerate(recent_txs):
            # Direction: 0 for deposit, 1 for withdraw
            d = 0 if tx.transaction_type == TransactionType.DEPOSIT else 1
            directions.append(d)

            # Amount
            amt_val = float(tx.amount)
            amt_conts.append(np.log1p(amt_val))
            if len(self.amount_bins) > 0:
                b_amt = int(np.searchsorted(self.amount_bins, amt_val, side="right"))
            else:
                b_amt = min(31, int(np.log1p(amt_val)))
            amt_bins.append(b_amt)

            # Delta-t
            if i == 0:
                dt_sec = 0.0
            else:
                dt_sec = max(0.0, (tx.timestamp - recent_txs[i-1].timestamp).total_seconds())
            dt_conts.append(np.log1p(dt_sec))
            if len(self.dt_bins) > 0:
                b_dt = int(np.searchsorted(self.dt_bins, dt_sec, side="right"))
            else:
                b_dt = min(31, int(np.log1p(dt_sec)))
            dt_bins.append(b_dt)

            hours.append(tx.timestamp.hour)
            dows.append(tx.timestamp.weekday())

        seq_len = len(directions)
        event_surprises: List[Dict[str, Any]] = []

        # Convert to tensors
        dir_t = torch.tensor([directions], dtype=torch.long, device=self.device)
        amt_b_t = torch.tensor([amt_bins], dtype=torch.long, device=self.device)
        amt_c_t = torch.tensor([amt_conts], dtype=torch.float, device=self.device)
        dt_b_t = torch.tensor([dt_bins], dtype=torch.long, device=self.device)
        dt_c_t = torch.tensor([dt_conts], dtype=torch.float, device=self.device)
        hour_t = torch.tensor([hours], dtype=torch.long, device=self.device)
        dow_t = torch.tensor([dows], dtype=torch.long, device=self.device)
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            hidden, dir_logits, amt_logits, dt_logits = self.tf_model(
                dir_t,
                amt_b_t,
                dt_b_t,
                hour_t,
                dow_t,
                amt_c_t,
                dt_c_t,
                mask_padding=padding_mask
            )

            # Cross-entropy loss per event
            log_sm_dir = torch.log_softmax(dir_logits[0], dim=-1)
            log_sm_amt = torch.log_softmax(amt_logits[0], dim=-1)
            log_sm_dt = torch.log_softmax(dt_logits[0], dim=-1)

            nll_totals = []
            for i in range(seq_len):
                loss_d = -log_sm_dir[i, directions[i]].item()
                loss_a = -log_sm_amt[i, min(amt_bins[i], amt_logits.size(-1) - 1)].item()
                loss_t = -log_sm_dt[i, min(dt_bins[i], dt_logits.size(-1) - 1)].item()
                total_loss = loss_d + loss_a + loss_t
                nll_totals.append(total_loss)

                tx_obj = recent_txs[i]
                event_surprises.append({
                    "timestamp": tx_obj.timestamp.isoformat(),
                    "direction": tx_obj.transaction_type.value,
                    "amount": float(tx_obj.amount),
                    "asset": tx_obj.asset,
                    "nll": round(total_loss, 2)
                })

        mean_surprise = float(np.mean(nll_totals)) if nll_totals else 0.0

        # Sort for top-3 most surprising events
        event_surprises.sort(key=lambda ev: ev["nll"], reverse=True)
        top_3 = event_surprises[:3]

        return mean_surprise, top_3

    def _build_narrative(
        self,
        entity_id: str,
        ensemble_score: float,
        if_score: float,
        tf_surprise: float,
        insufficient_history: bool,
        top_features: List[Dict[str, Any]],
        top_events: List[Dict[str, Any]]
    ) -> str:
        """Generates clear, interpretable audit prose for analysts."""
        if ensemble_score >= 90.0:
            severity = "CRITICAL / HIGH ANOMALY"
        elif ensemble_score >= 70.0:
            severity = "ELEVATED RISK"
        else:
            severity = "NORMAL / LOW RISK"

        feat_strs = [f"{f['feature']} (z={f.get('z_score', 0):+.2f})" for f in top_features[:3]]
        feat_desc = ", ".join(feat_strs) if feat_strs else "standard parameters"

        narrative = (
            f"Entity {entity_id[:16]}... evaluated as {severity} (Score: {ensemble_score:.1f}/100, IF: {if_score:.1f}). "
            f"Key driving behavioral factors: {feat_desc}. "
        )
        if insufficient_history:
            narrative += "Entity possesses fewer than 3 transactions; evaluated via tabular profile only."
        else:
            narrative += f"Sequence Transformer surprise: {tf_surprise:.2f} NLL. "
            if top_events:
                ev = top_events[0]
                narrative += f"Most unusual event: {ev['direction'].upper()} {ev['amount']:.2f} {ev['asset']} (NLL={ev['nll']}) at {ev['timestamp']}."

        return narrative
