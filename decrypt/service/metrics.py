"""
Operational Metrics Collector for DeCrypt Ingestion Service.
"""

from datetime import datetime, timezone
import time
from typing import Dict, Any


class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.transactions_received_total = 0
        self.transactions_ingested_total = 0
        self.duplicates_detected_total = 0
        self.ingestion_failures_total = 0
        self.webhooks_received_total = 0
        self.webhook_failures_total = 0
        self.high_risk_entities_total = 0

        # Latency tracking
        self.inference_latencies_ms = []

    def record_transaction(self, newly_inserted: bool):
        self.transactions_received_total += 1
        if newly_inserted:
            self.transactions_ingested_total += 1
        else:
            self.duplicates_detected_total += 1

    def record_failure(self):
        self.ingestion_failures_total += 1

    def record_webhook(self, verified: bool):
        self.webhooks_received_total += 1
        if not verified:
            self.webhook_failures_total += 1

    def record_inference_latency(self, latency_ms: float):
        self.inference_latencies_ms.append(latency_ms)
        if len(self.inference_latencies_ms) > 1000:
            self.inference_latencies_ms = self.inference_latencies_ms[-500:]

    def record_high_risk_alert(self):
        self.high_risk_entities_total += 1

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        uptime_sec = time.time() - self.start_time
        tx_per_min = (self.transactions_received_total / (uptime_sec / 60.0)) if uptime_sec > 0 else 0.0
        avg_latency = float(sum(self.inference_latencies_ms) / len(self.inference_latencies_ms)) if self.inference_latencies_ms else 0.0

        return {
            "uptime_seconds": round(uptime_sec, 2),
            "transactions_received_total": self.transactions_received_total,
            "transactions_ingested_total": self.transactions_ingested_total,
            "duplicates_detected_total": self.duplicates_detected_total,
            "ingestion_failures_total": self.ingestion_failures_total,
            "duplicate_rate_pct": round((self.duplicates_detected_total / max(1, self.transactions_received_total)) * 100.0, 2),
            "transactions_per_minute": round(tx_per_min, 2),
            "webhooks_received_total": self.webhooks_received_total,
            "webhook_failures_total": self.webhook_failures_total,
            "high_risk_entities_flagged": self.high_risk_entities_total,
            "avg_inference_latency_ms": round(avg_latency, 2)
        }


metrics = MetricsCollector()
