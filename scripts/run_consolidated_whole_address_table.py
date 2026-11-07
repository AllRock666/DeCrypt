import sys
import json
from pathlib import Path
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from decrypt.config import Config
from decrypt.benchmark import (
    run_single_seed_benchmark,
    compute_bootstrap_ci,
    compute_paired_bootstrap_ci
)

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
archetypes = ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]

model_mapping = {
    'Legacy IF': 'Legacy IF',
    'Fixed IF': 'Fixed IF',
    'Transformer (Mean)': 'Transformer (Mean)',
    'Transformer (Window-5)': 'Transformer (Window-5)',
    'Batch Fusion (0.8/0.2 Mean)': 'Ensemble (Current 0.8/0.2 Mean)',
    'Streaming Fusion (3-Signal)': 'Ensemble (3-Signal 0.2/0.1/0.7)'
}

eval_models = list(model_mapping.keys())

print(f"Running Consolidated Whole-Address Benchmark across {len(seeds)} held-out seeds (42-51)...")
seed_runs = []
for s in seeds:
    print(f"--> Executing seed {s}...")
    cfg = Config(epochs=20, seed=s, random_state=s, contamination=0.1)
    res = run_single_seed_benchmark(df, seed=s, config=cfg, n_per_type=20, w_if=0.8, w_tf=0.2)
    seed_runs.append(res)

summary = {}
for display_name, internal_name in model_mapping.items():
    aurocs = [r['metrics'][internal_name]['auroc'] for r in seed_runs]
    aps = [r['metrics'][internal_name]['ap'] for r in seed_runs]
    recs = [r['metrics'][internal_name]['recall_at_k'] for r in seed_runs]
    
    arch_agg = {}
    for a in archetypes:
        vals = [r['metrics'][internal_name]['type_recalls'][a] for r in seed_runs]
        arch_agg[a] = compute_bootstrap_ci(vals)
        
    summary[display_name] = {
        'auroc': compute_bootstrap_ci(aurocs),
        'ap': compute_bootstrap_ci(aps),
        'recall_at_k': compute_bootstrap_ci(recs),
        'archetype_recalls': arch_agg,
        'raw_aurocs': aurocs,
        'raw_aps': aps,
        'raw_recs': recs
    }

# Paired differences vs Fixed IF
paired_vs_if = {}
for display_name in ['Batch Fusion (0.8/0.2 Mean)', 'Streaming Fusion (3-Signal)', 'Transformer (Mean)', 'Transformer (Window-5)', 'Legacy IF']:
    paired_vs_if[display_name] = {
        'auroc': compute_paired_bootstrap_ci(summary[display_name]['raw_aurocs'], summary['Fixed IF']['raw_aurocs']),
        'ap': compute_paired_bootstrap_ci(summary[display_name]['raw_aps'], summary['Fixed IF']['raw_aps']),
        'recall_at_k': compute_paired_bootstrap_ci(summary[display_name]['raw_recs'], summary['Fixed IF']['raw_recs']),
        'archetypes': {
            a: compute_paired_bootstrap_ci(
                [r['metrics'][model_mapping[display_name]]['type_recalls'][a] for r in seed_runs],
                [r['metrics']['Fixed IF']['type_recalls'][a] for r in seed_runs]
            ) for a in archetypes
        }
    }

# Paired differences vs Batch Fusion
paired_vs_batch = {}
for display_name in ['Streaming Fusion (3-Signal)', 'Fixed IF', 'Transformer (Window-5)']:
    paired_vs_batch[display_name] = {
        'auroc': compute_paired_bootstrap_ci(summary[display_name]['raw_aurocs'], summary['Batch Fusion (0.8/0.2 Mean)']['raw_aurocs']),
        'ap': compute_paired_bootstrap_ci(summary[display_name]['raw_aps'], summary['Batch Fusion (0.8/0.2 Mean)']['raw_aps']),
        'recall_at_k': compute_paired_bootstrap_ci(summary[display_name]['raw_recs'], summary['Batch Fusion (0.8/0.2 Mean)']['raw_recs']),
        'archetypes': {
            a: compute_paired_bootstrap_ci(
                [r['metrics'][model_mapping[display_name]]['type_recalls'][a] for r in seed_runs],
                [r['metrics']['Ensemble (Current 0.8/0.2 Mean)']['type_recalls'][a] for r in seed_runs]
            ) for a in archetypes
        }
    }

print("\n" + "="*120)
print("CONSOLIDATED WHOLE-ADDRESS V1.1 BENCHMARK TABLE (SEEDS 42-51, 95% CIs)")
print("="*120)

table_header = "| Model | AUROC [95% CI] | AP [95% CI] | Recall@K [95% CI] | Structuring | Burst | Alternating | Bot-Periodic | Dormant-Spike |"
print(table_header)
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for m in eval_models:
    s = summary[m]
    arch = s['archetype_recalls']
    print(
        f"| **{m}** | "
        f"{s['auroc']['mean']:.3f} [{s['auroc']['ci_lower']:.3f}, {s['auroc']['ci_upper']:.3f}] | "
        f"{s['ap']['mean']:.3f} [{s['ap']['ci_lower']:.3f}, {s['ap']['ci_upper']:.3f}] | "
        f"{s['recall_at_k']['mean']:.3f} [{s['recall_at_k']['ci_lower']:.3f}, {s['recall_at_k']['ci_upper']:.3f}] | "
        f"{arch['structuring']['mean']:.2f} [{arch['structuring']['ci_lower']:.2f}, {arch['structuring']['ci_upper']:.2f}] | "
        f"{arch['burst']['mean']:.2f} [{arch['burst']['ci_lower']:.2f}, {arch['burst']['ci_upper']:.2f}] | "
        f"{arch['pass_through']['mean']:.2f} [{arch['pass_through']['ci_lower']:.2f}, {arch['pass_through']['ci_upper']:.2f}] | "
        f"{arch['bot_periodic']['mean']:.2f} [{arch['bot_periodic']['ci_lower']:.2f}, {arch['bot_periodic']['ci_upper']:.2f}] | "
        f"{arch['dormant_spike']['mean']:.2f} [{arch['dormant_spike']['ci_lower']:.2f}, {arch['dormant_spike']['ci_upper']:.2f}] |"
    )

print("\n" + "="*120)
print("PAIRED DIFFERENCES VS FIXED IF")
print("="*120)
for m in ['Batch Fusion (0.8/0.2 Mean)', 'Streaming Fusion (3-Signal)']:
    p = paired_vs_if[m]
    print(f"\n--- {m} vs Fixed IF ---")
    print(f"  AUROC Diff:     {p['auroc']['mean']:+.4f} [{p['auroc']['ci_lower']:+.4f}, {p['auroc']['ci_upper']:+.4f}]")
    print(f"  AP Diff:        {p['ap']['mean']:+.4f} [{p['ap']['ci_lower']:+.4f}, {p['ap']['ci_upper']:+.4f}]")
    print(f"  Recall@K Diff:  {p['recall_at_k']['mean']:+.4f} [{p['recall_at_k']['ci_lower']:+.4f}, {p['recall_at_k']['ci_upper']:+.4f}]")
    print("  Archetype Diffs:")
    for a in archetypes:
        ad = p['archetypes'][a]
        print(f"    {a:<15}: {ad['mean']:+.3f} [{ad['ci_lower']:+.3f}, {ad['ci_upper']:+.3f}]")

print("\n" + "="*120)
print("PAIRED DIFFERENCES VS BATCH FUSION (0.8/0.2 MEAN)")
print("="*120)
for m in ['Streaming Fusion (3-Signal)']:
    p = paired_vs_batch[m]
    print(f"\n--- {m} vs Batch Fusion (0.8/0.2 Mean) ---")
    print(f"  AUROC Diff:     {p['auroc']['mean']:+.4f} [{p['auroc']['ci_lower']:+.4f}, {p['auroc']['ci_upper']:+.4f}]")
    print(f"  AP Diff:        {p['ap']['mean']:+.4f} [{p['ap']['ci_lower']:+.4f}, {p['ap']['ci_upper']:+.4f}]")
    print(f"  Recall@K Diff:  {p['recall_at_k']['mean']:+.4f} [{p['recall_at_k']['ci_lower']:+.4f}, {p['recall_at_k']['ci_upper']:+.4f}]")
    print("  Archetype Diffs:")
    for a in archetypes:
        ad = p['archetypes'][a]
        print(f"    {a:<15}: {ad['mean']:+.3f} [{ad['ci_lower']:+.3f}, {ad['ci_upper']:+.3f}]")

# Save to output/consolidated_v1_1_benchmark.json
out_data = {
    'description': 'Consolidated v1.1 whole-address synthetic benchmark across seeds 42-51 with 95% bootstrap CIs',
    'seeds': seeds,
    'models': summary,
    'paired_vs_if': paired_vs_if,
    'paired_vs_batch': paired_vs_batch
}
with open('output/consolidated_v1_1_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2)
print("\nSaved consolidated whole-address benchmark results to output/consolidated_v1_1_benchmark.json")
