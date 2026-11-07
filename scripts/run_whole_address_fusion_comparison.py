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
    compute_paired_bootstrap_ci,
    compute_hierarchical_paired_ci,
    compute_hierarchical_bootstrap_ci
)

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
archetypes = ["structuring", "burst", "pass_through", "bot_periodic", "dormant_spike"]
eval_models = [
    'Fixed IF',
    'Transformer',
    'Ensemble (Current 0.8/0.2 Mean)',
    'Ensemble (0.8/0.2 Window-5)',
    'Ensemble (3-Signal 0.2/0.1/0.7)'
]

print(f"Running Whole-Address Benchmark across {len(seeds)} held-out seeds (42-51)...")
seed_runs = []
for s in seeds:
    print(f"--> Executing seed {s}...")
    cfg = Config(epochs=20, seed=s, random_state=s, contamination=0.1)
    res = run_single_seed_benchmark(df, seed=s, config=cfg, n_per_type=20, w_if=0.8, w_tf=0.2)
    seed_runs.append(res)

# Aggregate results
summary = {}
for m in eval_models:
    aurocs = [r['metrics'][m]['auroc'] for r in seed_runs]
    aps = [r['metrics'][m]['ap'] for r in seed_runs]
    recs = [r['metrics'][m]['recall_at_k'] for r in seed_runs]
    
    arch_agg = {}
    for a in archetypes:
        vals = [r['metrics'][m]['type_recalls'][a] for r in seed_runs]
        arch_agg[a] = compute_bootstrap_ci(vals)
        
    summary[m] = {
        'auroc': compute_bootstrap_ci(aurocs),
        'ap': compute_bootstrap_ci(aps),
        'recall_at_k': compute_bootstrap_ci(recs),
        'archetype_recalls': arch_agg,
        'raw_aurocs': aurocs,
        'raw_aps': aps,
        'raw_recs': recs
    }

# Paired Differences vs Fixed IF
paired_vs_if = {}
for m in ['Ensemble (Current 0.8/0.2 Mean)', 'Ensemble (0.8/0.2 Window-5)', 'Ensemble (3-Signal 0.2/0.1/0.7)']:
    paired_vs_if[m] = {
        'auroc': compute_paired_bootstrap_ci(summary[m]['raw_aurocs'], summary['Fixed IF']['raw_aurocs']),
        'ap': compute_paired_bootstrap_ci(summary[m]['raw_aps'], summary['Fixed IF']['raw_aps']),
        'recall_at_k': compute_paired_bootstrap_ci(summary[m]['raw_recs'], summary['Fixed IF']['raw_recs']),
        'archetypes': {
            a: compute_paired_bootstrap_ci(
                [r['metrics'][m]['type_recalls'][a] for r in seed_runs],
                [r['metrics']['Fixed IF']['type_recalls'][a] for r in seed_runs]
            ) for a in archetypes
        }
    }

# Paired Differences vs Ensemble (0.8/0.2 Mean)
paired_vs_mean = {}
for m in ['Ensemble (0.8/0.2 Window-5)', 'Ensemble (3-Signal 0.2/0.1/0.7)']:
    paired_vs_mean[m] = {
        'auroc': compute_paired_bootstrap_ci(summary[m]['raw_aurocs'], summary['Ensemble (Current 0.8/0.2 Mean)']['raw_aurocs']),
        'ap': compute_paired_bootstrap_ci(summary[m]['raw_aps'], summary['Ensemble (Current 0.8/0.2 Mean)']['raw_aps']),
        'recall_at_k': compute_paired_bootstrap_ci(summary[m]['raw_recs'], summary['Ensemble (Current 0.8/0.2 Mean)']['raw_recs']),
        'archetypes': {
            a: compute_paired_bootstrap_ci(
                [r['metrics'][m]['type_recalls'][a] for r in seed_runs],
                [r['metrics']['Ensemble (Current 0.8/0.2 Mean)']['type_recalls'][a] for r in seed_runs]
            ) for a in archetypes
        }
    }

# Print Markdown Table
print("\n" + "="*120)
print("WHOLE-ADDRESS BENCHMARK: 3-SIGNAL VS 0.8/0.2 WINDOW-5 VS FIXED IF VS 0.8/0.2 MEAN (SEEDS 42-51)")
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
for m in paired_vs_if:
    print(f"\n--- {m} vs Fixed IF ---")
    d = paired_vs_if[m]
    print(f"  Diff AUROC:     {d['auroc']['mean']:+.4f} [{d['auroc']['ci_lower']:+.4f}, {d['auroc']['ci_upper']:+.4f}]")
    print(f"  Diff AP:        {d['ap']['mean']:+.4f} [{d['ap']['ci_lower']:+.4f}, {d['ap']['ci_upper']:+.4f}]")
    print(f"  Diff Recall@K:  {d['recall_at_k']['mean']:+.4f} [{d['recall_at_k']['ci_lower']:+.4f}, {d['recall_at_k']['ci_upper']:+.4f}]")
    for a in archetypes:
        da = d['archetypes'][a]
        print(f"  Diff {a:<14}: {da['mean']:+.4f} [{da['ci_lower']:+.4f}, {da['ci_upper']:+.4f}]")

print("\n" + "="*120)
print("PAIRED DIFFERENCES VS ENSEMBLE (0.8/0.2 MEAN)")
print("="*120)
for m in paired_vs_mean:
    print(f"\n--- {m} vs Ensemble (0.8/0.2 Mean) ---")
    d = paired_vs_mean[m]
    print(f"  Diff AUROC:     {d['auroc']['mean']:+.4f} [{d['auroc']['ci_lower']:+.4f}, {d['auroc']['ci_upper']:+.4f}]")
    print(f"  Diff AP:        {d['ap']['mean']:+.4f} [{d['ap']['ci_lower']:+.4f}, {d['ap']['ci_upper']:+.4f}]")
    print(f"  Diff Recall@K:  {d['recall_at_k']['mean']:+.4f} [{d['recall_at_k']['ci_lower']:+.4f}, {d['recall_at_k']['ci_upper']:+.4f}]")
    for a in archetypes:
        da = d['archetypes'][a]
        print(f"  Diff {a:<14}: {da['mean']:+.4f} [{da['ci_lower']:+.4f}, {da['ci_upper']:+.4f}]")

output_json = {
    'seeds': seeds,
    'summary': {m: {k: v for k, v in summary[m].items() if not k.startswith('raw_')} for m in eval_models},
    'paired_vs_fixed_if': paired_vs_if,
    'paired_vs_mean_fusion': paired_vs_mean
}

out_path = Path('output/whole_address_fusion_comparison.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output_json, f, indent=2)
print(f"\nSaved results to {out_path}")
