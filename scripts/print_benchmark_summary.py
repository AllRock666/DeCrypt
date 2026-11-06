import json

with open('output/benchmark_results.json') as f:
    data = json.load(f)

print("=== PASS-THROUGH RECALLS WITH 95% CIs ===")
for m in ['Fixed IF', 'Transformer', 'Ensemble (Tuned 0.8/0.2)']:
    arch = data['summary'][m]['archetype_recalls']['pass_through']
    print(f"{m:25s}: mean = {arch['mean']:.3f} [{arch['ci_lower']:.3f}, {arch['ci_upper']:.3f}] (std={arch['std']:.3f})")

print("\n=== PASS-THROUGH DELAY VARIANTS ===")
for d, vals in data['pass_through_delay_summary'].items():
    print(f"Delay {d}:")
    for m in ['Fixed IF', 'Transformer', 'Ensemble (Tuned 0.8/0.2)']:
        v = vals[m]
        print(f"  {m:25s}: {v['mean']:.3f} [{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]")

print("\n=== OVERALL BENCHMARK METRICS ===")
for m in ['Legacy IF', 'Fixed IF', 'Transformer', 'Ensemble (Mean-Rank)', 'Ensemble (Max-Rank)', 'Ensemble (Tuned 0.8/0.2)']:
    s = data['summary'][m]
    print(f"{m:25s}: AUROC={s['auroc']['mean']:.3f} [{s['auroc']['ci_lower']:.3f}, {s['auroc']['ci_upper']:.3f}] | AP={s['ap']['mean']:.3f} [{s['ap']['ci_lower']:.3f}, {s['ap']['ci_upper']:.3f}] | Recall@K={s['recall_at_k']['mean']:.3f} [{s['recall_at_k']['ci_lower']:.3f}, {s['recall_at_k']['ci_upper']:.3f}]")

print("\n=== ARCHETYPE RECALLS TABLE ===")
for m in ['Legacy IF', 'Fixed IF', 'Transformer', 'Ensemble (Mean-Rank)', 'Ensemble (Max-Rank)', 'Ensemble (Tuned 0.8/0.2)']:
    arch = data['summary'][m]['archetype_recalls']
    print(f"{m:25s} | Struct={arch['structuring']['mean']:.2f} | Burst={arch['burst']['mean']:.2f} | PassThrough={arch['pass_through']['mean']:.2f} | Bot={arch['bot_periodic']['mean']:.2f} | Dormant={arch['dormant_spike']['mean']:.2f}")

print("\n=== PAIRED DIFFERENCES (Tuned Fusion vs Fixed IF) ===")
for k, v in data['paired_differences_tuned_vs_fixed_if'].items():
    print(f"{k:15s}: mean={v['mean']:+.4f} [{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}]")
