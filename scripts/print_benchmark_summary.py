import json

with open('output/benchmark_results.json') as f:
    data = json.load(f)

tuned_label = data.get('tuned_label', 'Ensemble (Tuned 0.8/0.2)')
models_to_show = ['Legacy IF', 'Fixed IF', 'Transformer', 'Ensemble (Mean-Rank)', 'Ensemble (Max-Rank)', tuned_label]
if 'Ensemble (Current 0.8/0.2 Mean)' in data['summary']:
    models_to_show.append('Ensemble (Current 0.8/0.2 Mean)')

print(f"=== PASS-THROUGH RECALLS WITH 95% CIs (Tuned: {tuned_label}) ===")
for m in ['Fixed IF', 'Transformer', tuned_label]:
    if m in data['summary']:
        arch = data['summary'][m]['archetype_recalls']['pass_through']
        print(f"{m:32s}: mean = {arch['mean']:.3f} [{arch['ci_lower']:.3f}, {arch['ci_upper']:.3f}] (std={arch['std']:.3f})")

print("\n=== PASS-THROUGH DELAY VARIANTS ===")
for d, vals in data['pass_through_delay_summary'].items():
    print(f"Delay {d}:")
    for m in ['Fixed IF', 'Transformer', tuned_label]:
        if m in vals:
            v = vals[m]
            print(f"  {m:32s}: {v['mean']:.3f} [{v['ci_lower']:.3f}, {v['ci_upper']:.3f}]")

print("\n=== OVERALL BENCHMARK METRICS ===")
for m in models_to_show:
    if m in data['summary']:
        s = data['summary'][m]
        print(f"{m:32s}: AUROC={s['auroc']['mean']:.3f} [{s['auroc']['ci_lower']:.3f}, {s['auroc']['ci_upper']:.3f}] | AP={s['ap']['mean']:.3f} [{s['ap']['ci_lower']:.3f}, {s['ap']['ci_upper']:.3f}] | Recall@K={s['recall_at_k']['mean']:.3f} [{s['recall_at_k']['ci_lower']:.3f}, {s['recall_at_k']['ci_upper']:.3f}]")

print("\n=== ARCHETYPE RECALLS TABLE ===")
for m in models_to_show:
    if m in data['summary']:
        arch = data['summary'][m]['archetype_recalls']
        print(f"{m:32s} | Struct={arch['structuring']['mean']:.2f} | Burst={arch['burst']['mean']:.2f} | PassThrough={arch['pass_through']['mean']:.2f} | Bot={arch['bot_periodic']['mean']:.2f} | Dormant={arch['dormant_spike']['mean']:.2f}")

print(f"\n=== PAIRED DIFFERENCES ({tuned_label} vs Fixed IF) ===")
for k, v in data['paired_differences_tuned_vs_fixed_if'].items():
    print(f"{k:15s}: mean={v['mean']:+.4f} [{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}]")

if 'paired_differences_tuned_vs_previous_fusion' in data:
    print(f"\n=== PAIRED DIFFERENCES ({tuned_label} vs Previous Ensemble 0.8/0.2 Mean) ===")
    for k, v in data['paired_differences_tuned_vs_previous_fusion'].items():
        print(f"{k:15s}: mean={v['mean']:+.4f} [{v['ci_lower']:+.4f}, {v['ci_upper']:+.4f}]")
