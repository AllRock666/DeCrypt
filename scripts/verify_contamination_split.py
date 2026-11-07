import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Load real dataset
df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df = df.sort_values(['address_id', 'timestamp'])

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

seed_data = {}
all_contaminated_addrs = set()
all_test_addrs_set = set()

for s in seeds:
    ss = np.random.SeedSequence(s)
    child_seeds = ss.generate_state(5)
    seed_split = int(child_seeds[0]) % (2**31 - 1)
    seed_injection = int(child_seeds[1]) % (2**31 - 1)
    
    rng_split = np.random.default_rng(seed_split)
    real_addresses = np.sort(df['address_id'].unique())
    rng_split.shuffle(real_addresses)
    
    split_idx = int(len(real_addresses) * 0.70)
    train_addrs = set(real_addresses[:split_idx])
    test_real_addrs = set(real_addresses[split_idx:])
    all_test_addrs_set.update(test_real_addrs)
    
    test_real_df = df[df['address_id'].isin(test_real_addrs)].copy()
    test_counts = test_real_df.groupby('address_id').size()
    
    strata_addrs = {
        '5-9': sorted(list(test_counts[(test_counts >= 5) & (test_counts <= 9)].index)),
        '10-24': sorted(list(test_counts[(test_counts >= 10) & (test_counts <= 24)].index)),
        '25+': sorted(list(test_counts[test_counts >= 25].index)),
        'overall': sorted(list(test_counts[test_counts >= 5].index))
    }
    
    rng_inj = np.random.default_rng(seed_injection)
    chosen_by_stratum = {}
    for st, count in [('5-9', 6), ('10-24', 6), ('25+', 5)]:
        avail = strata_addrs[st]
        n_pick = min(count, len(avail))
        chosen_by_stratum[st] = sorted(list(rng_inj.choice(avail, size=n_pick, replace=False)))
        
    all_chosen = []
    for c in [chosen_by_stratum['5-9'], chosen_by_stratum['10-24'], chosen_by_stratum['25+']]:
        all_chosen.extend(c)
    chosen_by_stratum['overall'] = sorted(all_chosen)
    
    all_contaminated_addrs.update(all_chosen)
    
    seed_data[s] = {
        'n_test_total': len(test_real_addrs),
        'avail': {st: len(strata_addrs[st]) for st in strata_addrs},
        'positives_count': {st: len(chosen_by_stratum[st]) for st in chosen_by_stratum},
        'chosen': chosen_by_stratum
    }

print("="*80)
print("CONTAMINATION TEST SPLIT VERIFICATION (SEEDS 42-51)")
print("="*80)

print(f"Total unique addresses in dataset: {len(df['address_id'].unique())}")
print(f"Total unique addresses in test sets across 10 seeds: {len(all_test_addrs_set)}")
print(f"Total unique contaminated addresses across 10 seeds: {len(all_contaminated_addrs)}")

print("\nPer-Seed Positive Counts per Stratum:")
print(f"{'Seed':<6} | {'Test Total':<10} | {'Avail 5-9':<10} | {'Avail 10-24':<12} | {'Avail 25+':<10} | {'Pos 5-9':<8} | {'Pos 10-24':<10} | {'Pos 25+':<8} | {'Pos Overall':<11}")
print("-" * 95)
for s in seeds:
    d = seed_data[s]
    print(
        f"{s:<6} | {d['n_test_total']:<10} | "
        f"{d['avail']['5-9']:<10} | {d['avail']['10-24']:<12} | {d['avail']['25+']:<10} | "
        f"{d['positives_count']['5-9']:<8} | {d['positives_count']['10-24']:<10} | {d['positives_count']['25+']:<8} | "
        f"{d['positives_count']['overall']:<11}"
    )

# Check pairwise overlap of contaminated addresses
print("\nPairwise overlap matrix of contaminated addresses across seeds:")
seed_addrs_overall = [set(seed_data[s]['chosen']['overall']) for s in seeds]
overlaps = []
for i in range(len(seeds)):
    for j in range(i + 1, len(seeds)):
        inter = len(seed_addrs_overall[i] & seed_addrs_overall[j])
        overlaps.append(inter)
print(f"Mean pairwise overlap: {np.mean(overlaps):.2f} / 17 addresses (min={min(overlaps)}, max={max(overlaps)})")
