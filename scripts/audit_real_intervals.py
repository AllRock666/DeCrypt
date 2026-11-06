import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

df = pd.read_csv('data/bitcoin_transactions.csv', header=None, names=['address_id', 'hash', 'timestamp', 'type', 'amount'])
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
df['type_val'] = df['type'].apply(lambda x: 1 if str(x).lower().strip() == 'deposit' else -1)
df['abs_amt'] = df['amount'].abs()

df = df.sort_values(['address_id', 'timestamp'])

addrs_under_60s = set()
addrs_dep_then_wd_15m = set()

for addr, group in df.groupby('address_id'):
    times = group['timestamp'].values
    types = group['type_val'].values
    n = len(times)
    if n > 1:
        diffs = (times[1:] - times[:-1]) / np.timedelta64(1, 's')
        if np.any(diffs < 60.0):
            addrs_under_60s.add(addr)
            
        for i in range(n - 1):
            if types[i] == 1:
                for j in range(i + 1, n):
                    dt = (times[j] - times[i]) / np.timedelta64(1, 's')
                    if dt <= 900.0:
                        if types[j] == -1:
                            addrs_dep_then_wd_15m.add(addr)
                            break
                    else:
                        break

total_addrs = df['address_id'].nunique()
print(f"Total unique addresses: {total_addrs}")
print(f"Addresses with inter-event gaps < 60s: {len(addrs_under_60s)} ({len(addrs_under_60s)/total_addrs*100:.2f}%)")
print(f"Addresses with deposit followed by withdrawal within 15m: {len(addrs_dep_then_wd_15m)} ({len(addrs_dep_then_wd_15m)/total_addrs*100:.2f}%)")
