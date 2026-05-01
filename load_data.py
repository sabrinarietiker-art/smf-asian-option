"""Load AAPL data from the attached xlsx and produce the dataset used by the
assignment: daily log-returns of AAPL from 2020-01-01 through the latest
available date in 2026."""
import openpyxl, datetime as dt, math, json, os

WB = openpyxl.load_workbook('/home/user/workspace/Assignment_SMF_2026-2.xlsx',
                            data_only=True)
ws = WB['Sheet1']

rows = []
for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0:
        continue
    idx, date_str, aapl, fx = row
    if date_str is None or aapl is None:
        continue
    if isinstance(date_str, str):
        d = dt.datetime.strptime(date_str, '%Y-%m-%d').date()
    else:
        d = date_str.date() if hasattr(date_str, 'date') else date_str
    rows.append((d, float(aapl)))

# Filter 2020-01-01 to end (2026)
rows.sort(key=lambda x: x[0])
filtered = [r for r in rows if r[0] >= dt.date(2020, 1, 1)]

print(f"Total rows in xlsx: {len(rows)}")
print(f"Rows from 2020-01-01: {len(filtered)}")
print(f"First date: {filtered[0][0]}, last date: {filtered[-1][0]}")
print(f"S0 (last price): {filtered[-1][1]}")

# Compute daily log-returns
import math
log_returns = []
for i in range(1, len(filtered)):
    r = math.log(filtered[i][1] / filtered[i-1][1])
    log_returns.append((filtered[i][0], r))

n = len(log_returns)
mean = sum(r for _, r in log_returns) / n
var = sum((r-mean)**2 for _, r in log_returns) / (n-1)  # sample variance
sigma_daily = math.sqrt(var)
sigma_annual = sigma_daily * math.sqrt(250)

print(f"\nNumber of daily log-returns: {n}")
print(f"Mean daily log-return: {mean:.6f}")
print(f"Sample std (daily, log-returns): {sigma_daily:.6f}")
print(f"Annualized sigma (sqrt(250)): {sigma_annual:.6f}")

# Save to JSON for downstream scripts
out = {
    'first_date': str(filtered[0][0]),
    'last_date': str(filtered[-1][0]),
    'S0': filtered[-1][1],
    'n_obs_prices': len(filtered),
    'n_returns': n,
    'sigma_daily': sigma_daily,
    'sigma_annual': sigma_annual,
    'mean_daily_logret': mean,
    'prices': [(str(d), p) for d, p in filtered],
    'log_returns': [(str(d), r) for d, r in log_returns],
}
with open('/home/user/workspace/smf/dataset.json', 'w') as f:
    json.dump(out, f)
print("\nSaved /home/user/workspace/smf/dataset.json")
