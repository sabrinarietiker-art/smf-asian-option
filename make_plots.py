"""Generate all figures for the report."""
import json, math, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

with open('/home/user/workspace/smf/dataset.json') as f:
    data = json.load(f)
with open('/home/user/workspace/smf/results.json') as f:
    res = json.load(f)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 200, 'savefig.bbox': 'tight'
})

OUT = '/home/user/workspace/smf/figs'
import os
os.makedirs(OUT, exist_ok=True)

# ---- Fig 1: AAPL price + log-returns ----
prices = data['prices']
dates = [p[0] for p in prices]
px    = [p[1] for p in prices]
import datetime as dt
dates_dt = [dt.datetime.strptime(s, '%Y-%m-%d') for s in dates]
rets = data['log_returns']
ret_dates = [dt.datetime.strptime(r[0], '%Y-%m-%d') for r in rets]
ret_vals  = [r[1] for r in rets]

fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
axes[0].plot(dates_dt, px, color='#1f77b4', lw=0.8)
axes[0].set_ylabel('AAPL close (USD)')
axes[0].set_title(f'Apple Inc. — daily close, 2020-01-02 to {dates[-1]}')
axes[0].grid(alpha=0.3)
axes[1].plot(ret_dates, ret_vals, color='#444', lw=0.4)
axes[1].set_ylabel('Daily log-return')
axes[1].set_xlabel('Date')
axes[1].grid(alpha=0.3)
axes[1].axhline(0, color='k', lw=0.4)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_prices_returns.png')
plt.close()
print('Saved fig1')

# ---- Fig 2: histogram of daily log-returns + normal overlay ----
fig, ax = plt.subplots(figsize=(6.5, 3.8))
arr = np.array(ret_vals)
ax.hist(arr, bins=60, density=True, alpha=0.6, color='#1f77b4', edgecolor='white')
xs = np.linspace(arr.min(), arr.max(), 300)
ax.plot(xs, norm.pdf(xs, arr.mean(), arr.std(ddof=1)), 'r-', lw=1.4,
        label=f'N({arr.mean():.4f}, {arr.std(ddof=1)**2:.5f})')
ax.set_xlabel('Daily log-return')
ax.set_ylabel('Density')
ax.set_title('Distribution of daily log-returns (AAPL, 2020-2026)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_returns_hist.png')
plt.close()
print('Saved fig2')

# ---- Fig 3: binomial tree of stock prices ----
S0 = res['S0']; u = res['u']; d = res['d']; n = int(res['n'])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for t in range(n+1):
    for j in range(t+1):
        S = S0 * (u**j) * (d**(t-j))
        ax.plot(t, S, 'o', color='#1f77b4', markersize=2)
        if t < n:
            S_up = S0 * (u**(j+1)) * (d**(t-j))
            S_dn = S0 * (u**j) * (d**(t+1-j))
            ax.plot([t, t+1], [S, S_up], color='gray', lw=0.3, alpha=0.5)
            ax.plot([t, t+1], [S, S_dn], color='gray', lw=0.3, alpha=0.5)
ax.set_xlabel('Step t')
ax.set_ylabel(r'Stock price $S_t$ (USD)')
ax.set_title(f'Binomial tree, n={n}, u={u:.4f}, d={d:.4f}, $S_0$={S0:.2f}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_tree.png')
plt.close()
print('Saved fig3')

# ---- Fig 4: terminal payoff scatter (S_n vs avg, color = payoff) ----
import csv
S_n = []; avg = []; pay = []; pq = []
with open('/home/user/workspace/smf/terminal_dist_q.csv') as f:
    r_ = csv.reader(f); next(r_)
    for row in r_:
        S_n.append(float(row[0])); avg.append(float(row[1]))
        pay.append(float(row[2])); pq.append(float(row[3]))
S_n = np.array(S_n); avg = np.array(avg); pay = np.array(pay); pq = np.array(pq)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
sc = ax.scatter(avg, S_n, c=pay, s=2, cmap='viridis')
lim = [min(S_n.min(), avg.min())*0.95, max(S_n.max(), avg.max())*1.05]
ax.plot(lim, lim, 'r--', lw=0.8, label=r'$S_n = \bar S_n$ (zero payoff line)')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'Average $\bar S_n$')
ax.set_ylabel(r'Terminal $S_n$')
ax.set_title('Terminal nodes — payoff $\\max(S_n - \\bar S_n,0)$')
plt.colorbar(sc, label='Payoff')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_terminal_scatter.png')
plt.close()
print('Saved fig4')

# ---- Fig 5: distribution of D_n (empirical vs normal) ----
D = S_n - avg
# Empirical histogram weighted by risk-neutral prob
fig, ax = plt.subplots(figsize=(6.5, 3.8))
counts, bins, _ = ax.hist(D, bins=80, weights=pq, density=True, alpha=0.55,
                          color='#1f77b4', edgecolor='white',
                          label='Empirical (tree, $\\mathbb{Q}$)')
xs = np.linspace(D.min(), D.max(), 400)
ax.plot(xs, norm.pdf(xs, res['mu_D'], res['sd_D']),
        'r-', lw=1.5,
        label=f'Normal({res["mu_D"]:.2f}, {res["sd_D"]:.2f}$^2$)')
ax.axvline(0, color='k', lw=0.6, linestyle=':')
ax.set_xlabel(r'$D_n = S_n - \bar S_n$')
ax.set_ylabel('Density')
ax.set_title(r'Distribution of $D_n$ under $\mathbb{Q}$ — empirical vs. normal')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_Dn_distribution.png')
plt.close()
print('Saved fig5')

# ---- Fig 6: robustness heatmap ----
r_grid = res['r_grid']; sigma_grid = res['sigma_grid']
prices_grid = np.array(res['robust_prices'])
fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(prices_grid, aspect='auto', cmap='YlOrRd', origin='lower')
ax.set_xticks(range(len(sigma_grid)))
ax.set_xticklabels([f'{s:.3f}' for s in sigma_grid])
ax.set_yticks(range(len(r_grid)))
ax.set_yticklabels([f'{rv:.3f}' for rv in r_grid])
ax.set_xlabel(r'Annualized volatility $\sigma$')
ax.set_ylabel('Risk-free rate $r$')
ax.set_title('Asian floating-strike call price — robustness grid')
vmin, vmax = prices_grid.min(), prices_grid.max()
for i in range(prices_grid.shape[0]):
    for j in range(prices_grid.shape[1]):
        # Use white text on dark cells, black on light cells
        rel = (prices_grid[i,j] - vmin) / (vmax - vmin)
        text_color = 'white' if rel > 0.55 else 'black'
        ax.text(j, i, f'{prices_grid[i,j]:.2f}', ha='center', va='center',
                color=text_color, fontsize=8)
plt.colorbar(im, label='Price (USD)')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_robustness.png')
plt.close()
print('Saved fig6')

# ---- Fig 7: price as function of sigma (and r) ----
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
for i, rv in enumerate(r_grid):
    axes[0].plot(sigma_grid, prices_grid[i], 'o-', lw=1, ms=4,
                 label=f'r={rv:.3f}')
axes[0].set_xlabel(r'$\sigma$ (annualized)')
axes[0].set_ylabel('Option price (USD)')
axes[0].set_title('Sensitivity to volatility')
axes[0].legend(fontsize=7, loc='upper left')
axes[0].grid(alpha=0.3)
for j, sv in enumerate(sigma_grid):
    axes[1].plot(r_grid, prices_grid[:, j], 'o-', lw=1, ms=4,
                 label=f'σ={sv:.3f}')
axes[1].set_xlabel('r (annualized)')
axes[1].set_ylabel('Option price (USD)')
axes[1].set_title('Sensitivity to risk-free rate')
axes[1].legend(fontsize=7, loc='upper left')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_sensitivities.png')
plt.close()
print('Saved fig7')
print('\nAll figures saved to', OUT)
