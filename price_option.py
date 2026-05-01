"""
Stochastic Methods in Finance — HSG Bachelor Spring 2026
Asian floating-strike call: pricing via 25-period CRR-style binomial tree
with augmented (S, C) state to handle path dependence,
plus a normal-approximation alternative.

Author: solution worked out for the assignment of Prof. Enrico De Giorgi.
"""

import json, math, numpy as np
from collections import defaultdict
from scipy.stats import norm

# ---------- 1. Load market data and parameters ------------------------
with open('/home/user/workspace/smf/dataset.json') as f:
    data = json.load(f)

S0          = data['S0']                  # 267.61, last available AAPL close
sigma_daily = data['sigma_daily']         # sample std of daily log-returns
sigma       = data['sigma_annual']        # = sigma_daily * sqrt(250)

T   = 0.5            # maturity in years (6 months)
n   = 25             # number of binomial periods
r   = 0.01           # annual risk-free rate
dt  = T / n
u   = math.exp(sigma * math.sqrt(dt))
d   = math.exp(-sigma * math.sqrt(dt))    # = 1/u
p   = 0.5            # physical probability per assignment
q   = (math.exp(r*dt) - d) / (u - d)      # risk-neutral up-probability
disc = math.exp(-r * dt)                  # one-step discount factor

print("=== Parameters ===")
print(f"S0 = {S0:.4f}")
print(f"sigma_daily = {sigma_daily:.6f}, sigma_annual = {sigma:.6f}")
print(f"T = {T}, n = {n}, dt = {dt:.6f}")
print(f"u = {u:.6f}, d = {d:.6f}")
print(f"p (physical) = {p}, q (risk-neutral) = {q:.6f}")
print(f"one-step discount = exp(-r*dt) = {disc:.6f}")

# ---------- 2. Forward pass: enumerate reachable (k, C) states --------
# At step t, the stock is S_t = S0 * u^j * d^{t-j} where j = #ups so far.
# We carry the cumulative sum C_t = S_0 + S_1 + ... + S_t.
# State variable: (j, C_t) with j in 0..t.
#
# We store, at each (t, j), a dictionary {C_value -> probability_mass_unused_here}.
# Because we only need the *set* of reachable C-values (not probabilities) to
# allocate option-value memory, we build a dict (t, j) -> list of unique C's.
#
# Number of reachable C-values at (t, j) is at most C(t, j); for n = 25 the total
# state count is sum over t of 2^t ≈ 6.7e7 in the worst case. We must keep this
# tractable via a dict keyed by C *rounded* — but exact arithmetic is preferable.
# In practice the number of distinct cumulative sums grows like 2^t, and at
# n = 25 it equals 2^25 ≈ 3.4e7 paths. Memory matters.
#
# However, many paths share the same (j, C). Empirically the set of reachable
# C's at (t, j) is much smaller than C(t, j). To stay safe, we enumerate all
# 2^n paths once, but only at maturity, and aggregate by (j, C) there.
# For backward induction we work directly with paths: dynamic programming on
# (t, j, C) where C is rounded to a stable key.

def price_asian_floating_strike(S0, n, u, d, q, disc, return_full=False):
    """
    Backward-induction Asian option price using path enumeration.
    For n = 25 we enumerate 2^25 = 33,554,432 terminal paths efficiently
    via a recursive dict-based DP.

    State at time t: (j = # ups, C = sum of S_0..S_t).  We collapse identical
    (j, C) states (reached by different orderings) into a single state with
    multiplicity tracked implicitly by the DP value function V(t, j, C).

    Recurrence:
        V(n, j, C) = max(S0 * u^j * d^{n-j}  -  C/(n+1),  0)
        V(t, j, C) = disc * [ q * V(t+1, j+1, C + S_{t+1}^{up})
                            + (1-q) * V(t+1, j,   C + S_{t+1}^{dn}) ]
    """
    # Forward pass to enumerate all reachable (t, j, C) keys.
    # states[t] = dict mapping (j, round(C, 8)) -> exact C
    states = [dict() for _ in range(n+1)]
    states[0][(0, round(S0, 10))] = S0

    # Precompute stock price at any (t, j)
    def Stj(t, j):
        return S0 * (u ** j) * (d ** (t - j))

    for t in range(n):
        next_states = states[t+1]
        for (j, Ckey), Cval in states[t].items():
            # up
            S_up = Stj(t+1, j+1)
            C_up = Cval + S_up
            key_up = (j+1, round(C_up, 8))
            if key_up not in next_states:
                next_states[key_up] = C_up
            # down
            S_dn = Stj(t+1, j)
            C_dn = Cval + S_dn
            key_dn = (j, round(C_dn, 8))
            if key_dn not in next_states:
                next_states[key_dn] = C_dn

    # Report number of reachable states
    counts = [len(s) for s in states]
    print(f"\nReachable (j, C) states per step: {counts[:5]} ... {counts[-3:]}")
    print(f"Total terminal states at t=n: {counts[-1]}")
    print(f"Sum of states across all t: {sum(counts)}")

    # Terminal payoff
    V = [dict() for _ in range(n+1)]
    for (j, Ckey), C in states[n].items():
        S_term = Stj(n, j)
        avg = C / (n + 1)
        V[n][(j, Ckey)] = max(S_term - avg, 0.0)

    # Backward induction
    for t in range(n-1, -1, -1):
        for (j, Ckey), Cval in states[t].items():
            S_up = Stj(t+1, j+1); C_up = Cval + S_up
            S_dn = Stj(t+1, j);   C_dn = Cval + S_dn
            v_up = V[t+1][(j+1, round(C_up, 8))]
            v_dn = V[t+1][(j,   round(C_dn, 8))]
            V[t][(j, Ckey)] = disc * (q * v_up + (1 - q) * v_dn)

    price = V[0][(0, round(S0, 10))]
    if return_full:
        return price, states, V
    return price

print("\n=== 3. Pricing the Asian floating-strike call ===")
price, states, V = price_asian_floating_strike(
    S0, n, u, d, q, disc, return_full=True)
print(f"Tree price A_0 = {price:.6f}")

# ---------- 4. Diagnostics: terminal distribution ---------------------
# Collect terminal payoff distribution under risk-neutral measure
# We need risk-neutral probabilities per (j, C) at maturity.
def terminal_distribution(S0, n, u, d, prob_up):
    """Returns list of (S_n, avg, payoff, probability) under measure with prob_up."""
    # path_prob_per_state: at maturity, sum over paths reaching (j, C).
    # We forward-propagate probabilities.
    cur = {(0, round(S0, 10), S0): 1.0}  # key includes exact C for value, rounded for hash
    for t in range(n):
        nxt = defaultdict(float)
        for (j, Ckey, Cval), pr in cur.items():
            S_up = S0 * (u ** (j+1)) * (d ** (t - j))
            S_dn = S0 * (u ** j) * (d ** (t + 1 - j))
            C_up = Cval + S_up
            C_dn = Cval + S_dn
            nxt[(j+1, round(C_up, 8), C_up)] += pr * prob_up
            nxt[(j,   round(C_dn, 8), C_dn)] += pr * (1 - prob_up)
        cur = nxt

    out = []
    for (j, Ckey, Cval), pr in cur.items():
        S_term = S0 * (u ** j) * (d ** (n - j))
        avg = Cval / (n + 1)
        payoff = max(S_term - avg, 0.0)
        out.append((S_term, avg, payoff, pr))
    return out

print("\n=== 4. Terminal distribution (risk-neutral) ===")
term_q = terminal_distribution(S0, n, u, d, q)
total_p = sum(pr for _,_,_,pr in term_q)
EA_q = sum(payoff * pr for _,_,payoff,pr in term_q)
print(f"Sum of risk-neutral probs (sanity): {total_p:.10f}")
print(f"E^Q[A_T] = {EA_q:.6f}")
print(f"Discounted E^Q[A_T] = {EA_q * math.exp(-r*T):.6f}  (should equal tree price)")

# Also under physical p=1/2 for the normal-approximation comparison below
print("\nTerminal distribution under p=1/2...")
term_p = terminal_distribution(S0, n, u, d, 0.5)

# Save terminal distribution summary
import csv
with open('/home/user/workspace/smf/terminal_dist_q.csv','w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['S_n','S_bar','payoff','prob_Q'])
    for row in sorted(term_q, key=lambda x: x[0]):
        w.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.10f}"])

# ---------- 5. Robustness: vary r and sigma --------------------------
print("\n=== 5. Robustness checks ===")
r_grid     = [0.00, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
sigma_grid = [sigma*0.5, sigma*0.75, sigma, sigma*1.25, sigma*1.5]

robust = []
for rr in r_grid:
    row = []
    for ss in sigma_grid:
        uu = math.exp(ss * math.sqrt(dt))
        dd = 1.0/uu
        qq = (math.exp(rr*dt) - dd) / (uu - dd)
        ds = math.exp(-rr*dt)
        pr_ = price_asian_floating_strike(S0, n, uu, dd, qq, ds)
        row.append(pr_)
        print(f"  r={rr:.3f}, sigma={ss:.4f}, q={qq:.4f}: price={pr_:.4f}")
    robust.append(row)

# Save
np.save('/home/user/workspace/smf/robust_grid.npy',
        {'r_grid': r_grid, 'sigma_grid': sigma_grid, 'prices': robust,
         'base_sigma': sigma, 'base_r': r})

with open('/home/user/workspace/smf/robust.csv','w', newline='') as f:
    w = csv.writer(f)
    w.writerow([''] + [f"sigma={s:.4f}" for s in sigma_grid])
    for rr, row in zip(r_grid, robust):
        w.writerow([f"r={rr:.3f}"] + [f"{v:.4f}" for v in row])

# ---------- 6. Normal approximation -----------------------------------
# We approximate D_n = S_n - bar S_n under the risk-neutral measure as Normal.
# The increments X_i = log(S_i/S_{i-1}) under Q are i.i.d. with
#   X_i = +sigma*sqrt(dt) w.p. q,  -sigma*sqrt(dt) w.p. 1-q.
# Hence E^Q[X_i] = (2q-1)*sigma*sqrt(dt) =: mu_X
#       Var^Q[X_i] = sigma^2 * dt * (1 - (2q-1)^2)
#
# log S_t = log S_0 + sum_{i=1..t} X_i
# So S_t is lognormal with parameters (log S0 + t*mu_X, t*Var(X)).
# A common, tractable simplification: use a *lognormal* approximation of the
# distribution of (S_n, bar S_n) jointly, then normal approximation of D_n.
#
# A cleaner closed-form normal approximation (as suggested in the assignment):
# Treat log(S_t/S_0) as Normal(mu_X * t, Var(X) * t). Then S_t is lognormal.
# bar S_n = (1/(n+1)) sum_{t=0..n} S_t  is approximately lognormal too,
# but the assignment asks for a *normal* approximation of D_n itself.
#
# Strategy: compute E[D_n], Var[D_n] under Q, then approximate
#   D_n ≈ Normal(mu_D, var_D)
# and price as
#   A_0 ≈ exp(-r T) * E[max(D_n, 0)]
#       = exp(-r T) * [ mu_D * Phi(mu_D/sigma_D) + sigma_D * phi(mu_D/sigma_D) ]
# (this is the standard formula for E[max(X,0)] when X ~ N(mu, sigma^2)).

# Compute moments of S_t under Q exactly using lognormal moment generating
# function with the exact Bernoulli increments.
# E^Q[S_t]   = S_0 * (q*u + (1-q)*d)^t = S_0 * exp(r*dt)^t = S_0 * exp(r*t).
# That's the martingale property.
# For Var and Cov we need E^Q[S_t S_s]. With independent increments,
# for s <= t: S_t = S_s * prod_{i=s+1..t} (u or d), so
#   E[S_t S_s] = E[S_s^2] * E[prod_{i=s+1..t} factor]
#              = E[S_s^2] * (q*u + (1-q)*d)^{t-s}
#              = E[S_s^2] * exp(r*(t-s)).
# And E[S_s^2] = S_0^2 * (q*u^2 + (1-q)*d^2)^s.
# Define a := q*u^2 + (1-q)*d^2.

a = q*u*u + (1-q)*d*d
mu_S = lambda t: S0 * math.exp(r*dt*t)
ES2  = lambda t: S0*S0 * (a ** t)

# E[S_n] and Var[S_n]
ESn  = mu_S(n)
ESn2 = ES2(n)
VarSn = ESn2 - ESn**2

# E[bar S_n] = (1/(n+1)) sum_{t=0..n} E[S_t]
EbarS = sum(mu_S(t) for t in range(n+1)) / (n+1)
# Var[bar S_n] = (1/(n+1))^2 * sum_{s,t} Cov(S_s,S_t)
#  Cov(S_s,S_t) = E[S_s S_t] - E[S_s]E[S_t]
def cov_ss(s, t):
    if s > t: s, t = t, s
    ESsSt = ES2(s) * math.exp(r*dt*(t-s))
    return ESsSt - mu_S(s)*mu_S(t)

# Var[D_n] = Var[S_n] + Var[barS_n] - 2 Cov(S_n, barS_n)
VarBar = 0.0
CovSnBar = 0.0
for s in range(n+1):
    for t in range(n+1):
        VarBar += cov_ss(s, t)
    CovSnBar += cov_ss(s, n)   # Cov(S_n, S_s)
VarBar /= (n+1)**2
CovSnBar /= (n+1)

mu_D = ESn - EbarS
var_D = VarSn + VarBar - 2*CovSnBar
sd_D = math.sqrt(var_D)

print("\n=== 6. Normal approximation of D_n ===")
print(f"E^Q[S_n]      = {ESn:.6f}")
print(f"E^Q[bar S_n]  = {EbarS:.6f}")
print(f"E^Q[D_n]      = {mu_D:.6f}")
print(f"Var^Q[S_n]    = {VarSn:.6f}")
print(f"Var^Q[bar S_n]= {VarBar:.6f}")
print(f"Cov^Q(S_n,bar)= {CovSnBar:.6f}")
print(f"Var^Q[D_n]    = {var_D:.6f}, SD = {sd_D:.6f}")

# E[max(N(mu, sigma^2), 0)] = mu * Phi(mu/sigma) + sigma * phi(mu/sigma)
z = mu_D / sd_D
EmaxD = mu_D * norm.cdf(z) + sd_D * norm.pdf(z)
price_normal = math.exp(-r*T) * EmaxD
print(f"\nE^Q[max(D_n,0)] (normal approx) = {EmaxD:.6f}")
print(f"Discounted = {price_normal:.6f}")
print(f"Tree price (for comparison) = {price:.6f}")
print(f"Absolute difference: {abs(price_normal - price):.6f}")
print(f"Relative difference: {abs(price_normal - price)/price*100:.3f}%")

# Sanity: empirical mean/var of D_n under Q from the terminal distribution
EDn_emp  = sum((S - A) * pr for S, A, _, pr in term_q)
ED2n_emp = sum(((S - A)**2) * pr for S, A, _, pr in term_q)
VarDn_emp = ED2n_emp - EDn_emp**2
print(f"\nEmpirical (from tree) E^Q[D_n] = {EDn_emp:.6f}, Var = {VarDn_emp:.6f}")

# Save summary
summary = {
    'S0': S0, 'sigma_daily': sigma_daily, 'sigma_annual': sigma,
    'T': T, 'n': n, 'dt': dt, 'r': r,
    'u': u, 'd': d, 'p': p, 'q': q,
    'tree_price': price,
    'normal_price': price_normal,
    'mu_D': mu_D, 'var_D': var_D, 'sd_D': sd_D,
    'ESn': ESn, 'EbarS': EbarS, 'VarSn': VarSn, 'VarBar': VarBar,
    'CovSnBar': CovSnBar,
    'EDn_emp': EDn_emp, 'VarDn_emp': VarDn_emp,
    'r_grid': r_grid, 'sigma_grid': sigma_grid, 'robust_prices': robust,
}
with open('/home/user/workspace/smf/results.json','w') as f:
    json.dump(summary, f, indent=2, default=float)
print("\nSaved /home/user/workspace/smf/results.json")
