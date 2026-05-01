# SMF HSG Assignment — Asian Option Pricing

Public companion repository for the Bachelor assignment in
**Stochastic Methods in Finance** (Spring 2026, University of St. Gallen,
Prof. E. De Giorgi).

The task: price a **European floating-strike Asian call option** on
Apple Inc. with maturity 6 months, using a 25-period binomial tree, and
compare against a normal-distribution approximation.

> **Headline result:** Tree price **USD 13.89**.
> Normal approximation **USD 14.10** (1.5% above tree).

## Files

| File | What it does |
|---|---|
| `load_data.py` | Reads AAPL daily prices from the course `.xlsx`, computes daily log-returns and annualised σ, saves to `dataset.json`. |
| `price_option.py` | Builds the augmented `(S, C)` binomial tree, prices the option by backward induction, runs a robustness grid in `(r, σ)`, and computes the closed-form normal approximation. |
| `make_plots.py` | Reproduces all seven figures used in the report. |
| `figs/` | Generated figures (PNG). |

## Reproducing

```bash
python3 load_data.py     # reads xlsx, writes dataset.json
python3 price_option.py  # ~3 minutes; writes results.json + CSVs
python3 make_plots.py    # writes all PNGs in figs/
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `openpyxl` (plus the
standard library).

## Method, in one paragraph

A vanilla binomial tree recombines: the order of up- and down-moves
doesn't matter, only their count. An Asian option's payoff depends on
the running average, so the order matters and the tree no longer
recombines. We restore tractability by carrying a second state
variable `C_t = S_0 + S_1 + ... + S_t` at every node. The pair
`(S_t, C_t)` is a Markov state, so backward induction works. We
collapse identical `(j, C)` states (different paths, same outcome)
to keep the state count manageable: about 2.4 million states total
versus the 33.5 million distinct paths.

## Author

Sabrina Rietiker · University of St. Gallen
