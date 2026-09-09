# Results

This directory contains selected aggregate outputs from the original
RA-MoE-4E experiments and from the public post-hoc analysis scripts.

Raw predictions and row-level OptionMetrics-derived datasets are not
included because of data-licensing restrictions.

## Canonical Pricing Results

`pricing_summary.csv` contains the cross-market SPX/AAPL comparison
used for the headline pricing results in the repository README.

## Financial Consistency

The `financial_consistency/` directory contains aggregate SPX
post-hoc diagnostic results for:

- strike monotonicity and discrete convexity;
- calendar-spread consistency;
- delta-hedging residuals.

## Expert Contribution

Public expert-contribution tables should be generated using
`analysis/expert_contribution.py`.

Historical files labelled as architectural "ablation" are not
distributed because the original analysis was post-hoc and the
residual condition attenuated the residual correction rather than
retraining a reduced architecture.
