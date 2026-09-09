# Data

The empirical RA-MoE-4E experiments use proprietary option data and
therefore do not redistribute the original datasets in this repository.

## Data Availability

The original empirical analysis uses option data obtained from
**OptionMetrics IvyDB U.S.**

Raw OptionMetrics observations and large derived datasets are not
included because of licensing restrictions.

Users with access to compatible option data can run the public
pipeline by providing a cleaned CSV file:

```bash
python scripts/train.py --data /path/to/cleaned_option_data.csv
```

## Expected Schema

The public implementation expects the following core columns:

| Column | Description |
|---|---|
| `secid` | Security identifier |
| `optionid` | Option identifier used for temporal grouping |
| `date` | Observation date |
| `S` | Underlying spot price |
| `K` | Strike price |
| `sigma` | Implied volatility |
| `tau` | Time to maturity |
| `r` | Risk-free rate |
| `q` | Dividend yield |
| `market_price` | Observed option market price |
| `cp_flag` | Call/put indicator |
| `volume` | Trading volume |
| `open_interest` | Open interest |
| `delta` | Option delta |
| `gamma` | Option gamma |
| `vega` | Option vega |
| `theta` | Option theta |

The preprocessing pipeline additionally constructs or uses features
such as:

- moneyness;
- bid–ask spread;
- log returns;
- RV5;
- RV20;
- implied-volatility dispersion;
- implied-volatility deviation;
- change in open interest.

Where available, a `skew` column is used by the residual-expert feature
set. The supplied experimental implementation uses a zero-valued
placeholder when this column is absent.

## Column Aliases

The preprocessing pipeline recognizes several aliases used in the
original cleaned datasets:

| Raw column | Internal name |
|---|---|
| `strike_price` | `K` |
| `impl_vol` | `sigma` |
| `div_yield` | `q` |
| `mid_price` | `market_price` |

## Temporal Input

The supplied final implementation constructs fixed-length temporal
windows within `optionid`.

The default sequence length is **10 observations**, using:

- implied volatility;
- volume;
- underlying return;
- bid–ask spread;
- open interest.

## Important

This repository does **not** include:

- raw OptionMetrics data;
- large derived prediction datasets;
- proprietary observations;
- model checkpoints generated from proprietary data.

The repository is intended to document and expose the model,
preprocessing, training, and evaluation logic without redistributing
licensed data.
