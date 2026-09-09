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
python scripts/train.py \
    --data /path/to/cleaned_option_data.csv \
    --output-dir outputs/experiment
```

## Expected Schema

The model and accompanying analysis scripts use the following columns.
Some analysis-specific fields are not required by every execution path.

| Column | Description |
|---|---|
| `secid` | Security identifier used for gate-level grouping |
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

### Option Type

`cp_flag` should be provided when the dataset contains both call and
put options.

When `cp_flag` is available, the preprocessing pipeline evaluates the
BSM baseline using the corresponding call or put pricing formula.

If `cp_flag` is absent, the supplied implementation retains the
original fallback behaviour and evaluates the BSM baseline as calls.

Users working with mixed call/put datasets should therefore provide
`cp_flag`.

## Column Aliases

The preprocessing pipeline recognizes several aliases used in the
original cleaned datasets:

| Raw column | Internal name |
|---|---|
| `strike_price` | `K` |
| `impl_vol` | `sigma` |
| `div_yield` | `q` |
| `mid_price` | `market_price` |

## Derived Features

The preprocessing pipeline constructs or uses features including:

- moneyness;
- bid–ask spread;
- log returns;
- RV5;
- RV20;
- implied-volatility dispersion;
- implied-volatility deviation;
- change in open interest.

Where available, a `skew` column is used by the residual-expert feature
set.

When `skew` is absent, the supplied experimental implementation uses a
zero-valued placeholder.

## Temporal Input

The supplied final implementation constructs fixed-length temporal
windows within `optionid`.

The default sequence length is **10 observations**.

Each temporal sequence contains:

- implied volatility;
- trading volume;
- underlying return;
- bid–ask spread;
- open interest.

Short histories are left-padded using the earliest available
observation in the sequence.

## Experimental Split

The supplied final implementation preserves observation order and uses
an **80/10/10 train-validation-test split** with `shuffle=False`.

The public refactor retains this splitting behaviour for consistency
with the original experiment.

This should not be interpreted as a claim that every compatible input
dataset is globally sorted in strict chronological order; users should
ensure that their cleaned data have the intended observation ordering
before running the experiment.

## Scaling

Static features are standardized using a `StandardScaler` fitted on the
training subset.

The fitted static-feature scaler is reused for the validation and test
subsets.

The supplied implementation standardizes residual-specific and
gate-specific extra features using the statistics of each dataset
instance.

This behaviour is retained in the public refactor for consistency with
the original experimental implementation.

## Generated Outputs

The training entry point can save aligned test predictions and model
diagnostics using:

```bash
python scripts/train.py \
    --data /path/to/cleaned_option_data.csv \
    --output-dir outputs/experiment
```

The resulting:

```text
outputs/experiment/test_results.csv
```

contains the aligned test observations together with:

- RA-MoE-4E predictions;
- BSM baseline predictions;
- four expert outputs;
- four routing weights.

These row-level outputs are intended for local post-hoc analysis and
are excluded from version control because they may contain information
derived from proprietary option data.

## Important

This repository does **not** include:

- raw OptionMetrics data;
- large derived prediction datasets;
- proprietary row-level observations;
- locally generated `test_results.csv` files;
- model checkpoints generated from proprietary data.

The repository is intended to expose the model, preprocessing,
training, evaluation, and analysis logic without redistributing
licensed observations.
