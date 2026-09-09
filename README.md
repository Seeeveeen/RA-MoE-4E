# RA-MoE-4E

### Regime-Aware Mixture-of-Experts for Robust Option Pricing

RA-MoE-4E is a theory-guided Mixture-of-Experts framework that dynamically combines a Black–Scholes–Merton (BSM) theoretical anchor with specialized neural experts for option pricing across heterogeneous market conditions.

The project investigates a broader machine-learning question:

> **Can theory-driven and data-driven experts specialize and cooperate through state-dependent dynamic routing?**

The framework was developed as an individual research project and evaluated on S&P 500 (SPX) index options and Apple (AAPL) equity options.

---

## Overview

Classical option-pricing models provide strong theoretical structure and economic interpretability, but their assumptions may become restrictive under heterogeneous or changing market conditions.

Purely data-driven models offer greater flexibility but may sacrifice structural interpretability and robustness.

RA-MoE-4E explores a hybrid approach by combining four heterogeneous experts through a regime-aware gating network:

1. **BSM Expert** — provides a theory-driven pricing anchor.
2. **Residual MLP Expert** — learns systematic corrections relative to the BSM baseline.
3. **Static MLP Expert** — captures nonlinear cross-sectional pricing relationships.
4. **Transformer Expert** — models sequential patterns from historical option-level features.

A gating network dynamically assigns state-dependent weights to the four experts, allowing the mixture to adapt its internal pricing mechanism across observations.

---

## Architecture

RA-MoE-4E combines four heterogeneous experts through a **regime-aware gating network**.

For each option observation, the final prediction is a dynamically weighted combination of the four expert outputs:

$$
P_{\mathrm{hybrid}} = \sum_{i=1}^{4} w_i E_i(X_i)
$$

where

$$
w_i \geq 0,
\qquad
\sum_{i=1}^{4} w_i = 1.
$$

The weights are generated through a temperature-controlled softmax gating network using option and market-state features.

### Four Specialized Experts

| Expert | Role | Main Idea |
|---|---|---|
| **BSM** | Theoretical anchor | Provides a classical theory-based pricing benchmark |
| **Residual MLP** | Theory correction | Learns systematic corrections relative to the BSM baseline |
| **Static MLP** | Cross-sectional learning | Captures nonlinear relationships across static option and market features |
| **Transformer** | Sequential learning | Processes fixed-length histories of option-level temporal features |

The overall architecture can be summarized as:

```text
                     Option & Market Features
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
       Specialized Experts              Gating Network
              │                               │
    ┌─────────┼─────────┬─────────┐           │
    ▼         ▼         ▼         ▼           ▼
   BSM     Residual   Static  Transformer   Softmax
             MLP      MLP                   Weights
    │         │         │         │       w₁ w₂ w₃ w₄
    └─────────┴─────────┴─────────┴───────────┘
                              │
                              ▼
                    Weighted Aggregation
                              │
                              ▼
                     Final Option Price
```

---

## Research Questions

This project investigates four main questions:

**RQ1.** Can dynamic expert routing improve option-pricing performance across heterogeneous observations and market structures?

**RQ2.** Can theory-driven and neural experts provide complementary inductive biases?

**RQ3.** Do different experts exhibit distinct allocation patterns across different markets and option characteristics?

**RQ4.** Can a theory-guided mixture improve predictive performance while retaining interpretable connections to classical option-pricing structure?

---

## Model Inputs

The implementation uses separate feature representations for different components of the mixture.

### Static MLP

The static expert receives features including:

- spot price;
- strike price;
- moneyness;
- time to maturity;
- implied volatility;
- short- and medium-window realized-volatility statistics;
- implied-volatility dispersion;
- bid–ask spread;
- trading volume;
- open interest.

### Residual Expert

The residual-correction network receives the standardized static representation together with residual-specific features including option Greeks, moneyness, spread, and the available skew representation.

During residual-expert pretraining, the learning target is:

$$
P_{\mathrm{market}} - P_{\mathrm{BSM}}.
$$

### Transformer Expert

The supplied final implementation constructs fixed-length sequences from:

- implied volatility;
- trading volume;
- underlying return;
- bid–ask spread;
- open interest.

The default sequence length is **10 observations**.

### Gating Network

The gating network combines the standardized static representation with additional state features including:

- implied volatility;
- moneyness;
- time to maturity;
- realized-volatility measures;
- implied-volatility deviation;
- bid–ask spread;
- change in open interest.

The resulting routing distribution determines the contribution of each expert to the final prediction.

---

## Data

The empirical study uses two option markets with substantially different characteristics:

- **SPX index options:** approximately 3.94 million observations after filtering.
- **AAPL equity options:** approximately 0.85 million observations after filtering.

The analysis uses option characteristics, implied-volatility information, liquidity variables, Greeks, and temporal market features.

Raw option data were obtained from **OptionMetrics IvyDB U.S.**

> **Data availability:** Raw OptionMetrics data are not distributed in this repository because of licensing restrictions. The repository provides the model implementation and preprocessing logic without redistributing proprietary observations.

---

## Training Strategy

Training follows the procedure implemented in the supplied final experimental codebase.

### Stage 1 — Residual-Expert Pretraining

The residual MLP is first trained for **5 epochs** against the difference between observed market prices and BSM prices:

$$
y_{\mathrm{residual}}
=
P_{\mathrm{market}}
-
P_{\mathrm{BSM}}.
$$

### Stage 2 — Expert Training with the Gate Frozen

The gating network is frozen while the trainable experts are optimized through the hybrid training objective for **10 epochs**.

### Stage 3 — Joint Fine-Tuning

The gate is unfrozen and the complete mixture is jointly optimized for up to **20 epochs**.

Validation-based early stopping is used with a patience of **5 epochs**, after which the best validation model state is restored.

The supplied implementation uses:

- pretraining learning rate: `5e-5`;
- joint fine-tuning learning rate: `1e-4`;
- batch size: `512`;
- gating temperature: `4.0`;
- random seed: `42`.

---

## Evaluation

The core evaluation compares RA-MoE-4E against the BSM baseline using pricing-error metrics including:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Relative Pricing Error (RPE)**

The analysis also examines:

- routing-weight distributions;
- pricing-error distributions;
- cross-market expert allocation;
- post-hoc expert contribution sensitivity;
- supplementary implied-volatility diagnostics.

The public repository separates core evaluation metrics from market-specific post-hoc analyses.

---

## Main Results

The original experimental evaluation found lower pricing errors for RA-MoE-4E than for the BSM benchmark in both SPX and AAPL.

| Market | RA-MoE-4E RMSE | BSM RMSE |
|---|---:|---:|
| **SPX** | 1.297 | 1.379 |
| **AAPL** | 0.375 | 0.405 |

The implementation additionally computes relative improvement against BSM using MSE:

$$
\mathrm{Improvement}
=
\frac{
\mathrm{MSE}_{\mathrm{BSM}}
-
\mathrm{MSE}_{\mathrm{RA\text{-}MoE}}
}{
\mathrm{MSE}_{\mathrm{BSM}}
}
\times 100.
$$

These results motivate examining not only aggregate prediction accuracy, but also how the learned mixture distributes responsibility among heterogeneous experts.

---

## Cross-Market Expert Specialization

A central interpretability component of RA-MoE-4E is the learned gating distribution.

For each market, the average routing weights of the four experts are computed:

- BSM;
- Residual MLP;
- Static MLP;
- Transformer.

Comparing these allocations between SPX and AAPL provides a direct view of whether the mixture relies on different expert structures across different option markets.

> Cross-market gating visualization will be added from the original experimental outputs.

---

## Expert Contribution Analysis

A post-hoc contribution analysis examines how predictions change when selected expert outputs are modified or removed.

For the **Static MLP** and **Transformer** experts, the selected expert output is removed and the remaining routing weights are renormalized.

For the **Residual MLP**, the original analysis attenuates the learned residual correction rather than fully removing the expert.

This analysis is therefore interpreted as a **sensitivity diagnostic for expert contribution and specialization**, rather than as a retrained architectural ablation study.

---

## Supplementary Implied-Volatility Analysis

The original project also included supplementary implied-volatility inversion and IVRMSE analyses.

These analyses are retained as supporting diagnostics rather than headline evaluation results because the original SPX and AAPL analysis scripts use slightly different filtering rules, volatility-search ranges, and data schemas.

They are therefore not interpreted as a fully standardized cross-market IVRMSE benchmark.

---

## Repository Structure

```text
RA-MoE-4E/
│
├── README.md
├── .gitignore
│
├── scripts/
│   └── train.py
│
├── src/
│   └── ra_moe/
│       │
│       ├── models/
│       │   ├── bsm.py
│       │   ├── mlp.py
│       │   ├── transformer.py
│       │   ├── gating.py
│       │   └── ra_moe.py
│       │
│       ├── data/
│       │   ├── schema.py
│       │   ├── preprocessing.py
│       │   ├── temporal.py
│       │   ├── features.py
│       │   ├── split.py
│       │   ├── dataset.py
│       │   └── pipeline.py
│       │
│       ├── training/
│       │   ├── losses.py
│       │   ├── epoch.py
│       │   └── workflow.py
│       │
│       └── evaluation/
│           └── metrics.py
│
└── analysis/
    └── [post-hoc analysis scripts]
```

The repository is being progressively refactored from the original experimental codebase while preserving the implemented research workflow.

---

## Running the Training Pipeline

The core training workflow can be launched with:

```bash
python scripts/train.py --data /path/to/cleaned_option_data.csv
```

For a smaller experimental run:

```bash
python scripts/train.py \
    --data /path/to/cleaned_option_data.csv \
    --sample-fraction 0.1
```

The default configuration preserves the main settings of the supplied final implementation.

Because the underlying OptionMetrics data are proprietary, the original datasets are not included in this repository.

---

## Reproducibility Notes

This repository is a modular refactor of the original experimental RA-MoE-4E codebase.

The refactoring aims to improve readability and organization while preserving the implemented model and training behavior.

Where the written methodology and supplied final implementation differ, the repository prioritizes the **implemented experimental behavior** rather than silently modifying the code to match the written description.

Examples include differences involving:

- Transformer positional encoding;
- temporal feature definitions;
- residual-expert feature specification;
- selected hyperparameters;
- training-stage descriptions;
- routing regularization behavior.

These differences are retained transparently rather than retrospectively altering the experiments that produced the original results.

---

## Limitations

The empirical evaluation focuses on SPX and AAPL options and therefore does not establish generalization across broader asset classes or market structures.

The temporal expert in the supplied implementation operates on fixed-length option-level histories and does not constitute a general market-level temporal representation.

The expert contribution analysis is post-hoc and does not retrain reduced architectures after removing individual experts.

Some supplementary analyses, including implied-volatility diagnostics, use market-specific preprocessing and inversion settings and should therefore not be interpreted as a unified cross-market evaluation protocol.

Finally, this repository represents a refactoring of an experimental research codebase rather than a production option-pricing library.

---

## Author

**Yitong Li**

Research interests: **Machine Learning for Science · Robust & Generalizable ML · Representation Learning · Biomedical AI**
