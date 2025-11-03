# Real-Time Pitch Recommendation Model

This project implements a real-time, context-aware pitch recommendation engine using pitch-level Statcast data. The objective is to assist pitching strategy decisions by estimating the probability of a favorable outcome for each available pitch type given the current game situation.
Unlike static run expectancy charts or traditional scouting reports, this model evaluates the complex interaction between pitch characteristics, batter tendencies, count leverage, and historical outcomes using machine learning.

## Methodology

We train a Gradient Boosting Classifier on historical pitch events enriched with contextual variables:
- Batter handedness
- Current count
- Pitcher handedness
- Base state
- Inning/score leverage
- Previous pitch sequencing features
- Swing/take tendencies
- Whiff and chase zones
- Command consistency indicators

Numerical features are standardized; categorical features are one-hot encoded. The architecture is modular so analysts can iterate on feature engineering easily.

We score pitches on a binary target reflecting *successful* outcomes (strike, whiff, out, etc.) without conflating balls in play with randomness unless the context suggests otherwise.

## Why a Bayesian-flavored approach?

Pitching is riddled with uncertainty. Batter behavior is not fixed; counts evolve pitch-to-pitch; and our confidence changes as new information arrives. Leveraging probability distributions rather than deterministic heuristics helps capture:

- Shrinkage toward league baselines in low-sample areas
- Stable inference on emerging pitch types
- Contextual priors (e.g., 0-2 vs 3-1)

This yields calibrated probabilities rather than noisy point estimates.

## Features

- Vectorized inference across all pitch types
- Safe handling of missing categorical data
- Automatic dtype hygiene
- Seamless future integration with:
  - Pitcher fatigue models
  - Zone-level command projections
  - Batter swing-decision models
  - Expected value of contact quality

The result is a tool that mirrors how elite pitching strategists think—without needing to sift through spreadsheets mid-inning.

## Example Usage

```python
best, probabilities = recommend_pitch_vectorized_safe(
    context_row=current_pitch_context,
    pitch_types=['FF','SL','CH','CU'],
    clf=clf,
    numeric_cols=numeric_features,
    categorical_cols=categorical_features
)
```

## Example Output

Recommended Pitch: SL
Probability Estimates:
FF: 0.47
SL: 0.61
CH: 0.55
CU: 0.44

Meaning: in this game state, throwing a slider yields the highest expected success probability.

## Repo Structure
```bash
├── data/               # Raw/clean Statcast extracts
├── notebooks/          # EDA, feature engineering, calibration
├── model/              # Serialized model artifacts + transforms
├── scripts/            # Inference utilities
└── README.md           # You're here
```

## Future Moves

Potential development paths:
- Zone-level location recommendation
- PitchCom incorporation, ordering suggestions by success prob.
- Run value optimization rather than binary success
- Batter-specific decay curves on pitch recognition
- Fatigue/command drift modeling across innings
- Front offices who integrate these layers gain a sustainable, compounding edge.

## Final Notes

Pitching strategy is probabilistic and adversarial. This project operationalizes that philosophy:
- Learn from context
- Update beliefs
- Optimize expected value on every pitch
