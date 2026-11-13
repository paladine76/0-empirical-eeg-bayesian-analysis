# empirical-eeg-bayesian-analysis

Analysis code for the behavioural association analyses in Hardiansyah et al., "Genetic influences on infant brain activity and its links to behavioural development". The study recorded EEG at 5 months in two Swedish infant cohorts, BATSS (twins from the general population) and EASE (infants at elevated or low familial likelihood of autism). It derived 216 EEG phenotypes from four measure types: relative band power (RBP), phase-amplitude coupling (PAC), functional connectivity (FC) and network-level metrics. Behaviour was assessed concurrently and at 36 months.

This repository covers the step that links 179 of these phenotypes to nine behavioural traits. Phenotypes correlated at |r| >= 0.85 are grouped by hierarchical clustering, and each trait is regressed on all predictors jointly under a regularised horseshoe prior, with a pair-level random intercept for twins. The fit is repeated 100 times over random draws of one member per cluster, and units ranked in the top ten by inclusion Bayes factor in at least 90% of repeats are shortlisted. Replicability is then tested over 10 random half-splits: units are selected in one half and estimated in the other, and directional support is the median held-out posterior probability that the coefficient has the training-half sign. The twin modelling of the EEG phenotypes is not part of this repository.

Participant data are not included in this repository.

## Structure

```
.
├── pyproject.toml
├── src/
│   └── bayesian_shrinkage_analyses/
│       ├── eeg_hierarchical_clustering.py   # groups phenotypes correlated at |r| >= 0.85
│       ├── cluster_labels.py                # reporting labels for the clusters
│       ├── ssvs_horseshoe.py                # regularised horseshoe regression (EASE)
│       ├── ssvs_horseshoe_twin.py           # regularised horseshoe with twin-pair random intercept (BATSS)
│       ├── stability_selection.py           # helpers: declustered predictor sets, standardisation
│       ├── pooled_summary.py                # full-sample shrinkage, repeated over declustering draws
│       ├── split_sample_validation.py       # train/held-out split (twin pairs kept together)
│       ├── shrinkage_backends.py            # common interface to the shrinkage samplers
│       ├── baseline_bayesian_lm.py          # held-out inference model (EASE)
│       ├── baseline_bayesian_lmm.py         # held-out inference model with pair random intercept (BATSS)
│       ├── split_selection_inference.py     # one split: selection in training half, inference in held-out half
│       ├── repeated_splits.py               # repeats the split analysis over 10 splits
│       └── split_support_summary.py         # directional support of shortlisted units across splits
└── tests/
    ├── conftest.py                          # synthetic test data
    ├── test_horseshoe.py
    └── test_clustering_and_labels.py
```

## Installation

Requires Python >= 3.10.

```
pip install -e ".[dev]"
```

## Tests

The tests use synthetic data only.

```
pytest
```
