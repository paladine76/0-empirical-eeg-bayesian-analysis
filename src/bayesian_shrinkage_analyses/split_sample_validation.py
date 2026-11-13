"""
Split-sample validation: selection on one half, inference on the held-out half.

This is the one route to per-predictor inference that survives scrutiny. The
selection procedure (declustered repeated horseshoe refits) runs on the
TRAIN half only; the selected predictors are then fitted once on the TEST
half with a plain, non-shrinking Bayesian model. Because the test half never
informed which predictors were chosen, the pd / credible interval / Bayes
factor reported from it are ordinary post-selection-free quantities -- no
enrichment, no double-dipping, no caveat needed.

Contrast with refitting the selected set on the FULL sample: there the
predictors survived selection precisely because they looked most associated
in that same data, so pd is inflated by an unquantifiable amount. Freezing
the selection does not fix this, because the selection was a function of the
data being refitted. Splitting does fix it, at the cost of sample size.

STAGE A -- selection (train half)
    n_select_repeats declustered refits: each draws one representative per
    cluster plus all singletons, fits the regularised horseshoe, ranks by
    inclusion Bayes factor, and records the top-K. A unit (cluster label or
    singleton) is selected when it reaches the top-K in >= select_threshold
    of repeats. Identical in form to the main analysis loop, just on half the
    data.

STAGE B -- inference (test half)
    Each selected unit is mapped back to one concrete variable. For a
    cluster, the member with the highest mean train-half BF10 is used, so the
    choice is made from train data only and leaks nothing. Those variables
    are fitted jointly on the test half with fit_bayesian_lm /
    fit_bayesian_lmm -- Normal(0, prior_scale^2) on every coefficient, no
    shrinkage, nothing selected away -- and beta, pd, 95% CI and
    Savage-Dickey BF10 are read off.

REPEATED SPLITS
    A single split is noisy, especially for EASE (n=88 -> 44/44). The whole
    cycle is therefore repeated over n_splits random splits and aggregated.

    The headline number is
        replication_rate = n_pd_pass / n_splits
    i.e. out of ALL splits run, the fraction in which this unit was both
    selected on train AND reached pd >= pd_threshold on test. The
    denominator is n_splits for every unit, so units are directly
    comparable: a unit selected in 1 split of 20 that passes there scores
    0.05, not 1.00. Splits in which a unit was not selected count as
    non-replications, which is the honest reading -- failing to be picked
    is itself a failure to replicate.

    conditional_rate = n_pd_pass / n_splits_tested is also reported for
    reference ("when it was selected, how often did it hold up?"), but its
    denominator varies per unit, so it is not comparable across units and
    is not the number to report.

    Splits are parallelised over cores with joblib, and the whole job is
    embarrassingly parallel, so wall time is roughly serial time / n_jobs.

SPLITTING RULES
    BATSS: split at the PAIR level -- both members of a twin pair land in the
    same half. Splitting within a pair would put correlated observations on
    both sides, and the test half would no longer be independent of the
    train half, which is the whole point of the exercise.
    Bernoulli outcomes: the split is stratified on y so both halves carry
    cases; an unstratified split of a rare binary outcome can leave a half
    with almost no positives.
"""

import os

# Must precede any numpy/jax import in this module. Each worker process
# imports it fresh, so this pins one BLAS thread per worker and stops N
# workers from oversubscribing the cores.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gc
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def make_split(y, pair_ids, cohort, family, train_frac, rng):
    """Return (train_rows, test_rows).

    BATSS splits whole pairs so the halves stay independent; bernoulli
    outcomes are stratified so both halves contain cases.
    """
    n = len(y)

    if cohort == "BATSS":
        upairs = pd.unique(pair_ids)
        if family == "bernoulli":
            # stratify pairs by whether the pair contains any case
            has_case = np.array([y[pair_ids == p].max() > 0 for p in upairs])
            train_pairs = []
            for mask in (has_case, ~has_case):
                grp = upairs[mask]
                k = int(round(train_frac * len(grp)))
                train_pairs.extend(rng.choice(grp, size=k, replace=False))
            train_pairs = np.array(train_pairs)
        else:
            k = int(round(train_frac * len(upairs)))
            train_pairs = rng.choice(upairs, size=k, replace=False)
        train_rows = np.flatnonzero(np.isin(pair_ids, train_pairs))
        test_rows = np.flatnonzero(~np.isin(pair_ids, train_pairs))
        return train_rows, test_rows

    if family == "bernoulli":
        train_rows = []
        for val in np.unique(y):
            grp = np.flatnonzero(y == val)
            k = int(round(train_frac * len(grp)))
            train_rows.extend(rng.choice(grp, size=k, replace=False))
        train_rows = np.sort(np.array(train_rows))
    else:
        k = int(round(train_frac * n))
        train_rows = np.sort(rng.choice(n, size=k, replace=False))

    test_rows = np.setdiff1d(np.arange(n), train_rows)
    return train_rows, test_rows


def _standardize(X):
    sd = X.std(0, ddof=1)
    ok = np.isfinite(sd) & (sd > 0)
    return (X - X.mean(0)) / np.where(ok, sd, 1.0), ok.all()


# ---------------------------------------------------------------------------
# one split: select on train, test on held-out (module level so it pickles)
# ---------------------------------------------------------------------------
def _one_split(split_i, sub_values, preds_index, y_obs, pair_ids, cluster_vars,
               singleton_vars, predictor_to_label, cohort, family, top_k,
               n_select_repeats, select_threshold, train_frac, prior_scale,
               sel_warmup, sel_samples, test_warmup, test_samples, num_chains,
               split_seed):
    """Returns (split_i, selection rows, test rows, diagnostics dict)."""
    import jax
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst
    from src.bayesian_shrinkage_analyses import baseline_bayesian_lm as blm
    from src.bayesian_shrinkage_analyses import baseline_bayesian_lmm as blmm

    t0 = time.time()
    rng = np.random.default_rng(split_seed + split_i)
    train_rows, test_rows = make_split(y_obs, pair_ids, cohort, family, train_frac, rng)

    # ---------------- STAGE A: selection on the TRAIN half ----------------
    hit_counts = defaultdict(int)       # unit -> times in top-K
    bf_by_variable = defaultdict(list)  # variable -> train BF10s (for rep choice)
    n_used = 0

    for r in range(n_select_repeats):
        reps = [rng.choice(cluster_vars[cid]) for cid in sorted(cluster_vars)]
        preds_r = list(reps) + list(singleton_vars)
        cols = [preds_index[v] for v in preds_r]

        X, ok = _standardize(sub_values[np.ix_(train_rows, cols)])
        if not ok:
            continue
        yr = y_obs[train_rows]
        if family == "normal":
            yr = (yr - yr.mean()) / yr.std()

        seed_r = int(rng.integers(1, 2**31 - 1))
        if cohort == "BATSS":
            mcmc, _ = hst.fit_twin_horseshoe(
                X, yr, pair_ids[train_rows], family=family,
                num_warmup=sel_warmup, num_samples=sel_samples,
                num_chains=num_chains, seed=seed_r, progress_bar=False)
            bf = hst.inclusion_bayes_factor(mcmc, len(train_rows))
        else:
            mcmc = hs.fit_horseshoe(
                X, yr, family=family,
                num_warmup=sel_warmup, num_samples=sel_samples,
                num_chains=num_chains, seed=seed_r, progress_bar=False)
            bf = hs.inclusion_bayes_factor(mcmc, len(train_rows))

        for v, b in zip(preds_r, bf):
            bf_by_variable[v].append(float(b))

        names = [predictor_to_label.get(p, p) for p in preds_r]
        for i in np.argsort(-bf)[:top_k]:
            hit_counts[names[i]] += 1
        n_used += 1

        del mcmc
        gc.collect()
        jax.clear_caches()

    if n_used == 0:
        raise RuntimeError(f"split {split_i}: no usable selection repeats")

    freq = {u: c / n_used for u, c in hit_counts.items()}
    selected = sorted([u for u, f in freq.items() if f >= select_threshold],
                      key=lambda u: -freq[u])

    sel_rows = [{"split": split_i, "unit": u, "train_freq": freq[u],
                 "selected": u in set(selected)}
                for u in sorted(freq, key=lambda x: -freq[x])]

    if not selected:
        return split_i, sel_rows, [], dict(
            split=split_i, n_train=len(train_rows), n_test=len(test_rows),
            n_selected=0, seconds=round(time.time() - t0, 1))

    # map each selected unit to ONE concrete variable, using train BF10 only
    label_members = defaultdict(list)
    for v, lab in predictor_to_label.items():
        label_members[lab].append(v)

    chosen, skipped = [], []
    for u in selected:
        if u in label_members:
            members = [m for m in label_members[u] if bf_by_variable.get(m)]
            if not members:
                skipped.append(u)
                continue
            chosen.append((u, max(members, key=lambda m: np.mean(bf_by_variable[m]))))
        else:
            chosen.append((u, u))

    units = [u for u, _ in chosen]
    variables = [v for _, v in chosen]

    # ---------------- STAGE B: inference on the HELD-OUT half ----------------
    cols = [preds_index[v] for v in variables]
    Xte, ok = _standardize(sub_values[np.ix_(test_rows, cols)])
    if not ok:
        raise RuntimeError(f"split {split_i}: zero-variance predictor in test half")
    yte = y_obs[test_rows]
    if family == "normal":
        yte = (yte - yte.mean()) / yte.std()

    seed_t = int(rng.integers(1, 2**31 - 1))
    if cohort == "BATSS":
        mcmc_t, _ = blmm.fit_bayesian_lmm(
            Xte, yte, pair_ids[test_rows], family=family, prior_scale=prior_scale,
            num_warmup=test_warmup, num_samples=test_samples,
            num_chains=num_chains, seed=seed_t, progress_bar=False)
        beta, pd_, lo, hi = blmm.posterior_summary(mcmc_t)
        bf10 = blmm.savage_dickey_bf10(mcmc_t, prior_scale=prior_scale)
        icc = blmm.icc_summary(mcmc_t, family=family)
    else:
        mcmc_t = blm.fit_bayesian_lm(
            Xte, yte, family=family, prior_scale=prior_scale,
            num_warmup=test_warmup, num_samples=test_samples,
            num_chains=num_chains, seed=seed_t, progress_bar=False)
        beta, pd_, lo, hi = blm.posterior_summary(mcmc_t)
        bf10 = blm.savage_dickey_bf10(mcmc_t, prior_scale=prior_scale)
        icc = None

    test_rows_out = [
        {"split": split_i, "unit": u, "variable": v, "train_freq": freq[u],
         "beta_mean": float(beta[j]), "pd": float(pd_[j]),
         "ci_lower": float(lo[j]), "ci_upper": float(hi[j]),
         "BF10": float(bf10[j]),
         "ci_excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
        for j, (u, v) in enumerate(zip(units, variables))
    ]

    del mcmc_t
    gc.collect()
    jax.clear_caches()

    return split_i, sel_rows, test_rows_out, dict(
        split=split_i, n_train=len(train_rows), n_test=len(test_rows),
        n_selected=len(units), n_skipped=len(skipped),
        icc=icc, seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_split_sample_validation(sub, y, preds_all, cluster_vars, singleton_vars,
                                predictor_to_label, *,
                                cohort="EASE", family="normal", pair_ids=None,
                                top_k=10, n_splits=20, n_select_repeats=25,
                                select_threshold=0.90, train_frac=0.5,
                                pd_threshold=0.95, prior_scale=2.5,
                                sel_warmup=400, sel_samples=600,
                                test_warmup=1000, test_samples=2000,
                                num_chains=1, split_seed=20260918,
                                n_jobs=8, out_prefix="splitval", out_dir=".",
                                verbose=True):
    """
    sub                : pandas DataFrame, already filtered to complete cases on
                         the outcome (sub_decl / sub_decl_t from the notebook)
    y                  : outcome vector aligned to sub (ys_decl / ys_decl_t)
    preds_all          : every EEG variable that may be drawn -- all cluster
                         members plus all singletons (eeg_var_names), NOT the
                         single declustered set
    cluster_vars       : CLUSTER_VARS   {cluster_id: [variable names]}
    singleton_vars     : SINGLETON_VARS
    predictor_to_label : PREDICTOR_TO_LABEL from clab.build_predictor_to_label_map
    pair_ids           : required when cohort == "BATSS"

    Returns (summary_df, test_df, selection_df, meta).
    summary_df is the headline table, sorted by replication_rate =
    n_pd_pass / n_splits: out of all splits run, the fraction in which the
    unit was selected on train AND reached pd >= pd_threshold on the
    held-out half.
    """
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    y = np.asarray(y, dtype=float)
    n = len(y)
    if cohort == "BATSS":
        if pair_ids is None:
            raise ValueError("pair_ids is required for cohort='BATSS'")
        pair_ids = np.asarray(pair_ids)
    else:
        pair_ids = np.arange(n)

    sub_values = sub[preds_all].values.astype(float)
    preds_index = {v: j for j, v in enumerate(preds_all)}

    probe = np.random.default_rng(split_seed)
    tr, te = make_split(y, pair_ids, cohort, family, train_frac, probe)
    if verbose:
        print(f"cohort={cohort}  n={n}  train~{len(tr)}  test~{len(te)}  "
              f"predictors per selection fit={len(cluster_vars) + len(singleton_vars)}")
        print(f"splits={n_splits}  select_repeats={n_select_repeats}  "
              f"select_threshold={select_threshold:.0%}  n_jobs={n_jobs}  "
              f"total fits={n_splits * (n_select_repeats + 1)}")

    common = dict(sub_values=sub_values, preds_index=preds_index, y_obs=y,
                  pair_ids=pair_ids, cluster_vars=cluster_vars,
                  singleton_vars=list(singleton_vars),
                  predictor_to_label=predictor_to_label, cohort=cohort,
                  family=family, top_k=top_k, n_select_repeats=n_select_repeats,
                  select_threshold=select_threshold, train_frac=train_frac,
                  prior_scale=prior_scale, sel_warmup=sel_warmup,
                  sel_samples=sel_samples, test_warmup=test_warmup,
                  test_samples=test_samples, num_chains=num_chains,
                  split_seed=split_seed)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)(
        delayed(_one_split)(i, **common) for i in range(n_splits)
    )

    sel_rows, test_rows, diags = [], [], []
    for _, s, t, d in results:
        sel_rows.extend(s)
        test_rows.extend(t)
        diags.append(d)

    selection_df = pd.DataFrame(sel_rows)
    test_df = pd.DataFrame(test_rows)
    diag_df = pd.DataFrame(diags).sort_values("split")

    # ---- aggregate across splits ----
    n_sel = (selection_df[selection_df["selected"]]
             .groupby("unit").size().rename("n_splits_selected"))

    if len(test_df):
        g = test_df.groupby("unit")
        agg = pd.DataFrame({
            "n_splits_tested": g.size(),
            "mean_train_freq": g["train_freq"].mean(),
            "mean_beta": g["beta_mean"].mean(),
            "sign_consistency": g["beta_mean"].apply(
                lambda b: max((b > 0).mean(), (b < 0).mean())),
            "mean_pd": g["pd"].mean(),
            "median_pd": g["pd"].median(),
            "n_pd_pass": g["pd"].apply(lambda s: int((s >= pd_threshold).sum())),
            "median_BF10": g["BF10"].median(),
            "n_ci_excl_zero": g["ci_excludes_zero"].sum(),
        })
        # denominator is ALL splits, identical for every unit, so units are
        # directly comparable; a split in which the unit was not selected
        # counts as a non-replication.
        agg["replication_rate"] = agg["n_pd_pass"] / n_splits
        # same numerator, per-unit denominator -- reference only, not comparable
        agg["conditional_rate"] = agg["n_pd_pass"] / agg["n_splits_tested"]
        summary = (agg.join(n_sel, how="left")
                      .reset_index()
                      .sort_values(["replication_rate", "median_pd"],
                                   ascending=[False, False])
                      .reset_index(drop=True))
    else:
        summary = pd.DataFrame(columns=["unit", "n_splits_tested", "replication_rate"])

    summary.to_csv(os.path.join(out_dir, f"{out_prefix}_summary.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{out_prefix}_per_split_tests.csv"), index=False)
    selection_df.to_csv(os.path.join(out_dir, f"{out_prefix}_selection.csv"), index=False)

    meta = dict(cohort=cohort, family=family, n=int(n),
                n_train=int(len(tr)), n_test=int(len(te)),
                top_k=top_k, n_splits=n_splits,
                n_select_repeats=n_select_repeats,
                select_threshold=select_threshold, train_frac=train_frac,
                pd_threshold=pd_threshold, prior_scale=prior_scale,
                split_seed=split_seed,
                mean_n_selected=float(diag_df["n_selected"].mean()),
                splits_with_no_selection=int((diag_df["n_selected"] == 0).sum()),
                wall_minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(out_dir, f"{out_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        cols = ["unit", "n_splits_selected", "n_pd_pass", "replication_rate",
                "conditional_rate", "mean_beta", "sign_consistency",
                "median_pd", "median_BF10"]
        cols = [c for c in cols if c in summary.columns]
        print("\n" + summary[cols].round(3).to_string(index=False))
        print(f"\nmean selected per split: {meta['mean_n_selected']:.1f}"
              f"   splits with none: {meta['splits_with_no_selection']}")
        print(f"wall: {meta['wall_minutes']:.1f} min")

    return summary, test_df, selection_df, meta
