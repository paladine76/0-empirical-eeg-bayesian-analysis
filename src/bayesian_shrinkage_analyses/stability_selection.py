"""
Stability selection (Meinshausen & Buhlmann, 2010) for the declustered EEG
predictor set.

ONE statistic, ONE number per unit. Each repeat draws a subsample of the
observations WITHOUT replacement, draws one representative per cluster plus
all singletons, fits the regularised horseshoe once, ranks by inclusion
Bayes factor, and records the top-K. The selection frequency across repeats
is the result. Units with frequency >= pi_thr form the stable set.

This is the same selection rate the main analysis already reports, with one
change: the observations are resampled each repeat instead of held fixed.
That change is what makes the frequency informative. With fixed data the
only things varying are the cluster representative and the MCMC seed, so the
ranking barely moves and frequencies saturate at 1.0 -- which is exactly
what the earlier fixed-data runs showed (seven units at 100/100). Subsampling
makes the frequency measure sensitivity to WHICH subjects are in the sample,
which is the thing that bears on generalisability.

ERROR CONTROL
    M&B Theorem 1 bounds the expected number of falsely selected variables:

        E(V) <= [1 / (2*pi_thr - 1)] * q^2 / p

    V    number of falsely selected units in the stable set (NOT within the
         top-K of any single subsample -- a common misreading)
    q    average number of units selected per subsample; here exactly top_k,
         since each fit contributes a hard top-K and only one representative
         per cluster is ever in the model, so the K entries always map to K
         distinct reporting units
    p    number of candidate units = n_clusters + n_singletons (122 here),
         i.e. the collapsed reporting space in which selection and counting
         both happen. Using the uncollapsed 179 would give a smaller, more
         flattering, and wrong bound.

    The bound REQUIRES pi_thr > 1/2 and degrades hyperbolically as pi_thr
    approaches it: at 0.51 the bound is ~41 for this setup, which permits
    essentially everything and says nothing. M&B themselves recommend
    pi_thr in (0.6, 0.9). run_stability_selection() prints the bound across
    a grid of thresholds so the choice can be made with the numbers in view.

    Two honest caveats for the Methods section:
      1. M&B define q over a regularisation path (lambda in Lambda). This
         procedure does not sweep lambda -- it takes a fixed top-K from a
         single horseshoe fit, so q = top_k deterministically. That is the
         simplest possible case, but the procedure is an analogue of theirs
         rather than a literal instance. Word it as "following the approach
         of", not "Theorem 1 applies".
      2. The bound assumes exchangeability of the null units. With
         correlated EEG variables that is not fully credible even after
         clustering at |r| >= 0.85. Shah & Samworth (2013) drop this
         assumption; running their complementary-pairs variant as a
         supplementary check answers the objection.

    The bound is also per outcome. Nine outcomes, each with E(V) <= ~1,
    gives an expected total of ~9 false retentions across the paper by
    linearity -- state that rather than leave it to be found.

SUBSAMPLING
    frac defaults to 0.632, which gives each fit the same number of unique
    observations as a bootstrap draw would, without duplicated rows. The
    duplicates matter here: a bootstrap would repeat individual infants,
    and for the twin data a repeated infant inflates its pair to three or
    four members with perfectly correlated entries, biasing sigma_pair down.
    Subsampling avoids that entirely.

    The fraction trades off two things. Higher frac means more power per
    fit but more overlap between repeats -- two subsamples at fraction f
    share a proportion f of each other's observations, so at f -> 1 the
    repeats converge on identical data and the frequency saturates again.
    0.5 is what M&B's theory is written for; 0.632 matches bootstrap's
    unique-observation count; 0.75 maximises per-fit power at the cost of
    the most overlap.

    BATSS subsamples whole PAIRS, never splitting a pair, so the random
    intercept sees intact families. Bernoulli outcomes are stratified so
    every subsample carries cases.
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
# subsampling
# ---------------------------------------------------------------------------
def make_subsample(y, pair_ids, cohort, family, frac, rng):
    """Row indices of one subsample drawn WITHOUT replacement.

    BATSS draws whole pairs. Bernoulli outcomes are stratified on y (on
    whether the pair contains a case, for BATSS).
    """
    n = len(y)

    if cohort == "BATSS":
        upairs = pd.unique(pair_ids)
        if family == "bernoulli":
            has_case = np.array([y[pair_ids == p].max() > 0 for p in upairs])
            keep = []
            for mask in (has_case, ~has_case):
                grp = upairs[mask]
                k = max(1, int(round(frac * len(grp)))) if len(grp) else 0
                if k:
                    keep.extend(rng.choice(grp, size=k, replace=False))
            keep = np.array(keep)
        else:
            k = int(round(frac * len(upairs)))
            keep = rng.choice(upairs, size=k, replace=False)
        return np.flatnonzero(np.isin(pair_ids, keep))

    if family == "bernoulli":
        rows = []
        for val in np.unique(y):
            grp = np.flatnonzero(y == val)
            k = max(1, int(round(frac * len(grp))))
            rows.extend(rng.choice(grp, size=k, replace=False))
        return np.sort(np.array(rows))

    k = int(round(frac * n))
    return np.sort(rng.choice(n, size=k, replace=False))


def _standardize(X):
    sd = X.std(0, ddof=1)
    ok = np.isfinite(sd) & (sd > 0)
    return (X - X.mean(0)) / np.where(ok, sd, 1.0), bool(ok.all())


# ---------------------------------------------------------------------------
# M&B Theorem 1
# ---------------------------------------------------------------------------
def mb_error_bound(q, p, pi_thr):
    """E(V) <= [1/(2*pi_thr - 1)] * q^2 / p.  NaN when pi_thr <= 0.5."""
    if pi_thr <= 0.5:
        return float("nan")
    return (1.0 / (2.0 * pi_thr - 1.0)) * (q ** 2) / p


# ---------------------------------------------------------------------------
# one repeat = one subsample, one fit (module level so it pickles to workers)
# ---------------------------------------------------------------------------
def _build_decluster_predictor_list(cluster_vars, singleton_vars, seed):
    """Verbatim reimplementation of the notebook's helper of the same name,
    so a given seed draws exactly the representatives the original loop would."""
    rng = np.random.default_rng(seed)
    reps = [rng.choice(cluster_vars[cid]) for cid in sorted(cluster_vars.keys())]
    predictor_list = list(reps) + list(singleton_vars)
    assert len(predictor_list) == len(set(predictor_list)), "duplicate predictor in decluster list"
    return predictor_list


def _results_dataframe(module, mcmc, predictor_names, n, family):
    """Verbatim reimplementation of the notebook's results_dataframe(), including
    the sort-then-round order. The rounding matters: the caller re-sorts on the
    ROUNDED BF10 before taking the top-K, so ranking on raw BF10 instead could
    break ties differently."""
    bf10 = module.inclusion_bayes_factor(mcmc, n)
    beta_cond, pd_, ci_lo, ci_hi, n_included = module.conditional_posterior_summary(mcmc, n)
    df = pd.DataFrame({
        "predictor": list(predictor_names),
        "BF10": bf10, "beta_mean": beta_cond, "pd": pd_,
        "ci_lower": ci_lo, "ci_upper": ci_hi, "n_included": n_included,
    })
    df = df.sort_values("BF10", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df = df.round(3)
    return df


def _one_repeat(rep_i, sub_values, preds_index, y_obs, pair_ids, cluster_vars,
                singleton_vars, predictor_to_label, cohort, family, top_k,
                frac, num_warmup, num_samples, num_chains, seed_offset,
                subsample_seed, check_convergence):
    """One repeat = one subsample, one fit. Mirrors the notebook's
    run_stability_loop() body; the ONLY departure is that the fit is on a
    subsample of rows rather than all of them.

    Returns (rep_i, top-K reporting names, {unit: beta}, n_rows, rhat, seconds).
    """
    import jax
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst

    t0 = time.time()
    module = hst if cohort == "BATSS" else hs

    # seed_i drives BOTH the representative draw and the MCMC seed, exactly as
    # in the original loop (seed_i = seed_offset + run_i).
    seed_i = seed_offset + rep_i

    # The subsample is drawn from a SEPARATE stream so it does not perturb the
    # representative draw -- at frac=1.0 the representatives, the MCMC seed and
    # hence the whole fit are then identical to the original loop's.
    rows = make_subsample(y_obs, pair_ids, cohort, family, frac,
                          np.random.default_rng(subsample_seed + rep_i))

    preds_i = _build_decluster_predictor_list(cluster_vars, singleton_vars, seed=seed_i)
    cols = [preds_index[v] for v in preds_i]

    Xi = sub_values[np.ix_(rows, cols)]
    # The original asserts np.isfinite(Xi).all(). Subsampling can additionally
    # leave a column constant within the drawn rows, which would divide by zero,
    # so that case is skipped rather than asserted.
    Xsi, ok = _standardize(Xi)
    if not ok or not np.isfinite(Xi).all():
        return rep_i, None, None, len(rows), np.nan, time.time() - t0

    ys = y_obs[rows]
    if family == "normal":
        ys = (ys - ys.mean()) / ys.std()

    if cohort == "BATSS":
        mcmc, _ = module.fit_twin_horseshoe(
            Xsi, ys, pair_ids[rows], family=family, seed=seed_i,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=False)
    else:
        mcmc = module.fit_horseshoe(
            Xsi, ys, family=family, seed=seed_i,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=False)

    rhat = np.nan
    if check_convergence and num_chains > 1:
        # same quantity check_rhat() prints, recorded instead of printed so the
        # parallel workers do not interleave output
        import arviz as az
        rhat = float(az.summary(az.from_numpyro(mcmc), round_to=None)["r_hat"].max())

    # n is the number of observations actually fitted. kappa = 1/(1+n*tau^2*
    # lam_tilde^2), so passing the full-sample n here while fitting a subsample
    # would misstate the shrinkage. This is the one argument subsampling forces
    # to change.
    res_i = _results_dataframe(module, mcmc, preds_i, len(rows), family)
    res_i["predictor"] = res_i["predictor"].map(
        lambda p: predictor_to_label.get(p, p))

    topk = res_i.sort_values("BF10", ascending=False)["predictor"].head(top_k).tolist()
    betas = dict(zip(res_i["predictor"], res_i["beta_mean"].astype(float)))

    del mcmc, res_i, Xi, Xsi
    gc.collect()
    jax.clear_caches()
    return rep_i, topk, betas, len(rows), rhat, time.time() - t0


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_stability_selection(sub, y, preds_all, cluster_vars, singleton_vars,
                            predictor_to_label, *,
                            cohort="EASE", family="normal", pair_ids=None,
                            top_k=10, n_repeats=100, frac=0.632,
                            pi_thr=0.75,
                            thr_grid=(0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
                            num_warmup=2000, num_samples=5000, num_chains=6,
                            seed_offset=100, subsample_seed=20260921,
                            check_convergence=True, n_jobs=1,
                            out_prefix="stabsel", out_dir=".", verbose=True):
    """
    sub                : pandas DataFrame already filtered to complete cases on
                         the outcome (sub_decl / sub_decl_t from the notebook)
    y                  : outcome vector aligned to sub (ys_decl / ys_decl_t)
    preds_all          : every EEG variable that may be drawn -- all cluster
                         members plus all singletons (eeg_var_names), NOT the
                         single declustered set, since representatives are
                         redrawn every repeat
    cluster_vars       : CLUSTER_VARS   {cluster_id: [variable names]}
    singleton_vars     : SINGLETON_VARS
    predictor_to_label : PREDICTOR_TO_LABEL from clab.build_predictor_to_label_map
    pair_ids           : required when cohort == "BATSS"
    pi_thr             : threshold defining the stable set
    thr_grid           : thresholds to tabulate (units clearing + E(V) bound)

    Returns (summary_df, threshold_df, meta).
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

    if pi_thr <= 0.5:
        raise ValueError("pi_thr must exceed 0.5 or the M&B bound is undefined")

    sub_values = sub[preds_all].values.astype(float)
    preds_index = {v: j for j, v in enumerate(preds_all)}
    p_units = len(cluster_vars) + len(singleton_vars)

    probe = make_subsample(y, pair_ids, cohort, family, frac,
                           np.random.default_rng(subsample_seed))
    if verbose:
        print(f"cohort={cohort}  n={n}  subsample={len(probe)} ({frac:.3f})  "
              f"candidate units p={p_units}  top_k={top_k}")
        print(f"repeats={n_repeats}  pi_thr={pi_thr:.2f}  "
              f"MCMC={num_warmup}+{num_samples} x {num_chains} chains  "
              f"n_jobs={n_jobs}  total fits={n_repeats}")

    # Each worker asks numpyro for num_chains parallel chains, so n_jobs
    # workers demand n_jobs*num_chains at once. Oversubscribing makes every
    # fit slower; the original loop was serial with 6 chains.
    ncpu = os.cpu_count() or 1
    if n_jobs * num_chains > ncpu and verbose:
        print(f"  NOTE: n_jobs({n_jobs}) x num_chains({num_chains}) = "
              f"{n_jobs * num_chains} > {ncpu} cores -- oversubscribed. Either "
              f"keep n_jobs=1 (chains parallelise) or lower num_chains, which "
              f"changes the fit.")

    common = dict(sub_values=sub_values, preds_index=preds_index, y_obs=y,
                  pair_ids=pair_ids, cluster_vars=cluster_vars,
                  singleton_vars=list(singleton_vars),
                  predictor_to_label=predictor_to_label, cohort=cohort,
                  family=family, top_k=top_k, frac=frac,
                  num_warmup=num_warmup, num_samples=num_samples,
                  num_chains=num_chains, seed_offset=seed_offset,
                  subsample_seed=subsample_seed,
                  check_convergence=check_convergence)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10 if verbose else 0)(
        delayed(_one_repeat)(i, **common) for i in range(n_repeats)
    )

    hits = defaultdict(int)
    betas = defaultdict(list)
    q_per_repeat, rows_per_repeat, secs, rhats = [], [], [], []
    n_used = 0
    for _, topk, b, nrows, rh, dt in results:
        secs.append(dt)
        if topk is None:          # zero-variance predictor in that subsample
            continue
        for u in set(topk):
            hits[u] += 1
        for u, v in b.items():    # beta for ALL units, as the original records
            betas[u].append(v)
        q_per_repeat.append(len(set(topk)))
        rows_per_repeat.append(nrows)
        rhats.append(rh)
        n_used += 1

    if n_used == 0:
        raise RuntimeError("no usable repeats")

    # q for the bound: measured, not assumed. Equals top_k unless a subsample
    # was dropped or two top-K entries collapsed onto the same label.
    q = float(np.mean(q_per_repeat))

    # every candidate unit gets a row, including those never selected (freq 0)
    summary = pd.DataFrame([
        {"unit": u,
         "n_selected": hits.get(u, 0),
         "selection_freq": hits.get(u, 0) / n_used,
         "mean_beta": float(np.mean(betas[u])) if betas[u] else np.nan,
         "beta_sd": float(np.std(betas[u], ddof=1)) if len(betas[u]) > 1 else 0.0,
         "sign_consistency": (max((np.array(betas[u]) > 0).mean(),
                                  (np.array(betas[u]) < 0).mean())
                              if betas[u] else np.nan),
         "n_runs_present": len(betas[u]),
         "stable": (hits.get(u, 0) / n_used) >= pi_thr}
        for u in betas
    ]).sort_values("selection_freq", ascending=False).reset_index(drop=True)

    threshold_df = pd.DataFrame([
        {"pi_thr": t,
         "n_units_stable": int((summary["selection_freq"] >= t).sum()),
         "EV_bound": mb_error_bound(q, p_units, t)}
        for t in thr_grid
    ])

    summary.to_csv(os.path.join(out_dir, f"{out_prefix}_summary.csv"), index=False)
    threshold_df.to_csv(os.path.join(out_dir, f"{out_prefix}_thresholds.csv"), index=False)

    n_stable = int(summary["stable"].sum())
    meta = dict(cohort=cohort, family=family, n=int(n),
                subsample_size=int(np.mean(rows_per_repeat)), frac=frac,
                top_k=top_k, q_measured=q, p_units=p_units,
                n_repeats=n_repeats, n_repeats_used=n_used,
                pi_thr=pi_thr, n_stable=n_stable,
                EV_bound=mb_error_bound(q, p_units, pi_thr),
                # informativeness check: frequencies bunched at 0 or 1 mean the
                # subsample fraction is wrong for this cohort
                n_units_touched=int(len(summary)),
                n_units_midrange=int(((summary["selection_freq"] > 0.1) &
                                      (summary["selection_freq"] < 0.9)).sum()),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains,
                max_rhat=(float(np.nanmax(rhats)) if len(rhats)
                          and not np.all(np.isnan(rhats)) else None),
                seed_offset=seed_offset, subsample_seed=subsample_seed,
                wall_minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(out_dir, f"{out_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        print("\n" + summary.head(20).round(3).to_string(index=False))
        print(f"\nthreshold grid (q={q:.2f}, p={p_units}):")
        print(threshold_df.round(3).to_string(index=False))
        print(f"\nstable set at pi_thr={pi_thr:.2f}: {n_stable} units, "
              f"E(V) <= {meta['EV_bound']:.2f}")
        print(f"units with frequency in (0.1, 0.9): {meta['n_units_midrange']} "
              f"of {meta['n_units_touched']} touched   "
              f"[bunching at the ends means frac is wrong for this cohort]")
        print(f"wall: {meta['wall_minutes']:.1f} min")

    return summary, threshold_df, meta
