"""
Pooled posterior summary across the repeated-shrinkage runs.

Post-processing companion to the notebook's repeated-shrinkage loop
(run_stability_loop). Same fitting path -- full sample every repeat, one
representative drawn per cluster, seed_i = seed_offset + run_i driving both
the draw and the sampler -- but instead of recording only which units
reached the top-K, it keeps each run's posterior draws for every unit and
pools them across runs.

No subsampling: each repeat fits all n rows, exactly as run_stability_loop
does. The only thing varying between repeats is which representative each
cluster contributes and the MCMC seed.

WHAT "POOLED" MEANS HERE
    Each repeat's posterior is one component; concatenating the draws across
    repeats gives the mixture posterior over the whole procedure. Median,
    credible interval and pd are then all read off THAT one pooled set, so
    they are mutually consistent by construction -- the median cannot fall
    outside its own interval, and pd agrees with which side of zero the
    interval sits on.

    The alternative (averaging each run's summary statistics) is not the same
    thing: the spread of per-run point estimates measures run-to-run
    variability of an estimator, not posterior uncertainty. That quantity is
    reported separately as beta_run_sd so both readings are available.

CONDITIONAL VS ALL DRAWS
    conditional=True (default) pools only draws where kappa_j < 0.5 -- the
    same included-draw subset conditional_posterior_summary() uses, so the
    numbers are on the same footing as everything else already reported.
    Units never included in any run get NaN, which is informative rather
    than a failure.
    conditional=False pools every draw. The horseshoe shrinks most
    coefficients to near zero, so those distributions pile up at 0 and pd
    collapses toward 0.5 for all but the strongest units.

pd IS COMPUTED FROM THE POOLED DRAWS, not from the interval -- pd = the
fraction of pooled draws on the dominant side of zero, max(P(b>0), P(b<0)).
An interval alone cannot give it. Both come from the same draws, so they
always agree.
"""

import os

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

from src.bayesian_shrinkage_analyses.ssvs_horseshoe import prior_inclusion_baseline
from src.bayesian_shrinkage_analyses.stability_selection import (
    _build_decluster_predictor_list, _standardize)


def _escape_rate(incl, n_used, draws_per_repeat):
    """Pooled proportion of draws with kappa<0.5, over all repeats at once
    (not an average of per-repeat rates). Counted before the max_draws cap,
    so capping the stored draws does not bias it."""
    if not draws_per_repeat:
        return np.nan
    return incl / (n_used * draws_per_repeat)


def _bf10(rate, prior_odds, eps=1e-12):
    """Inclusion Bayes factor from the pooled escape rate. Same
    posterior-odds / prior-odds construction as
    ssvs_horseshoe.inclusion_bayes_factor()."""
    if not np.isfinite(rate):
        return np.nan
    return float((rate / max(1.0 - rate, eps)) / prior_odds)


# ---------------------------------------------------------------------------
# worker: one subsample, one fit, keep the draws
# ---------------------------------------------------------------------------
def _one_repeat_draws(rep_i, sub_values, preds_index, y_obs, pair_ids, cluster_vars,
                      singleton_vars, predictor_to_label, cohort, family, top_k,
                      num_warmup, num_samples, num_chains, seed_offset,
                      conditional, max_draws, draw_seed, n_repeats, verbose):
    """Returns (rep_i, topk names, {unit: draws}, n_rows, diagnostics, seconds).

    Fitting is identical to stability_selection._one_repeat; the only
    difference is what is carried back out of the fit.
    """
    import jax
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst

    t0 = time.time()
    module = hst if cohort == "BATSS" else hs
    seed_i = seed_offset + rep_i

    preds_i = _build_decluster_predictor_list(cluster_vars, singleton_vars, seed=seed_i)
    cols = [preds_index[v] for v in preds_i]

    Xi = sub_values[:, cols]
    Xsi, ok = _standardize(Xi)
    if not ok or not np.isfinite(Xi).all():
        return rep_i, None, (None, None), len(y_obs), {}, time.time() - t0

    ys = y_obs
    if family == "normal":
        ys = (ys - ys.mean()) / ys.std()

    if cohort == "BATSS":
        mcmc, _ = module.fit_twin_horseshoe(
            Xsi, ys, pair_ids, family=family, seed=seed_i,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=False)
    else:
        mcmc = module.fit_horseshoe(
            Xsi, ys, family=family, seed=seed_i,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, progress_bar=False)

    # convergence diagnostics, read off the same fitted object -- nothing
    # about the fit is altered. numpyro collects 'diverging' by default.
    diag = {"repeat": rep_i, "seed": seed_i}
    try:
        div = np.asarray(mcmc.get_extra_fields()["diverging"])
        diag["n_divergences"] = int(div.sum())
        diag["n_draws_total"] = int(div.size)
    except Exception:
        diag["n_divergences"] = np.nan
        diag["n_draws_total"] = np.nan
    if num_chains > 1:
        # same call check_rhat() makes, so max_rhat / min_ess_bulk are directly
        # comparable with the numbers the original loop printed
        import arviz as az
        _s = az.summary(az.from_numpyro(mcmc), round_to=None)
        diag["max_rhat"] = float(_s["r_hat"].max())
        diag["n_params_rhat_gt_1.05"] = int((_s["r_hat"] > 1.05).sum())
        diag["min_ess_bulk"] = float(_s["ess_bulk"].min())
    else:
        diag["max_rhat"] = np.nan      # r_hat needs >= 2 chains
        diag["n_params_rhat_gt_1.05"] = np.nan
        diag["min_ess_bulk"] = np.nan

    n_used = len(y_obs)
    beta = np.asarray(mcmc.get_samples()["beta"])          # (draws, p)
    bf = module.inclusion_bayes_factor(mcmc, n_used)
    # always needed: the escape mask drives escape_rate/BF10 regardless of
    # whether the pooled beta draws are restricted to it
    included = module._kappa(mcmc, n_used) < 0.5

    # top-K exactly as the main loop ranks it: round to 3dp, then sort
    order = np.argsort(-np.round(bf, 3), kind="stable")[:top_k]
    names_all = [predictor_to_label.get(v, v) for v in preds_i]
    topk = [names_all[i] for i in order]

    rng = np.random.default_rng(draw_seed + rep_i)
    draws, incl = {}, {}
    diag["draws_per_repeat"] = int(beta.shape[0])
    for j, unit in enumerate(names_all):
        # escape count is always the kappa<0.5 count, never the pooled-draw
        # count -- otherwise conditional=False makes escape_rate 1.0
        incl[unit] = int(included[:, j].sum())
        b = beta[included[:, j], j] if conditional else beta[:, j]
        if b.size == 0:
            continue
        if b.size > max_draws:                     # cap memory, keep it unbiased
            b = rng.choice(b, size=max_draws, replace=False)
        draws[unit] = b.astype(np.float32)

    del mcmc, beta, Xi, Xsi
    gc.collect()
    jax.clear_caches()

    elapsed = time.time() - t0
    if verbose:
        rh = diag.get("max_rhat")
        eb = diag.get("min_ess_bulk")
        nb = diag.get("n_params_rhat_gt_1.05")
        rh_s = "n/a (1 chain)" if rh is None or (isinstance(rh, float) and np.isnan(rh)) else f"{rh:.3f}"
        eb_s = "n/a" if eb is None or (isinstance(eb, float) and np.isnan(eb)) else f"{eb:.0f}"
        nb_s = "n/a" if nb is None or (isinstance(nb, float) and np.isnan(nb)) else f"{int(nb)}"
        print(f"run {rep_i+1}/{n_repeats} (seed={seed_i})... "
              f"max r_hat = {rh_s} | params with r_hat>1.05: {nb_s} | "
              f"min bulk-ESS = {eb_s} | divergences: {diag.get('n_divergences')} "
              f"| done ({elapsed:.1f} sec)", flush=True)
    return rep_i, topk, (draws, incl), n_used, diag, elapsed


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_pooled_summary(sub, y, preds_all, cluster_vars, singleton_vars,
                       predictor_to_label, *,
                       cohort="EASE", family="normal", pair_ids=None,
                       top_k=10, n_repeats=100,
                       num_warmup=2000, num_samples=5000, num_chains=6,
                       seed_offset=100, conditional=True, max_draws=500, level=0.95,
                       draw_seed=999, n_jobs=1, n_print=20, sort_by="pd",
                       out_prefix="pooled", out_dir=".", verbose=True):
    """
    Arguments match stability_selection.run_stability_selection(); the extra
    ones are:

    conditional : pool only draws with kappa<0.5 (default, matches the rest
                  of the pipeline) or every draw
    max_draws   : cap on draws kept per unit per repeat, subsampled at random
                  when exceeded. Bounds memory; 500 x 122 units x 100 repeats
                  is roughly 25 MB as float32.
    level       : credible interval mass (0.95 -> 2.5th/97.5th percentiles)
    n_print     : how many rows to show in the notebook (default 20); the CSV
                  always holds all units regardless
    sort_by     : "pd" (default), "BF10", or "prop_selected"

    Reported columns are the conditional-on-inclusion summaries (median_beta,
    ci_lower, ci_upper, pd) plus the inclusion Bayes factor BF10 and the
    selection frequency. BF10 = posterior odds / prior odds of kappa<0.5, the
    prior odds taken from ssvs_horseshoe.prior_inclusion_baseline(n, p, p/11).
    The pooled escape_rate that BF10 is built from is written to the CSV but
    kept out of the printed table.
    pair_ids    : required for cohort="BATSS"; leave as None for EASE

    Returns (summary_df, meta). summary_df is sorted by pd, descending.
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

    if verbose:
        print(f"cohort={cohort}  n={n} (full sample, no subsampling)  "
              f"top_k={top_k}  repeats={n_repeats}")
        print(f"pooling {'conditional (kappa<0.5)' if conditional else 'all'} draws, "
              f"<= {max_draws}/unit/repeat  |  MCMC {num_warmup}+{num_samples} "
              f"x {num_chains} chains  n_jobs={n_jobs}")
        ncpu = os.cpu_count() or 1
        if n_jobs * num_chains > ncpu:
            print(f"  NOTE: n_jobs({n_jobs}) x num_chains({num_chains}) > {ncpu} cores")

    common = dict(sub_values=sub_values, preds_index=preds_index, y_obs=y,
                  pair_ids=pair_ids, cluster_vars=cluster_vars,
                  singleton_vars=list(singleton_vars),
                  predictor_to_label=predictor_to_label, cohort=cohort,
                  family=family, top_k=top_k,
                  num_warmup=num_warmup, num_samples=num_samples,
                  num_chains=num_chains, seed_offset=seed_offset,
                  conditional=conditional,
                  max_draws=max_draws, draw_seed=draw_seed,
                  n_repeats=n_repeats, verbose=verbose)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_repeat_draws)(i, **common) for i in range(n_repeats)
    )

    pooled = defaultdict(list)      # unit -> list of draw arrays, one per repeat
    hits = defaultdict(int)         # unit -> times in top-K
    runs_present = defaultdict(int)  # unit -> repeats contributing any draw
    per_run_med = defaultdict(list)  # unit -> per-repeat median (run-to-run spread)
    diags = []
    n_used = 0
    incl_total = defaultdict(int)     # escape counts, used only for BF10
    draws_per_repeat = None
    for _, topk, (draws, incl), nrows, diag, dt in results:
        if diag:
            diags.append(diag)
            draws_per_repeat = diag.get("draws_per_repeat", draws_per_repeat)
        if topk is None:
            continue
        for u in set(topk):
            hits[u] += 1
        for u, c in (incl or {}).items():
            incl_total[u] += c
        for u, b in draws.items():
            pooled[u].append(b)
            runs_present[u] += 1
            per_run_med[u].append(float(np.median(b)))
        n_used += 1

    if n_used == 0:
        raise RuntimeError("no usable repeats")

    # prior odds of kappa<0.5 for the BF10 denominator; p is the per-fit
    # predictor count (one representative per cluster + the singletons)
    p_fit = len(cluster_vars) + len(singleton_vars)
    prior_p = float(prior_inclusion_baseline(n, p_fit, p_fit / 11.0))
    prior_odds = prior_p / (1.0 - prior_p)

    a = (1 - level) / 2 * 100
    rows_out = []
    for u, chunks in pooled.items():
        b = np.concatenate(chunks)
        p_pos = float((b > 0).mean())
        lo, hi = np.percentile(b, [a, 100 - a])
        med = float(np.median(b))
        rate = _escape_rate(incl_total[u], n_used, draws_per_repeat)
        rows_out.append({
            "predictor": u,
            "n_selected": hits.get(u, 0),
            "prop_selected": 100.0 * hits.get(u, 0) / n_used,
            "median_beta": med,
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "pd": max(p_pos, 1.0 - p_pos),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "BF10": _bf10(rate, prior_odds),
            "escape_rate": rate,          # CSV only; not in the printed table
            "n_runs_present": runs_present[u],
            "n_draws_pooled": int(b.size),
            "beta_run_sd": (float(np.std(per_run_med[u], ddof=1))
                            if len(per_run_med[u]) > 1 else 0.0),
        })

    _SORTS = {"pd": (["pd", "prop_selected"], [False, False]),
              "BF10": (["BF10", "pd"], [False, False]),
              "prop_selected": (["prop_selected", "pd"], [False, False])}
    if sort_by not in _SORTS:
        raise ValueError(f"sort_by must be one of {list(_SORTS)}")
    _by, _asc = _SORTS[sort_by]
    summary = (pd.DataFrame(rows_out)
               .sort_values(_by, ascending=_asc)
               .reset_index(drop=True))
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(os.path.join(out_dir, f"{out_prefix}_pooled_summary.csv"), index=False)

    diag_df = pd.DataFrame(diags).sort_values("repeat").reset_index(drop=True)
    diag_df.to_csv(os.path.join(out_dir, f"{out_prefix}_pooled_diagnostics.csv"), index=False)

    meta = dict(cohort=cohort, family=family, n=int(n),
                top_k=top_k, n_repeats=n_repeats, n_repeats_used=n_used,
                conditional=conditional, max_draws=max_draws, level=level,
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, seed_offset=seed_offset,
                n_units=int(len(summary)),
                p_fit=int(p_fit), p0=round(p_fit / 11.0, 3),
                prior_P_kappa_lt_half=float(prior_p),   # full precision: BF10 must
                                                       # be recomputable exactly
                max_rhat=(float(np.nanmax(diag_df["max_rhat"]))
                          if len(diag_df) and diag_df["max_rhat"].notna().any() else None),
                total_divergences=(int(diag_df["n_divergences"].sum())
                                   if len(diag_df) and diag_df["n_divergences"].notna().any() else None),
                repeats_with_divergences=(int((diag_df["n_divergences"] > 0).sum())
                                          if len(diag_df) and diag_df["n_divergences"].notna().any() else None),
                min_ess_bulk=(float(np.nanmin(diag_df["min_ess_bulk"]))
                         if len(diag_df) and diag_df["min_ess_bulk"].notna().any() else None),
                wall_minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(out_dir, f"{out_prefix}_pooled_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        cols = ["rank", "predictor", "n_selected", "prop_selected",
                "median_beta", "ci_lower", "ci_upper", "pd", "BF10",
                "n_draws_pooled"]
        head = summary[cols].head(n_print).round(3)
        print(f"\ntop {n_print} by {sort_by}:")
        try:                       # renders as a table in Jupyter
            from IPython.display import display
            display(head)
        except Exception:
            print(head.to_string(index=False))
        print(f"\nunits with any pooled draws: {len(summary)}   "
              f"wall: {meta['wall_minutes']:.1f} min")
        print(f"CONVERGENCE  max r_hat: {meta['max_rhat']}   min ESS-bulk: {meta['min_ess_bulk']}   "
              f"total divergences: {meta['total_divergences']} "
              f"(in {meta['repeats_with_divergences']} of {n_used} repeats)")
        print(f"  per-repeat detail -> {out_prefix}_pooled_diagnostics.csv")

    return summary, meta
