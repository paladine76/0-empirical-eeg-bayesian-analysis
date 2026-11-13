"""
One fixed 50/50 split. Selection on the train half, a single inference fit on
the held-out half.

STAGE A -- selection (train half only)
    n_repeats declustered horseshoe refits on the SAME train rows. Each repeat
    draws one representative per cluster plus all singletons (seed_i =
    seed_offset + i, driving both the declustering draw and the sampler, as in
    run_stability_loop). Escape counts (kappa_j < 0.5) are summed over all
    repeats, and the pooled inclusion Bayes factor is

        escape_rate = sum(escape draws) / (n_repeats * draws_per_repeat)
        BF10        = [escape_rate/(1-escape_rate)] / [prior/(1-prior)]

    with prior = ssvs_horseshoe.prior_inclusion_baseline(n_train, p, p/11).
    This is the identical construction pooled_summary.py uses -- one BF per
    unit from the pooled count, not an average of per-repeat BFs. Units are
    ranked by BF10 and the top_k are carried forward.

STAGE B -- inference (test half, ONE fit)
    The top_k units are mapped to concrete variables. A singleton is itself; a
    cluster takes the member with the highest mean train-half BF10 across the
    repeats in which it was the drawn representative -- decided from train
    data only, so nothing leaks. Those variables are fitted jointly, once, on
    the test half with fit_bayesian_lm (EASE) or fit_bayesian_lmm (BATSS):
    Normal(0, prior_scale^2) on every coefficient, no shrinkage, nothing
    selected away. Output is beta / pd / credible interval / Savage-Dickey
    BF10, sorted by pd.

    The test half never influenced which predictors were chosen, so pd from it
    needs no post-selection caveat.

SPLITTING
    BATSS splits whole twin pairs, so both members land in the same half and
    the halves stay independent. bernoulli outcomes are stratified on the
    outcome (on pair-contains-a-case for BATSS) so both halves carry cases.
    Each half is standardized on its own moments -- test never sees train's.

COVARIATES  (optional, OFF by default)
    covariates=None runs unadjusted. Passing e.g. ("age5mo", "age3y") makes them
    enter BOTH stages as Z: forced in and NOT under the horseshoe prior in stage
    A (gamma ~ Normal(0,5)), and as ordinary unshrunk coefficients in stage B --
    so selection and held-out inference are adjusted together. sex, status
    (EASE) and Zygosity (BATSS) are deliberately not offered as defaults.

    Continuous covariates are standardized within each half, identical to what
    the notebook's build_Z() does for them. Rows missing any covariate are
    dropped BEFORE the split, so n can fall; the count is printed and recorded
    in the meta. age3y in particular is missing wherever there is no 3-year
    follow-up, which costs about a third of the sample for the 5-month and
    looking-time outcomes -- use ("age5mo",) there.
"""

import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.bayesian_shrinkage_analyses.ssvs_horseshoe import prior_inclusion_baseline
from src.bayesian_shrinkage_analyses.split_sample_validation import make_split
from src.bayesian_shrinkage_analyses.stability_selection import (
    _build_decluster_predictor_list, _standardize)


def _one_selection_repeat(rep_i, Xtr_all, preds_index, ytr, cluster_vars, singleton_vars,
                          predictor_to_label, family, pair_ids_tr, Ztr, top_k, num_warmup,
                          num_samples, num_chains, seed_offset, n_repeats, verbose,
                          predictors=None, algorithm="horseshoe", algo_kwargs=None):
    """One horseshoe fit on the train half. Returns per-unit escape counts,
    per-variable BF10, this repeat's top-K labels, and diagnostics. Module level
    so joblib can pickle it."""
    import gc
    import jax
    import numpy as np
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst

    t0 = time.time()
    seed_i = seed_offset + rep_i
    if predictors is not None:      # flat mode: fixed predictor set, no declustering draw
        preds_i = list(predictors)
    else:
        preds_i = _build_decluster_predictor_list(cluster_vars, singleton_vars, seed=seed_i)
    cols = [preds_index[v] for v in preds_i]
    Xi, ok = _standardize(Xtr_all[:, cols])
    if not ok:
        raise ValueError(f"repeat {rep_i}: zero-variance or non-finite predictor "
                         f"column in the train half")
    n_tr = len(ytr)

    try:                 # inside the package (how the notebook imports it)
        from src.bayesian_shrinkage_analyses.shrinkage_backends import get_backend
    except ImportError:  # loose script next to shrinkage_backends.py
        from shrinkage_backends import get_backend
    fit = get_backend(algorithm)
    res = fit(Xi, ytr, pair_ids=pair_ids_tr, Z=Ztr, family=family, seed=seed_i,
              num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
              **(algo_kwargs or {}))

    escaped = res["included"].sum(0).astype(int)      # per predictor, this repeat
    n_draws = res["n_draws"]
    bf = res["score"]
    prior_incl_prob = res["prior_incl_prob"]

    diag = {"repeat": rep_i, "seed": seed_i, "n_draws": n_draws,
            "algorithm": algorithm, **res["diag"]}

    labels_i = ([v for v in preds_i] if predictor_to_label is None
                else [predictor_to_label.get(v, v) for v in preds_i])
    # this repeat's top-K, ranked exactly as run_stability_loop does it:
    # round BF10 to 3dp, then stable-sort
    topk_i = [labels_i[j] for j in np.argsort(-np.round(bf, 3), kind="stable")[:top_k]]
    del res, Xi
    gc.collect()
    jax.clear_caches()

    diag["sec"] = round(time.time() - t0, 1)
    if verbose:
        rh = diag.get("max_rhat", np.nan)
        rh_s = ("n/a (Gibbs)" if not np.isfinite(rh) and algorithm == "slabstick"
                else "n/a (1 chain)" if not np.isfinite(rh) else f"{rh:.3f}")
        extra = (f"mean #incl/draw = {diag['mean_n_included_per_draw']:.2f}"
                 if "mean_n_included_per_draw" in diag
                 else f"divergences: {diag.get('n_divergences')}")
        print(f"select {rep_i+1}/{n_repeats} (seed={seed_i})... max r_hat = {rh_s} | "
              f"{extra} | done ({diag['sec']} sec)", flush=True)

    return preds_i, labels_i, escaped, bf, n_draws, topk_i, diag, prior_incl_prob


def run_split_selection_inference(
        sub, y, preds_all=None, cluster_vars=None, singleton_vars=None,
        predictor_to_label=None, *, predictors=None,
        algorithm="horseshoe", algo_kwargs=None,
        cohort="EASE", family="normal", pair_ids=None,
        covariates=None,
        train_frac=0.5, split_seed=0,
        top_k=10, n_repeats=25,
        num_warmup=2000, num_samples=5000, num_chains=6, seed_offset=100,
        prior_scale=2.5, lm_warmup=2000, lm_samples=3000, lm_chains=4, lm_seed=0,
        level=0.95, n_jobs=1, out_prefix="splitfit", out_dir=".", verbose=True):
    """
    sub                : DataFrame already reduced to complete cases on the outcome
    y                  : outcome array aligned to sub (raw; standardized per half
                         when family="normal")
    TWO MODES, pick one:

    CLUSTERED (the EEG phenotype analysis)
      preds_all          : every EEG variable name (the full pre-declustering list)
      cluster_vars       : {cluster_id: [member variables]}
      singleton_vars     : [variables in no cluster]
      predictor_to_label : {variable: cluster label}; singletons may be absent
      Each repeat draws one representative per cluster, so the repeats average
      over the declustering choice.

    FLAT (no clustering, e.g. the M1r Far/Near set)
      predictors         : the predictor column names, used as-is
      cluster_vars / singleton_vars / predictor_to_label / preds_all : leave None
      Every repeat fits the SAME predictor set, so repeats differ only in the
      sampler seed. One repeat is already 30,000 draws, which pins the escape
      rate to about +/-0.002, so n_repeats=1 is normally enough here -- the
      reason for 25 was averaging over declustering draws, which no longer
      applies. Higher n_repeats still works, it just buys almost nothing.
    cohort             : "EASE" or "BATSS" (BATSS splits whole pairs)
    pair_ids           : required for cohort="BATSS", None for EASE
    algorithm          : "horseshoe" (default, NUTS, kappa<0.5 inclusion,
                         supports covariates) or "slabstick" (discrete
                         spike-and-slab Gibbs, exact-zero z==1 inclusion, does
                         NOT support covariates). See shrinkage_backends.py.
                         BF10 divides by each backend's own prior inclusion
                         probability, so it is comparable across the two even
                         though the raw inclusion rates are not.
    algo_kwargs        : dict passed to the backend, e.g.
                         {"n_iter": 5000, "burn": 1000, "a0": 1.0, "b0": 10.0}
                         for slabstick. MCMC args (num_warmup/num_samples/
                         num_chains) are forwarded to the horseshoe backend and
                         ignored by slabstick.
    covariates         : forced-in, unshrunk covariates entering BOTH stages.
                         Default None = unadjusted. Pass ("age5mo", "age3y") for
                         the adjusted analysis, or ("age5mo",) for 5-month
                         outcomes where age3y missingness would cost a third of
                         the sample. Rows missing any covariate are dropped
                         before the split.
    train_frac         : fraction of the sample used for selection (0.5)
    top_k              : how many units are carried into the inference fit
    n_repeats          : horseshoe refits on the train half
    prior_scale        : SD of the Normal prior in the inference fit (no shrinkage)

    Returns (inference_df, selection_df, meta). inference_df is sorted by pd,
    descending.
    """
    from src.bayesian_shrinkage_analyses import baseline_bayesian_lm as blm
    from src.bayesian_shrinkage_analyses import baseline_bayesian_lmm as blmm

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    y = np.asarray(y, dtype=float)
    pair_ids = None if pair_ids is None else np.asarray(pair_ids)
    if cohort == "BATSS" and pair_ids is None:
        raise ValueError("cohort='BATSS' requires pair_ids")

    # ---- complete cases on the covariates, before the split ----------------
    covariates = list(covariates) if covariates else []
    n_before = len(y)
    if covariates:
        missing = [c for c in covariates if c not in sub.columns]
        if missing:
            raise ValueError(f"covariates not in sub: {missing}")
        keep = sub[covariates].notna().all(axis=1).values
        if not keep.all():
            sub = sub.loc[keep].reset_index(drop=True)
            y = y[keep]
            if pair_ids is not None:
                pair_ids = pair_ids[keep]
    n_dropped = n_before - len(y)

    # ---- the one split -----------------------------------------------------
    rng = np.random.default_rng(split_seed)
    train_rows, test_rows = make_split(y, pair_ids, cohort, family, train_frac, rng)
    # ---- mode check -------------------------------------------------------
    flat = predictors is not None
    if flat:
        if cluster_vars is not None or singleton_vars is not None:
            raise ValueError("pass either predictors= (flat) or "
                             "cluster_vars/singleton_vars (clustered), not both")
        predictors = list(predictors)
        missing = [v for v in predictors if v not in sub.columns]
        if missing:
            raise ValueError(f"predictors not in sub: {missing}")
        if preds_all is None:
            preds_all = predictors
    else:
        if cluster_vars is None or singleton_vars is None:
            raise ValueError("clustered mode needs cluster_vars and singleton_vars; "
                             "for an unclustered predictor set pass predictors=")
        if preds_all is None:
            raise ValueError("clustered mode needs preds_all")

    X_all = sub[preds_all].values.astype(float)
    preds_index = {v: i for i, v in enumerate(preds_all)}
    Xtr_all = X_all[train_rows]

    ytr = y[train_rows]
    yte = y[test_rows]
    if family == "normal":
        ytr = (ytr - ytr.mean()) / ytr.std()
        yte = (yte - yte.mean()) / yte.std()
    pair_tr = None if pair_ids is None else pair_ids[train_rows]
    pair_te = None if pair_ids is None else pair_ids[test_rows]
    n_tr, n_te = len(ytr), len(yte)
    p_fit = len(predictors) if flat else len(cluster_vars) + len(singleton_vars)
    if top_k >= p_fit:
        raise ValueError(f"top_k={top_k} must be smaller than p_fit={p_fit}")

    # ---- covariates: standardized within each half, never pooled -----------
    Ztr = Zte = None
    if covariates:
        Zraw = sub[covariates].values.astype(float)
        Ztr, ok_ztr = _standardize(Zraw[train_rows])
        Zte, ok_zte = _standardize(Zraw[test_rows])
        if not (ok_ztr and ok_zte):
            raise ValueError("zero-variance or non-finite covariate in one of the halves")

    if verbose:
        extra = ""
        if pair_ids is not None:
            extra = (f"  pairs: {pd.Series(pair_tr).nunique()} train / "
                     f"{pd.Series(pair_te).nunique()} test")
        cov_s = ", ".join(covariates) if covariates else "none (unadjusted)"
        drop_s = f", {n_dropped} row(s) dropped for missing covariates" if n_dropped else ""
        print(f"cohort={cohort}  n={len(y)} -> train {n_tr} / test {n_te}"
              f"  (train_frac={train_frac}, split_seed={split_seed}){extra}")
        print(f"covariates (forced in, unshrunk, both stages): {cov_s}{drop_s}")
        mode_s = ("fixed predictor set (no declustering)" if flat
                  else "declustered")
        print(f"STAGE A: {n_repeats} {mode_s} {algorithm} refits on the train half, "
              f"p={p_fit} per fit  |  MCMC {num_warmup}+{num_samples} x {num_chains} chains  "
              f"n_jobs={n_jobs}", flush=True)

    # ---- STAGE A: selection on the train half ------------------------------
    jobs = (delayed(_one_selection_repeat)(
                i, Xtr_all, preds_index, ytr, cluster_vars, singleton_vars,
                predictor_to_label, family, pair_tr, Ztr, top_k, num_warmup,
                num_samples, num_chains, seed_offset, n_repeats, verbose,
                predictors if flat else None, algorithm, algo_kwargs)
            for i in range(n_repeats))
    results = (Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(jobs)
               if n_jobs != 1 else [j[0](*j[1], **j[2]) for j in jobs])

    escape_count = defaultdict(int)        # unit label -> escaping draws, all repeats
    top10_count = defaultdict(int)         # unit label -> repeats where it made top-K
    var_bf_sum = defaultdict(float)        # concrete variable -> sum of per-repeat BF10
    var_bf_n = defaultdict(int)
    var_label = {}
    draws_total = 0
    n_used = 0
    diags = []
    prior_from_backend = None
    for preds_i, labels_i, escaped, bf, n_draws, topk_i, diag, prior_ip in results:
        prior_from_backend = prior_ip
        diags.append(diag)
        draws_total += n_draws
        n_used += 1
        for lab in set(topk_i):
            top10_count[lab] += 1
        for j, (v, lab) in enumerate(zip(preds_i, labels_i)):
            escape_count[lab] += int(escaped[j])
            var_bf_sum[v] += float(bf[j])
            var_bf_n[v] += 1
            var_label[v] = lab

    # the backend supplies its own prior inclusion probability: ~0.134 for the
    # horseshoe's kappa<0.5 event, a0/(a0+b0) for slabstick's z==1
    prior_p = float(prior_from_backend)
    prior_odds = prior_p / (1.0 - prior_p)

    sel_rows = []
    for lab, c in escape_count.items():
        rate = c / draws_total
        sel_rows.append({"predictor": lab,
                         "escape_rate": rate,
                         "BF10_train": (rate / max(1.0 - rate, 1e-12)) / prior_odds,
                         "top10_count": top10_count.get(lab, 0),
                         "top10_frequency": top10_count.get(lab, 0) / n_used,
                         "escape_draws": int(c)})
    selection = (pd.DataFrame(sel_rows)
                 .sort_values("BF10_train", ascending=False)
                 .reset_index(drop=True))
    selection.insert(0, "rank", np.arange(1, len(selection) + 1))

    selected_labels = list(selection.predictor.head(top_k))

    # map each selected unit to one concrete variable, using train BF10 only
    chosen = {}
    for lab in selected_labels:
        cands = [v for v, l in var_label.items() if l == lab]
        chosen[lab] = max(cands, key=lambda v: var_bf_sum[v] / max(var_bf_n[v], 1))
    sel_vars = [chosen[lab] for lab in selected_labels]

    if verbose:
        print(f"\ntop {top_k} by pooled train BF10 "
              f"(prior P(kappa<0.5) = {prior_p:.4f}, {draws_total} draws pooled):")
        for r, lab in enumerate(selected_labels, 1):
            row = selection.loc[selection.predictor == lab].iloc[0]
            v = chosen[lab]
            via = "" if v == lab else f"  [via {v}]"
            print(f"  {r:2d}. {lab:<32s} BF10={row.BF10_train:7.3f} "
                  f"top10 {int(row.top10_count)}/{n_used}{via}")
        print(f"\nSTAGE B: one {'LMM' if pair_ids is not None else 'LM'} fit on the "
              f"held-out half (n={n_te}, p={len(sel_vars)}, prior_scale={prior_scale})",
              flush=True)

    # ---- STAGE B: one inference fit on the held-out half -------------------
    Xte, ok_te = _standardize(sub.iloc[test_rows][sel_vars].values.astype(float))
    if not ok_te:
        raise ValueError("zero-variance or non-finite predictor column in the test half")
    if pair_ids is not None:
        mcmc_inf, _ = blmm.fit_bayesian_lmm(Xte, yte, pair_te, Z=Zte, family=family,
                                            prior_scale=prior_scale, seed=lm_seed,
                                            num_warmup=lm_warmup, num_samples=lm_samples,
                                            num_chains=lm_chains, progress_bar=False)
        beta_mean, pd_, ci_lo, ci_hi = blmm.posterior_summary(mcmc_inf, level=level)
        bf_sd = blmm.savage_dickey_bf10(mcmc_inf, prior_scale=prior_scale)
    else:
        mcmc_inf = blm.fit_bayesian_lm(Xte, yte, Z=Zte, family=family,
                                       prior_scale=prior_scale,
                                       seed=lm_seed, num_warmup=lm_warmup,
                                       num_samples=lm_samples, num_chains=lm_chains,
                                       progress_bar=False)
        beta_mean, pd_, ci_lo, ci_hi = blm.posterior_summary(mcmc_inf, level=level)
        bf_sd = blm.savage_dickey_bf10(mcmc_inf, prior_scale=prior_scale)

    # covariate coefficients from the same held-out fit (gamma, unshrunk)
    cov_tbl = None
    if covariates:
        gd = np.asarray(mcmc_inf.get_samples()["gamma"])
        a_ = (1 - level) / 2 * 100
        ppos = (gd > 0).mean(0)
        cov_tbl = pd.DataFrame({
            "covariate": covariates,
            "gamma_median": np.median(gd, axis=0),
            "ci_lower": np.percentile(gd, a_, axis=0),
            "ci_upper": np.percentile(gd, 100 - a_, axis=0),
            "pd": np.maximum(ppos, 1 - ppos),
        }).round(4)

    beta_draws = np.asarray(mcmc_inf.get_samples()["beta"])
    inf_diag = {}
    try:
        inf_diag["n_divergences"] = int(np.asarray(
            mcmc_inf.get_extra_fields()["diverging"]).sum())
    except Exception:
        inf_diag["n_divergences"] = np.nan
    if lm_chains > 1:
        import arviz as az
        _s = az.summary(az.from_numpyro(mcmc_inf), round_to=None)
        inf_diag["max_rhat"] = float(_s["r_hat"].max())
        inf_diag["min_ess_bulk"] = float(_s["ess_bulk"].min())

    inference = pd.DataFrame({
        "predictor": selected_labels,
        "variable_fitted": sel_vars,
        "rank_train_BF10": [int(selection.loc[selection.predictor == l, "rank"].iloc[0])
                            for l in selected_labels],
        "BF10_train": [float(selection.loc[selection.predictor == l, "BF10_train"].iloc[0])
                       for l in selected_labels],
        "top10_count": [int(top10_count.get(l, 0)) for l in selected_labels],
        "top10_frequency": [top10_count.get(l, 0) / n_used for l in selected_labels],
        "beta_mean": beta_mean,
        "beta_median": np.median(beta_draws, axis=0),
        "ci_lower": ci_lo, "ci_upper": ci_hi,
        "pd": pd_,
        "ci_excludes_zero": (ci_lo > 0) | (ci_hi < 0),
        "BF10_test_savage_dickey": bf_sd,
    }).sort_values("pd", ascending=False).reset_index(drop=True)
    inference.insert(0, "rank", np.arange(1, len(inference) + 1))

    for c in ["escape_rate", "BF10_train", "top10_frequency"]:
        selection[c] = selection[c].round(4)
    for c in ["BF10_train", "top10_frequency", "beta_mean", "beta_median",
              "ci_lower", "ci_upper", "pd", "BF10_test_savage_dickey"]:
        inference[c] = inference[c].round(4)

    diag_df = pd.DataFrame(diags).sort_values("repeat").reset_index(drop=True)
    meta = dict(cohort=cohort, family=family, n_total=int(len(y)),
                covariates=covariates, n_dropped_missing_covariates=int(n_dropped),
                n_train=int(n_tr), n_test=int(n_te), train_frac=train_frac,
                split_seed=split_seed, top_k=top_k, n_repeats=n_repeats,
                mode=("flat" if flat else "clustered"),
                algorithm=algorithm, algo_kwargs=(algo_kwargs or {}),
                p_fit=int(p_fit), p0=round(p_fit / 11.0, 3),
                prior_P_kappa_lt_half=float(prior_p),   # full precision: BF10 must
                                                       # be recomputable exactly
                draws_pooled_train=int(draws_total),
                num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, seed_offset=seed_offset,
                prior_scale=prior_scale, lm_warmup=lm_warmup,
                lm_samples=lm_samples, lm_chains=lm_chains, lm_seed=lm_seed,
                level=level,
                select_max_rhat=(float(np.nanmax(diag_df["max_rhat"]))
                                 if diag_df["max_rhat"].notna().any() else None),
                select_total_divergences=(int(diag_df["n_divergences"].sum())
                                          if diag_df["n_divergences"].notna().any() else None),
                select_min_ess_bulk=(float(np.nanmin(diag_df["min_ess_bulk"]))
                                     if diag_df["min_ess_bulk"].notna().any() else None),
                inference_diag=inf_diag,
                covariate_effects=(cov_tbl.to_dict("records") if cov_tbl is not None else None),
                wall_minutes=round((time.time() - t0) / 60, 2))

    if cov_tbl is not None:
        cov_tbl.to_csv(os.path.join(out_dir, f"{out_prefix}_covariates.csv"), index=False)
    selection.to_csv(os.path.join(out_dir, f"{out_prefix}_selection.csv"), index=False)
    inference.to_csv(os.path.join(out_dir, f"{out_prefix}_inference.csv"), index=False)
    diag_df.to_csv(os.path.join(out_dir, f"{out_prefix}_select_diagnostics.csv"), index=False)
    with open(os.path.join(out_dir, f"{out_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        cols = ["rank", "predictor", "variable_fitted", "rank_train_BF10",
                "top10_count", "beta_median", "ci_lower", "ci_upper", "pd",
                "BF10_test_savage_dickey"]
        print(f"\nheld-out inference, ranked by pd:")
        try:
            from IPython.display import display
            display(inference[cols])
        except Exception:
            print(inference[cols].to_string(index=False))
        if cov_tbl is not None:
            print("\ncovariates in the held-out fit (unshrunk):")
            print(cov_tbl.to_string(index=False))
        print(f"\nSELECTION  max r_hat: {meta['select_max_rhat']}   "
              f"min bulk-ESS: {meta['select_min_ess_bulk']}   "
              f"divergences: {meta['select_total_divergences']}")
        print(f"INFERENCE  {inf_diag}")
        print(f"wall: {meta['wall_minutes']} min   "
              f"-> {out_prefix}_inference.csv / _selection.csv / _select_diagnostics.csv")

    return inference, selection, meta
