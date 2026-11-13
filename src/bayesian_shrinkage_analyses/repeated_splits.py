"""
Repeat split_selection_inference over several 50/50 splits and consolidate.

Each split is an independent run of the two-stage procedure (n_repeats
declustered horseshoe refits on the train half -> pooled train BF10 -> top_k ->
one LM/LMM fit on the held-out half). This wrapper runs it for
split_seed = 0 .. n_splits-1 and summarises across splits.

CONSOLIDATION
    Order statistics throughout -- no means, no SD/sqrt(n_splits). The splits
    share subjects, and pd and BF10 are bounded/ratio-scaled, so neither a
    normal approximation nor a product of Bayes factors applies.

    counts (denominator n_splits, comparable across predictors):
        n_selected        splits where it reached the train-half top_k
        n_pd_pass         splits where it was selected AND pd >= pd_threshold
        prop_pd_pass_of_selected   n_pd_pass / n_selected

    over ALL selected splits:
        min_pd            worst held-out pd -- the worst case, which only means
                          anything before the pd filter is applied
        median_BF10_train, median_top10_frequency
        median_BF10_test  deliberately NOT filtered by pd: it includes the
                          splits where the predictor did badly, so it is the
                          unfiltered control sitting beside the conditional pd.
                          A predictor with median_pd 0.97 but median_BF10_test
                          well under 1 is one whose apparent strength comes from
                          the splits that happened to favour it.

    over the PASSING splits (pd >= pd_threshold):
        median_beta, median_ci_lower, median_ci_upper, median_pd
        These are conditional-on-passing and therefore optimistic, in the same
        way the horseshoe's conditional-on-escape beta is. Report them as such.
        median_pd is bounded below by pd_threshold by definition of the subset.

COST
    n_splits * (n_repeats horseshoe fits + 1 inference fit). At n_splits=10 and
    n_repeats=25 that is 250 horseshoe fits. Price it with probe_only=True
    first, which runs one split at n_repeats=2 and reports the per-fit time.

    n_repeats=25 is the measured default. On the EASE/ADOS train half (n=44,
    p=122, production MCMC settings), two DISJOINT random blocks of 25 repeats
    returned the same top-10 in essentially every partition of 200 tried (mean
    overlap 10.0/10, worst 9/10; rank 1 agreed in 100%). R=10 gives 9.7/10 mean
    with 9/10 worst, so 25 buys the last two slots for 2.5x the cost. Going
    above 25 cannot improve on 10.0/10. What the repeats average over is the
    DECLUSTERING draw, not Monte Carlo error -- one repeat is already 30,000
    draws, so the escape rate is precise to about +/-0.002 at R=1.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from src.bayesian_shrinkage_analyses.split_selection_inference import run_split_selection_inference


def run_repeated_splits(sub, y, preds_all=None, cluster_vars=None, singleton_vars=None,
                        predictor_to_label=None, *,
                        n_splits=10, pd_threshold=0.90,
                        probe_only=False,
                        out_prefix="rsplit", out_dir=".", keep_per_split_files=True,
                        verbose=True, **split_kwargs):
    """
    Two modes, matching run_split_selection_inference: pass preds_all +
    cluster_vars + singleton_vars + predictor_to_label for the clustered EEG
    analysis, or leave all four as None and pass predictors=[...] through
    split_kwargs for an unclustered predictor set. In flat mode every repeat
    fits the same predictors, so n_repeats=1 is normally enough.

    split_kwargs are passed straight through to run_split_selection_inference
    (cohort, family, pair_ids, covariates, top_k, n_repeats, num_warmup,
    num_samples, num_chains, seed_offset, prior_scale, lm_*, level, n_jobs).
    split_seed is set by this wrapper and must not be passed.

    Returns (consolidated_df, per_split_df, meta).
    """
    if "split_seed" in split_kwargs:
        raise ValueError("split_seed is set by this wrapper; remove it")
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    if probe_only:
        kw = dict(split_kwargs)
        kw["n_repeats"] = 2
        if verbose:
            print("PROBE: one split at n_repeats=2, to price the full job\n")
        _, _, m = run_split_selection_inference(
            sub, y, preds_all, cluster_vars, singleton_vars, predictor_to_label,
            split_seed=0, out_prefix=f"{out_prefix}_probe", out_dir=out_dir,
            verbose=verbose, **kw)
        sec = pd.read_csv(os.path.join(out_dir, f"{out_prefix}_probe_select_diagnostics.csv")).sec
        per_fit = float(sec.mean())
        n_rep = split_kwargs.get("n_repeats", 25)
        total_h = per_fit * n_rep * n_splits / 3600
        print(f"\nper horseshoe fit: {per_fit:.0f} s at THESE mcmc settings")
        print(f"full job = {n_splits} splits x {n_rep} refits = {n_splits*n_rep} fits"
              f"  ->  ~{total_h:.1f} h serial, ~{total_h/max(split_kwargs.get('n_jobs',1),1):.1f} h "
              f"at n_jobs={split_kwargs.get('n_jobs',1)}")
        return None, None, m

    frames, metas = [], []
    for s in range(n_splits):
        if verbose:
            print(f"\n{'#'*78}\n# split {s+1}/{n_splits}  (split_seed={s})\n{'#'*78}", flush=True)
        pfx = f"{out_prefix}_s{s}"
        inf_s, sel_s, meta_s = run_split_selection_inference(
            sub, y, preds_all, cluster_vars, singleton_vars, predictor_to_label,
            split_seed=s, out_prefix=pfx, out_dir=out_dir, verbose=verbose,
            **split_kwargs)
        inf_s = inf_s.copy()
        inf_s.insert(0, "split_seed", s)
        frames.append(inf_s)
        metas.append(meta_s)
        if not keep_per_split_files:
            for suf in ("_inference.csv", "_selection.csv", "_select_diagnostics.csv",
                        "_covariates.csv", "_meta.json"):
                p = os.path.join(out_dir, pfx + suf)
                if os.path.exists(p):
                    os.remove(p)

    per_split = pd.concat(frames, ignore_index=True)
    per_split["pd_pass"] = per_split["pd"] >= pd_threshold

    rows = []
    for name, g in per_split.groupby("predictor", sort=False):
        gp = g[g.pd_pass]
        rows.append({
            "predictor": name,
            "n_selected": int(len(g)),
            "prop_selected": len(g) / n_splits,
            "n_pd_pass": int(len(gp)),
            "prop_pd_pass": len(gp) / n_splits,
            "prop_pd_pass_of_selected": (len(gp) / len(g)) if len(g) else np.nan,
            "min_pd": float(g["pd"].min()),                     # over ALL selected splits
            "median_BF10_train": float(g["BF10_train"].median()),
            "median_top10_frequency": float(g["top10_frequency"].median()),
            # unfiltered on purpose: the honest control beside the conditional
            # pd, since it includes the splits where the predictor did badly
            "median_BF10_test": float(g["BF10_test_savage_dickey"].median()),
            # ---- conditional on pd >= pd_threshold ----
            "median_beta": float(gp["beta_median"].median()) if len(gp) else np.nan,
            "median_ci_lower": float(gp["ci_lower"].median()) if len(gp) else np.nan,
            "median_ci_upper": float(gp["ci_upper"].median()) if len(gp) else np.nan,
            "median_pd": float(gp["pd"].median()) if len(gp) else np.nan,
            "beta_min_over_pass": float(gp["beta_median"].min()) if len(gp) else np.nan,
            "beta_max_over_pass": float(gp["beta_median"].max()) if len(gp) else np.nan,
            "variables_fitted": "|".join(sorted(set(g["variable_fitted"]))),
        })

    cons = (pd.DataFrame(rows)
            .sort_values(["n_pd_pass", "median_pd", "n_selected"],
                         ascending=[False, False, False])
            .reset_index(drop=True))
    cons.insert(0, "rank", np.arange(1, len(cons) + 1))
    for c in cons.columns:
        if cons[c].dtype.kind == "f":
            cons[c] = cons[c].round(4)

    m0 = metas[0]
    meta = dict(n_splits=n_splits, pd_threshold=pd_threshold,
                split_seeds=list(range(n_splits)),
                cohort=m0["cohort"], family=m0["family"],
                covariates=m0["covariates"], top_k=m0["top_k"],
                n_repeats=m0["n_repeats"], p_fit=m0["p_fit"], p0=m0["p0"],
                prior_P_kappa_lt_half=m0["prior_P_kappa_lt_half"],
                algorithm=m0.get("algorithm"), mode=m0.get("mode"),
                n_train=[m["n_train"] for m in metas],
                n_test=[m["n_test"] for m in metas],
                # slabstick is conjugate Gibbs: every split returns None for
                # r_hat and divergences, so max() must not be called on an
                # empty sequence
                select_max_rhat=(max(v for m in metas
                                     if (v := m.get("select_max_rhat")) is not None)
                                 if any(m.get("select_max_rhat") is not None
                                        for m in metas) else None),
                select_total_divergences=(int(sum(m.get("select_total_divergences") or 0
                                                  for m in metas))
                                          if any(m.get("select_total_divergences") is not None
                                                 for m in metas) else None),
                wall_minutes=round((time.time() - t0) / 60, 2))

    cons.to_csv(os.path.join(out_dir, f"{out_prefix}_consolidated.csv"), index=False)
    per_split.to_csv(os.path.join(out_dir, f"{out_prefix}_per_split.csv"), index=False)
    with open(os.path.join(out_dir, f"{out_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        cols = ["rank", "predictor", "n_selected", "n_pd_pass", "min_pd",
                "median_beta", "median_ci_lower", "median_ci_upper", "median_pd",
                "median_BF10_test", "median_BF10_train"]
        print(f"\n{'='*118}\nconsolidated over {n_splits} splits\n"
              f"  beta / CI / median_pd : over splits with pd >= {pd_threshold} "
              f"(conditional, optimistic)\n"
              f"  min_pd / BF10_test / BF10_train : over ALL splits where selected "
              f"(unfiltered control)\n{'='*118}")
        head = cons[cols].head(25)
        try:
            from IPython.display import display
            display(head)
        except Exception:
            print(head.to_string(index=False))
        print(f"\nunits appearing in >=1 split's top-{m0['top_k']}: {len(cons)}")
        print(f"selection diagnostics across all splits: max r_hat "
              f"{meta['select_max_rhat']}, total divergences "
              f"{meta['select_total_divergences']}")
        print(f"wall: {meta['wall_minutes']} min  -> {out_prefix}_consolidated.csv "
              f"/ _per_split.csv / _meta.json")

    return cons, per_split, meta
