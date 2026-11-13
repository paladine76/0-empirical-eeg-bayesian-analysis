"""
Analysis B summary: held-out directional support across the repeated splits,
for the frequency-selected unit-outcome pairs.

Follows inference_analysis_handoff.md s.6-10 and
clustered_predictors_directional_support.md s.3-9. No model is fitted here.
The input is the <prefix>_per_split.csv written by repeated_splits.py with the
2026-10-13 version of split_selection_inference.py (training direction and
heldout_support_for_training_direction columns). Older per-split files lack
those columns and are rejected.

REPORTING POPULATION
    The unit-outcome pairs of main_result_48_units.csv (61 pairs, 48 units).
    Pairs are not expanded to other outcomes. Split-half screening stayed over
    the full eligible set; the reporting file only chooses which rows are
    summarised.

OUTPUTS  (<out_prefix>_*.csv in out_dir)
    split_level     one row per reporting pair x split, including the splits
                    in which the unit was not selected (evaluation_status =
                    "not_selected", held-out quantities NaN, never zero).
    unit_summary    one row per reporting pair:
        n_selected / n_splits      training halves that put the unit in the top 10
        n_evaluated                selected splits with a defined training
                                   direction and a successful held-out fit
        n_undefined_direction, n_fit_failure
            These three columns, and members_evaluated, are written ONLY if
            some selected split was not evaluated. When every selected split
            was evaluated (the expected case) they equal n_selected, 0, 0 and
            members_selected, so they are dropped and a line saying so is
            printed. If any selected split was not evaluated, they are kept
            and a warning lists the affected pairs.
        support_median             primary ranking statistic (author decision
                                   2026-10-13: median rather than the handoff's
                                   mean, because the per-split support values
                                   are skewed); each evaluated split contributes
                                   once, whichever member represented a cluster
                                   in that split
        support_mean, support_min, support_max   descriptive
        n_support_ge_0.90          descriptive count over n_evaluated; not a cutoff
        n_evaluated_rhat_flag      evaluated splits whose diagnostic_status
                                   carries an R-hat > 1.05 flag (kept in the
                                   summaries, shown for inspection)
        n_train_positive / n_train_negative
                                   training directions over the EVALUATED splits
                                   (sum = n_evaluated, the same splits as the
                                   support summaries; cluster doc s.7.2 step 5).
                                   Singletons only; for a cluster these are NA,
                                   because raw signs of different members are
                                   not pooled (cluster doc s.7.2). Raw
                                   per-member signs are in train_signs_by_member
                                   and member_summary, also over evaluated splits.
        n_train_positive_aligned / n_train_negative_aligned
                                   clusters only, and only when an orientation
                                   file is given: member signs over the evaluated
                                   splits multiplied by the member's fixed
                                   orientation (+1/-1) relative to the reference
                                   member.
        members_selected, members_evaluated   representative usage, "member:count"
        not_evaluated_reason       "" | never_selected | selected_not_evaluated(...)
        rank_within_outcome        by support_median descending; exact ties broken
                                   by n_evaluated descending; still-equal pairs
                                   share the rank. Unevaluated pairs: NA, last.
    member_summary  clusters in the reporting set: one row per pair x member
                    that represented the cluster in at least one split
                    (n_evaluated dropped under the same rule as above).

INTERPRETATION
    support_median (and support_mean) summarise separate per-split posterior
    probabilities descriptively. It is not a pooled posterior probability, a significance
    level or a replication probability. For a cluster it describes support for
    the training direction of whichever member represented the cluster in each
    split; it does not refer to the Analysis A representative specifically.
"""

import json
import os

import numpy as np
import pandas as pd

# outcome labels used in main_result_48_units.csv -> outcome codes
REPORTING_OUTCOME_CODE = {
    "Social Looking": "ratio_face", "Development Level": "MSEL_composite5mo",
    "Grammar": "CDI_grammar3y", "Vocabulary": "CDI_vocab3y",
    "Negative Affect": "SHNEGAF_36", "Surge": "SHSURGE_36",
    "Effortful Control": "SHEFFCO_36", "Symptoms": "ADOS_sum_CS_man",
    "Diagnosis": "ASD",
}

REQUIRED_COLS = [
    "split_seed", "predictor", "variable_fitted", "rank_train_BF10",
    "beta_median", "ci_lower", "ci_upper", "pd",
    "training_beta_median", "training_direction",
    "heldout_support_for_training_direction", "evaluation_status",
    "training_n", "heldout_n", "diagnostic_status",
]

SPLIT_COLS = [  # copied from per_split into split_level, renamed where needed
    ("variable_fitted", "representative_id"),
    ("rank_train_BF10", "rank_train_BF10"),
    ("training_beta_median", "training_beta_median"),
    ("training_direction", "training_direction"),
    ("beta_median", "heldout_beta_median"),
    ("ci_lower", "heldout_ci_lower"),
    ("ci_upper", "heldout_ci_upper"),
    ("pd", "heldout_pd"),
    ("heldout_support_for_training_direction", "heldout_support_for_training_direction"),
    ("evaluation_status", "evaluation_status"),
    ("diagnostic_status", "diagnostic_status"),
]

SIGN = {"positive": 1, "negative": -1, "undefined": 0}


def load_reporting_pairs(reporting_file):
    """main_result_48_units.csv -> DataFrame(outcome, unit_id, unit_type)."""
    r = pd.read_csv(reporting_file)
    rows = []
    for x in r.itertuples():
        for lab in str(x.Outcomes).split(";"):
            lab = lab.strip()
            if lab not in REPORTING_OUTCOME_CODE:
                raise ValueError(f"unknown outcome label {lab!r} in {reporting_file}")
            rows.append(dict(outcome=REPORTING_OUTCOME_CODE[lab], unit_id=x.Unit_Code,
                             unit_type="cluster" if x.Is_Cluster == "Yes" else "singleton"))
    pairs = pd.DataFrame(rows)
    if pairs.duplicated(["outcome", "unit_id"]).any():
        raise ValueError("duplicate unit-outcome pairs in the reporting file")
    return pairs


def load_orientation(orientation_file):
    """CSV with columns unit_id, member, orientation (+1 / -1), optionally
    reference_member. Returns {(unit_id, member): +1/-1}."""
    if orientation_file is None:
        return None
    o = pd.read_csv(orientation_file)
    if not set(o.orientation.unique()) <= {1, -1}:
        raise ValueError("orientation must be +1 or -1")
    return {(u, m): int(s) for u, m, s in zip(o.unit_id, o.member, o.orientation)}


def _usage(series):
    vc = series.value_counts()
    return "; ".join(f"{k}:{int(v)}" for k, v in sorted(vc.items()))


def _stats(s):
    s = s.dropna()
    if not len(s):
        return dict(support_mean=np.nan, support_median=np.nan,
                    support_min=np.nan, support_max=np.nan)
    return dict(support_mean=s.mean(), support_median=s.median(),
                support_min=s.min(), support_max=s.max())


def _rank_within_outcome(df):
    """support_median desc, then n_evaluated desc; equal pairs share the rank;
    unevaluated pairs get NA and go last."""
    df = df.copy()
    df["rank_within_outcome"] = pd.array([pd.NA] * len(df), dtype="Int64")
    out = []
    for _, g in df.groupby("outcome", sort=False):
        ev = g[g.n_evaluated > 0].sort_values(["support_median", "n_evaluated"],
                                              ascending=[False, False])
        rank, prev = 0, None
        for i, (idx, row) in enumerate(ev.iterrows(), 1):
            key = (row.support_median, row.n_evaluated)
            if key != prev:
                rank, prev = i, key
            ev.at[idx, "rank_within_outcome"] = rank
        out += [ev, g[g.n_evaluated == 0]]
    return pd.concat(out)


def summarise_split_support(per_split_files, reporting_file, n_splits=10,
                            orientation_file=None, support_ref=0.90,
                            out_dir=".", out_prefix="splitsupport", verbose=True):
    """
    per_split_files : {outcome_code: path to <prefix>_per_split.csv}. Reporting
        pairs whose outcome is not in this dict are skipped and listed.
    reporting_file  : main_result_48_units.csv
    n_splits        : number of splits each per-split file must contain
    orientation_file: optional, see load_orientation(); enables aligned
        training-sign counts for clusters
    support_ref     : value for the descriptive count n_support_ge_<ref>

    Returns (split_level, unit_summary, member_summary).
    """
    os.makedirs(out_dir, exist_ok=True)
    pairs = load_reporting_pairs(reporting_file)
    orient = load_orientation(orientation_file)
    skipped = sorted(set(pairs.outcome) - set(per_split_files))
    pairs = pairs[pairs.outcome.isin(per_split_files)].reset_index(drop=True)

    split_rows = []
    for o, path in per_split_files.items():
        ps = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in ps.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}. This per-split file was "
                             "written before the 2026-10-13 split_selection_inference.py.")
        seeds = sorted(ps.split_seed.unique())
        if seeds != list(range(n_splits)):
            raise ValueError(f"{path}: split seeds {seeds}, expected 0..{n_splits - 1}")
        if ps.duplicated(["split_seed", "predictor"]).any():
            raise ValueError(f"{path}: a unit appears twice in one split")
        sizes = ps.groupby("split_seed")[["training_n", "heldout_n"]].agg(
            lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
        if sizes.isna().any().any():
            raise ValueError(f"{path}: sample sizes differ within a split")
        by_key = ps.set_index(["split_seed", "predictor"])

        for p in pairs[pairs.outcome == o].itertuples():
            for s in range(n_splits):
                row = dict(outcome=o, unit_id=p.unit_id, unit_type=p.unit_type, split_id=s,
                           training_n=int(sizes.loc[s, "training_n"]),
                           heldout_n=int(sizes.loc[s, "heldout_n"]))
                if (s, p.unit_id) in by_key.index:
                    z = by_key.loc[(s, p.unit_id)]
                    if p.unit_type == "singleton" and z.variable_fitted != p.unit_id:
                        raise ValueError(f"{o}/{p.unit_id} split {s}: singleton fitted as "
                                         f"{z.variable_fitted}")
                    row["selected_in_training"] = True
                    row.update({new: z[old] for old, new in SPLIT_COLS})
                else:
                    row["selected_in_training"] = False
                    row.update({new: np.nan for _, new in SPLIT_COLS})
                    row["evaluation_status"] = "not_selected"
                    row["representative_id"] = None
                split_rows.append(row)

    split_level = pd.DataFrame(split_rows)
    sl = split_level
    ev_mask = (sl.evaluation_status == "evaluated") & sl.heldout_support_for_training_direction.notna()
    sl["_evaluated"] = ev_mask
    sl["_sign"] = sl.training_direction.map(SIGN)

    unit_rows, member_rows = [], []
    for (o, u), g in sl.groupby(["outcome", "unit_id"], sort=False):
        utype = g.unit_type.iloc[0]
        sel, ev = g[g.selected_in_training], g[g._evaluated]
        sup = ev.heldout_support_for_training_direction
        n_und = int((sel.evaluation_status == "undefined_direction").sum())
        n_fail = int((sel.evaluation_status == "fit_failure").sum())
        if len(ev):
            reason = ""
        elif not len(sel):
            reason = "never_selected"
        else:
            reason = f"selected_not_evaluated(undefined_direction={n_und};fit_failure={n_fail})"
        r = dict(outcome=o, unit_id=u, unit_type=utype,
                 n_selected=len(sel), n_splits=n_splits, n_evaluated=len(ev),
                 n_undefined_direction=n_und, n_fit_failure=n_fail,
                 **_stats(sup),
                 **{f"n_support_ge_{support_ref:.2f}": int((sup >= support_ref).sum())},
                 n_evaluated_rhat_flag=int(ev.diagnostic_status.str.contains("rhat>1.05").sum()),
                 not_evaluated_reason=reason)
        # training signs over the evaluated splits only (sum = n_evaluated)
        if utype == "singleton":
            r.update(n_train_positive=int((ev._sign == 1).sum()),
                     n_train_negative=int((ev._sign == -1).sum()),
                     n_train_positive_aligned=np.nan, n_train_negative_aligned=np.nan)
        else:
            r.update(n_train_positive=np.nan, n_train_negative=np.nan)
            if orient is not None and len(ev):
                miss = [m for m in ev.representative_id.unique() if (u, m) not in orient]
                if miss:
                    raise ValueError(f"orientation file has no entry for {u}: {miss}")
                al = ev._sign * ev.representative_id.map(lambda m: orient[(u, m)])
                r.update(n_train_positive_aligned=int((al == 1).sum()),
                         n_train_negative_aligned=int((al == -1).sum()))
            else:
                r.update(n_train_positive_aligned=np.nan, n_train_negative_aligned=np.nan)
        r["train_signs_by_member"] = "; ".join(
            f"{m}:+{int((h._sign == 1).sum())}/-{int((h._sign == -1).sum())}"
            for m, h in sorted(ev.groupby("representative_id")))
        r["members_selected"] = _usage(sel.representative_id)
        r["members_evaluated"] = _usage(ev.representative_id)
        unit_rows.append(r)

        if utype == "cluster":
            for m, h in sorted(sel.groupby("representative_id")):
                he = h[h._evaluated]
                member_rows.append(dict(
                    outcome=o, unit_id=u, member=m, n_selected_as_representative=len(h),
                    n_evaluated=len(he), **_stats(he.heldout_support_for_training_direction),
                    n_train_positive=int((he._sign == 1).sum()),
                    n_train_negative=int((he._sign == -1).sum())))

    unit_summary = _rank_within_outcome(pd.DataFrame(unit_rows))
    member_summary = pd.DataFrame(member_rows)
    split_level = sl.drop(columns=["_evaluated", "_sign"])

    # evaluated == selected is checked on every run, not assumed
    not_eval = unit_summary[unit_summary.n_evaluated != unit_summary.n_selected]
    all_evaluated = not len(not_eval)
    if all_evaluated:
        unit_summary = unit_summary.drop(columns=["n_evaluated", "n_undefined_direction",
                                                  "n_fit_failure", "members_evaluated"])
        if len(member_summary):
            member_summary = member_summary.drop(columns=["n_evaluated"])

    for df in (split_level, unit_summary, member_summary):
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].round(4)

    split_level.to_csv(os.path.join(out_dir, f"{out_prefix}_split_level.csv"), index=False)
    unit_summary.to_csv(os.path.join(out_dir, f"{out_prefix}_unit_summary.csv"), index=False)
    member_summary.to_csv(os.path.join(out_dir, f"{out_prefix}_member_summary.csv"), index=False)
    meta = dict(per_split_files={o: str(p) for o, p in per_split_files.items()},
                reporting_file=str(reporting_file), n_splits=n_splits,
                orientation_file=None if orientation_file is None else str(orientation_file),
                support_ref=support_ref,
                n_pairs=len(unit_summary), outcomes_skipped=skipped,
                all_selected_splits_evaluated=bool(all_evaluated))
    with open(os.path.join(out_dir, f"{out_prefix}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        if skipped:
            print(f"outcomes in the reporting file without a per-split file (skipped): {skipped}")
        print(f"{len(unit_summary)} unit-outcome pairs, {len(split_level)} split-level rows")
        if all_evaluated:
            print(f"all {int(split_level.selected_in_training.sum())} selected unit-splits "
                  f"were evaluated; n_evaluated equals n_selected and is not shown")
        else:
            print(f"WARNING: {len(not_eval)} pairs have selected splits that were not "
                  f"evaluated; n_evaluated, n_undefined_direction, n_fit_failure and "
                  f"members_evaluated are kept:")
            print(not_eval[["outcome", "unit_id", "n_selected", "n_evaluated",
                            "n_undefined_direction", "n_fit_failure"]].to_string(index=False))
        cols = ["outcome", "unit_id", "rank_within_outcome", "support_median", "n_selected",
                "support_mean", "support_min", "support_max",
                "n_train_positive", "n_train_negative", "not_evaluated_reason"]
        if not all_evaluated:
            cols.insert(5, "n_evaluated")
        print(unit_summary[cols].to_string(index=False))
    return split_level, unit_summary, member_summary
