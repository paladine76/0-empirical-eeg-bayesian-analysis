"""Clustering and cluster-label remapping: the declustering step depends on
both being correct, and an error here silently corrupts every downstream
selection frequency."""
import numpy as np
import pandas as pd
import pytest

from src.bayesian_shrinkage_analyses import eeg_hierarchical_clustering as hcl
from src.bayesian_shrinkage_analyses import cluster_labels as clab


@pytest.fixture(scope="module")
def clustered():
    """Two planted redundant blocks (r ~ .99) plus independent noise columns."""
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    cols = {
        "blockA_1": a,
        "blockA_2": a * 0.99 + rng.normal(0, 0.05, n),
        "blockA_3": a * 0.98 + rng.normal(0, 0.05, n),
        "blockB_1": b,
        "blockB_2": -b * 0.99 + rng.normal(0, 0.05, n),   # sign-flipped: |r| is used
    }
    for i in range(4):
        cols[f"noise_{i}"] = rng.normal(size=n)
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    res, labels, corr, _ = hcl.run_hierarchical_clustering(X, names, threshold=0.85)
    return dict(res=res, names=names)


def test_redundant_blocks_are_grouped(clustered):
    groups = [set(r["variables"]) for r in clustered["res"]]
    assert {"blockA_1", "blockA_2", "blockA_3"} in groups
    assert {"blockB_1", "blockB_2"} in groups


def test_independent_columns_stay_singletons(clustered):
    in_clusters = {v for r in clustered["res"] for v in r["variables"]}
    assert not any(nm.startswith("noise_") for nm in in_clusters)


def test_clustering_uses_absolute_correlation(clustered):
    """blockB_2 is anti-correlated with blockB_1; it must still cluster,
    otherwise sign-flipped duplicates escape declustering."""
    groups = [set(r["variables"]) for r in clustered["res"]]
    assert any({"blockB_1", "blockB_2"} <= g for g in groups)


def test_higher_threshold_groups_no_more_than_lower(clustered):
    """Monotonicity sanity check on the cut."""
    rng = np.random.default_rng(1)
    n = 200
    a = rng.normal(size=n)
    names = ["x1", "x2", "x3"]
    X = np.column_stack([a, a * 0.9 + rng.normal(0, .2, n), rng.normal(size=n)])
    loose, _, _, _ = hcl.run_hierarchical_clustering(X, names, threshold=0.70)
    tight, _, _, _ = hcl.run_hierarchical_clustering(X, names, threshold=0.99)
    n_loose = sum(len(r["variables"]) for r in loose)
    n_tight = sum(len(r["variables"]) for r in tight)
    assert n_tight <= n_loose


# ---- label remapping ----

def test_every_cluster_member_maps_to_its_label():
    cvars = {1: ["a1", "a2"], 4: ["b1", "b2", "b3"]}
    p2l = clab.build_predictor_to_label_map(cvars)
    assert len(p2l) == 5
    assert len({p2l[v] for v in ["a1", "a2"]}) == 1
    assert p2l["a1"] != p2l["b1"]


def test_singletons_pass_through_unchanged():
    p2l = clab.build_predictor_to_label_map({1: ["a1", "a2"]})
    assert clab.to_reporting_name("some_singleton", p2l) == "some_singleton"
    assert clab.to_reporting_name("a1", p2l) == p2l["a1"]


def test_unlabelled_cluster_id_raises():
    """Silently dropping an unlabelled cluster would under-count selections."""
    with pytest.raises(ValueError):
        clab.build_predictor_to_label_map({999999: ["z1", "z2"]})


def test_remap_column_collapses_members_and_keeps_others():
    p2l = clab.build_predictor_to_label_map({1: ["a1", "a2"]})
    df = pd.DataFrame({"predictor": ["a1", "a2", "keep_me"], "BF10": [3.0, 2.0, 1.0]})
    out = clab.remap_predictor_column(df, p2l)
    assert out.loc[0, "predictor"] == out.loc[1, "predictor"] == p2l["a1"]
    assert out.loc[2, "predictor"] == "keep_me"
    assert list(df["predictor"]) == ["a1", "a2", "keep_me"]   # input not mutated
