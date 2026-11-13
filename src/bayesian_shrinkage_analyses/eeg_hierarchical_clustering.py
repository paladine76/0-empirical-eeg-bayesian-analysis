"""
Runs hierarchical clustering (complete linkage) on the correlation matrix of
EEG variables, cutting the dendrogram at r >= 0.85 (equivalent to a distance
threshold of 1 - 0.85 = 0.15 using 1-|r| distance). Singleton clusters are
discarded; only clusters with >=2 variables are reported.

Analogous to ssvs_horseshoe.py / ssvs_horseshoe_twin.py: the main function
takes a plain numpy array (X) plus a list of variable names, exactly like
fit_horseshoe(X, y, ...) takes X/y as numpy arrays. No dataframe library
(pandas or polars), no file I/O, and no column dropping/exclusion/dataset
combining happens anywhere in this module -- all of that is the caller's
responsibility (e.g. the notebook's "Load + build design matrix" cells),
just as X/y/Z are built in the notebook before being passed to
fit_horseshoe() / fit_twin_horseshoe().
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def run_hierarchical_clustering(X, variable_names, threshold=0.85, method="complete"):
    """
    X: (n_obs, n_vars) numpy array of EEG variables, ASSUMED ALREADY COMBINED/FILTERED
    by the caller (analogous to X in fit_horseshoe(X, y, ...) being assumed
    already standardized -- no I/O or dataframe handling here).
    variable_names: length-n_vars list of column names corresponding to X's columns.
    threshold: correlation cutoff (r >= threshold) for grouping variables into a cluster.
    method: linkage method passed to scipy.cluster.hierarchy.linkage (default 'complete').

    Returns (results, cluster_labels, corr, variable_names) where:
    results: list of dicts, one per non-singleton cluster, each with keys
    'cluster_id', 'variables' (list of names), 'sub_corr' (within-cluster
    correlation sub-matrix).
    cluster_labels: (n_vars,) array of cluster ids, one per variable (aligned
    with variable_names), from scipy.cluster.hierarchy.fcluster.
    corr: (n_vars, n_vars) full correlation matrix.
    """
    X = np.asarray(X, dtype=float)
    n_vars = X.shape[1]
    assert len(variable_names) == n_vars, "variable_names length must match X.shape[1]"

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist_condensed = squareform(dist, checks=False)

    Z = linkage(dist_condensed, method=method)
    threshold_dist = 1 - threshold
    cluster_labels = fcluster(Z, t=threshold_dist, criterion='distance')

    uniq, counts = np.unique(cluster_labels, return_counts=True)
    multi_clusters = uniq[counts > 1]
    sizes_map = dict(zip(uniq, counts))

    results = []
    for cid in sorted(multi_clusters, key=lambda c: -sizes_map[c]):
        idx = np.where(cluster_labels == cid)[0]
        vars_in = [variable_names[i] for i in idx]
        sub_corr = corr[np.ix_(idx, idx)]
        results.append({'cluster_id': int(cid), 'variables': vars_in, 'sub_corr': sub_corr})

    return results, cluster_labels, corr, variable_names

def report_results(results):
    """
    Pure transform, no I/O: turns the results list (from run_hierarchical_clustering)
    into a flat list of dicts (one row per variable per cluster), ready for the caller
    to wrap in whatever dataframe library it prefers (pandas, polars, etc.).
    """
    rows = []
    for r in results:
        cid = r['cluster_id']
        vars_in = r['variables']
        sub_corr = r['sub_corr']
        for i, v in enumerate(vars_in):
            others = [(j, o) for j, o in enumerate(vars_in) if o != v]
            corr_str = "; ".join([f"{o}={sub_corr[i, j]:.3f}" for j, o in others])
            rows.append({
                'cluster': cid,
                'n_vars': len(vars_in),
                'variable': v,
                'correlations_with_others': corr_str
            })

    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 200, 30
    X_demo = rng.normal(size=(n, p))
    X_demo[:, 1] = X_demo[:, 0] * 0.95 + rng.normal(scale=0.1, size=n)  # correlated pair
    names_demo = [f"v{i}" for i in range(p)]

    results, cluster_labels, corr, names = run_hierarchical_clustering(
        X_demo, names_demo, threshold=0.85
    )
    print(f"Number of non-singleton clusters: {len(results)}")
    for r in results:
        print(f"Cluster {r['cluster_id']} (n={len(r['variables'])}): {r['variables']}")