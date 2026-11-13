"""
Cluster-level labeling for the 21 hierarchical clusters found by
eeg_hierarchical_clustering.py (r >= 0.85, complete linkage, on the pooled
BATSS+EASE EEG variables). Used to treat each cluster as a single reporting
unit in the repeated declustering analysis: whichever member variable is
drawn as that cluster's representative in a given run, occurrences in the
top-K ranking are recorded under the cluster's label rather than the
individual variable's name. Singleton variables are unaffected -- they keep
their own name, since each is already a single-variable "cluster" of one.

CLUSTER_LABELS values are descriptive, derived directly from what the
member variables of each cluster have in common (same metric/region/band,
correlated across stimulus condition; or same metric/band/condition,
correlated across region; or PAC/network-metric redundancy). See
cluster_report.csv for the underlying variable membership this is based on.
"""

"""
Cluster-level labeling for the 21 hierarchical clusters found by
eeg_hierarchical_clustering.py (r >= 0.85, complete linkage, on the pooled
BATSS+EASE EEG variables). Used to treat each cluster as a single reporting
unit in the repeated declustering analysis: whichever member variable is
drawn as that cluster's representative in a given run, occurrences in the
top-K ranking are recorded under the cluster's label rather than the
individual variable's name. Singleton variables are unaffected -- they keep
their own name, since each is already a single-variable "cluster" of one.

Two cluster ID NUMBERING SCHEMES are provided for the same 21 clusters --
"A" (the original scipy fcluster ids) and "B" (an alternate numbering, e.g.
from a different clustering run/library version). Verified 1:1 by member
variable set (each cluster in B has an identical variable membership to
exactly one cluster in A; see CLUSTER_ID_MAP). Only the cluster_id keys
differ between schemes -- the descriptive labels and grouping logic
("Type A" condition-invariant, "Type B" region-dominant, "Type C" PAC
region-dominant, "Type D" network-metric equivalence) are identical.
"""

CLUSTER_LABELS_A = {
    # Type A -- condition-invariant (same metric/region/band, fc<->ty)
    1:  "RBP_gamma_rtemp",
    4:  "RBP_gamma_ltemp",
    8:  "RBP_gamma_occip",
    30: "RBP_theta_ltemp",

    # Type B -- region-dominant (whole-scalp signal driven by specific region(s))
    2:  "RBP_gamma_ty_whole-front",
    3:  "RBP_gamma_fc_whole-front",
    29: "RBP_theta_fc_whole-front-rtemp",
    31: "RBP_theta_ty_whole-front-rtemp",
    51: "RBP_alpha_ty_global",   # all 6 regions collapse
    52: "RBP_alpha_fc_global",   # all 6 regions collapse

    # Type C -- PAC region-dominant (coupling-type and condition collapse)
    5:  "PAC_ltemp",
    6:  "PAC_front",
    9:  "PAC_occip",
    10: "PAC_parie",
    7:  "PAC_whole-rtemp",       # largest cluster, 12 members

    # Type D -- network-metric equivalence (clustering coefficient ~= SWI)
    36:  "NetGlobal_theta_ty",
    37:  "NetGlobal_theta_fc",
    53:  "NetGlobal_alpha_fc",
    54:  "NetGlobal_alpha_ty",
    121: "NetGlobal_gamma_fc",
    122: "NetGlobal_gamma_ty",
}

# Scheme B: same 21 clusters, same labels, different id numbers.
# Mapping (scheme A id -> scheme B id), for reference:
#   1->92  2->93  3->94  4->95  5->96  6->97  7->98  8->99  9->100 10->101
#   29->47 30->48 31->49 36->54 37->55 51->61 52->62 53->63 54->64
#   121->144 122->145
CLUSTER_LABELS_B = {
    92:  "RBP_gamma_rtemp",
    95:  "RBP_gamma_ltemp",
    99:  "RBP_gamma_occip",
    48:  "RBP_theta_ltemp",

    93:  "RBP_gamma_ty_whole-front",
    94:  "RBP_gamma_fc_whole-front",
    47:  "RBP_theta_fc_whole-front-rtemp",
    49:  "RBP_theta_ty_whole-front-rtemp",
    61:  "RBP_alpha_ty_global",
    62:  "RBP_alpha_fc_global",

    96:  "PAC_ltemp",
    97:  "PAC_front",
    100: "PAC_occip",
    101: "PAC_parie",
    98:  "PAC_whole-rtemp",

    54:  "NetGlobal_theta_ty",
    55:  "NetGlobal_theta_fc",
    63:  "NetGlobal_alpha_fc",
    64:  "NetGlobal_alpha_ty",
    144: "NetGlobal_gamma_fc",
    145: "NetGlobal_gamma_ty",
}

# Scheme C: the 20 clusters found after removing the 18 whole-scalp RBP, FC and
# PAC phenotypes (bpow_whole_*, funC_whole_*, pac_whole_*) before clustering.
# 66 clustered variables, 95 singletons, 115 predictors per fit. Ids are the
# scipy fcluster ids in cluster_report.csv of that run.
CLUSTER_LABELS_C = {
    # Type A -- condition-invariant (same metric/region/band, fc<->ty)
    12: "RBP_gamma_rtemp",
    14: "RBP_gamma_front",
    15: "RBP_gamma_ltemp",
    17: "RBP_gamma_occip",
    54: "RBP_theta_front",
    55: "RBP_theta_rtemp",
    56: "RBP_theta_ltemp",

    # Type B -- region-dominant (same metric/band/condition, all 5 regions)
    76: "RBP_alpha_ty_global",
    77: "RBP_alpha_fc_global",

    # Type C -- PAC per region (coupling type and condition collapse)
    11: "PAC_rtemp",
    13: "PAC_front",
    16: "PAC_ltemp",
    18: "PAC_occip",
    19: "PAC_parie",

    # Type D -- network-metric equivalence (clustering coefficient ~= SWI)
    52: "NetGlobal_gamma_fc",
    53: "NetGlobal_gamma_ty",
    61: "NetGlobal_theta_ty",
    62: "NetGlobal_theta_fc",
    78: "NetGlobal_alpha_fc",
    79: "NetGlobal_alpha_ty",
}

CLUSTER_LABEL_SCHEMES = {"A": CLUSTER_LABELS_A, "B": CLUSTER_LABELS_B, "C": CLUSTER_LABELS_C}

# Backwards-compatible alias: existing code importing CLUSTER_LABELS keeps working unchanged.
CLUSTER_LABELS = CLUSTER_LABELS_A


def build_predictor_to_label_map(cluster_vars, cluster_labels=None, scheme="A"):
    """
    cluster_vars: dict {cluster_id: [variable names]}, e.g. CLUSTER_VARS from
    the notebook (built from run_hierarchical_clustering()'s output). Its
    cluster_id keys must match the chosen scheme's numbering.
    cluster_labels: dict {cluster_id: label}, overrides `scheme` if given.
    scheme: "A", "B" or "C" -- selects CLUSTER_LABELS_A / _B / _C when
    cluster_labels is not explicitly provided. "C" is the 20-cluster solution
    without the whole-scalp RBP/FC/PAC phenotypes.

    Returns dict {variable_name: cluster_label} covering every variable that
    belongs to a cluster. Variables not present in the returned dict (i.e.
    singletons) should be reported under their own name -- see
    to_reporting_name() below, which handles that fallback.
    """
    if cluster_labels is None:
        if scheme not in CLUSTER_LABEL_SCHEMES:
            raise ValueError(f"Unknown scheme {scheme!r}; choose from {list(CLUSTER_LABEL_SCHEMES)}")
        cluster_labels = CLUSTER_LABEL_SCHEMES[scheme]

    missing = set(cluster_vars) - set(cluster_labels)
    if missing:
        raise ValueError(f"No label defined for cluster id(s): {sorted(missing)}")
    predictor_to_label = {}
    for cid, variables in cluster_vars.items():
        label = cluster_labels[cid]
        for v in variables:
            predictor_to_label[v] = label
    return predictor_to_label


def to_reporting_name(predictor, predictor_to_label):
    """Cluster label if predictor belongs to a cluster, else the predictor's own name."""
    return predictor_to_label.get(predictor, predictor)


def remap_predictor_column(df, predictor_to_label, column="predictor"):
    """
    Returns a copy of df with `column` remapped through to_reporting_name():
    clustered variables become their cluster's label, singletons are
    unchanged. Use this once, right after building a run's results_dataframe,
    so every downstream aggregation (top-K lists, beta_records, etc.)
    naturally operates in the collapsed cluster-or-singleton identity space.
    """
    out = df.copy()
    out[column] = out[column].map(lambda p: to_reporting_name(p, predictor_to_label))
    return out