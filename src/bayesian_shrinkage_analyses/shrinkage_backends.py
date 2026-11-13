"""
Pluggable shrinkage backends for the selection stage.

Every backend fits one model on one design matrix and returns the same record,
so split_selection_inference does not care which algorithm produced it:

    included  (draws, p) bool   per-draw inclusion mask
    beta      (draws, p) float  coefficient draws
    score     (p,)       float  ranking statistic (inclusion BF10)
    prior_incl_prob      float  P(included) under the prior alone -- the
                                denominator for the POOLED BF10 computed across
                                repeats. Differs by backend, so the backend
                                supplies it rather than the caller assuming one.
    n_draws              int
    diag                 dict   backend-specific diagnostics; keys the caller
                                aggregates (max_rhat, min_ess_bulk,
                                n_params_rhat_gt_1.05, n_divergences) are always
                                present, NaN where the algorithm has no analogue.

BACKENDS

  "horseshoe"  regularized horseshoe via NUTS (ssvs_horseshoe / _twin).
               Inclusion is the shrinkage-factor threshold kappa_j < 0.5, a
               continuous relaxation -- there is no exact zero. prior_incl_prob
               comes from prior_inclusion_baseline(n, p, p/11) and is ~0.134.
               Supports covariates (Z).

  "slabstick"  discrete spike-and-slab Gibbs (ssvs_slabstick / _twin).
               Inclusion is the sampled indicator z_j == 1, an exact zero.
               prior_incl_prob is analytic: a0/(a0+b0), = 1/11 at the default
               Beta(1,10), which matches the horseshoe's p0 = p/11 target.
               Supports covariates (Z) as of the revised samplers: an unshrunk
               Normal(0, 5^2) block with its own conjugate Gibbs step, no
               inclusion indicator, same role as gamma in ssvs_horseshoe.

The two are not on the same scale: kappa < 0.5 counts any draw where the
likelihood contributes more than half the coefficient, which is a looser reading
of "included" than z == 1. Measured on EASE/ADOS, horseshoe escape rates run
2-4x the slabstick PIPs for the same predictor, while BF10 -- which divides by
each backend's own prior -- agrees to within a few hundredths.
"""

import numpy as np

BACKENDS = ("horseshoe", "slabstick")


def _diag_skeleton():
    return {"max_rhat": np.nan, "min_ess_bulk": np.nan,
            "n_params_rhat_gt_1.05": np.nan, "n_divergences": np.nan}


# ---------------------------------------------------------------------------
# horseshoe (NUTS)
# ---------------------------------------------------------------------------
def fit_horseshoe_backend(X, y, *, pair_ids=None, Z=None, family="normal", seed=0,
                          num_warmup=2000, num_samples=5000, num_chains=6,
                          want_rhat=True, **ignored):
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
    from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst

    n, p = X.shape
    if pair_ids is not None:
        mcmc, _ = hst.fit_twin_horseshoe(X, y, pair_ids, Z=Z, family=family, seed=seed,
                                         num_warmup=num_warmup, num_samples=num_samples,
                                         num_chains=num_chains, progress_bar=False)
        module = hst
    else:
        mcmc = hs.fit_horseshoe(X, y, Z=Z, family=family, seed=seed,
                                num_warmup=num_warmup, num_samples=num_samples,
                                num_chains=num_chains, progress_bar=False)
        module = hs

    kap = module._kappa(mcmc, n)
    included = kap < 0.5
    beta = np.asarray(mcmc.get_samples()["beta"])
    score = np.asarray(module.inclusion_bayes_factor(mcmc, n), dtype=float)
    prior_p = float(hs.prior_inclusion_baseline(n, p, p / 11.0))

    diag = _diag_skeleton()
    try:
        diag["n_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    except Exception:
        pass
    if want_rhat and num_chains > 1:
        import arviz as az
        s = az.summary(az.from_numpyro(mcmc), round_to=None)
        diag["max_rhat"] = float(s["r_hat"].max())
        diag["n_params_rhat_gt_1.05"] = int((s["r_hat"] > 1.05).sum())
        diag["min_ess_bulk"] = float(s["ess_bulk"].min())

    del mcmc, kap
    return dict(included=included, beta=beta, score=score,
                prior_incl_prob=prior_p, n_draws=int(beta.shape[0]), diag=diag)


# ---------------------------------------------------------------------------
# slab-stick (discrete spike-and-slab Gibbs)
# ---------------------------------------------------------------------------
def fit_slabstick_backend(X, y, *, pair_ids=None, Z=None, family="normal", seed=0,
                          n_iter=5000, burn=1000, a0=1.0, b0=10.0, ig_a=2.0, ig_b=1.0,
                          **ignored):
    from src.bayesian_shrinkage_analyses import ssvs_slabstick as ss
    from src.bayesian_shrinkage_analyses import ssvs_slabstick_twin as sst

    if pair_ids is not None:
        betas, zs, sigma_pairs, sigma_withins, gammas = sst.fit_twin_jax_gibbs(
            X, y, pair_ids, Z=Z, family=family, n_iter=n_iter, burn=burn,
            a0=a0, b0=b0, ig_a=ig_a, ig_b=ig_b, seed=seed)
    else:
        betas, zs, gammas = ss.fit_jax_gibbs(
            X, y, Z=Z, family=family, n_iter=n_iter, burn=burn,
            a0=a0, b0=b0, ig_a=ig_a, ig_b=ig_b, seed=seed)
        sigma_pairs = sigma_withins = None

    included = zs > 0
    pip = included.mean(0)
    prior_p = ss.prior_inclusion_baseline(a0, b0)
    score = np.asarray(ss.inclusion_bayes_factor(zs, a0=a0, b0=b0), dtype=float)

    diag = _diag_skeleton()          # conjugate Gibbs: no r_hat/ESS/divergences
    diag["mean_n_included_per_draw"] = float(included.sum(1).mean())
    diag["max_pip"] = float(pip.max())
    if sigma_pairs is not None:
        diag["mean_sigma_pair2"] = float(np.mean(sigma_pairs))
    if gammas.shape[1]:
        diag["gamma_mean"] = [round(float(v), 4) for v in gammas.mean(0)]

    return dict(included=included, beta=np.asarray(betas, dtype=float), score=score,
                prior_incl_prob=float(prior_p), n_draws=int(betas.shape[0]), diag=diag)


_REGISTRY = {"horseshoe": fit_horseshoe_backend, "slabstick": fit_slabstick_backend}


def get_backend(name):
    if name not in _REGISTRY:
        raise ValueError(f"algorithm must be one of {BACKENDS}, got {name!r}")
    return _REGISTRY[name]


def default_settings(name):
    """MCMC settings a backend is normally run with, for printouts and meta."""
    if name == "horseshoe":
        return dict(num_warmup=2000, num_samples=5000, num_chains=6)
    return dict(n_iter=5000, burn=1000, a0=1.0, b0=10.0, ig_a=2.0, ig_b=1.0)
