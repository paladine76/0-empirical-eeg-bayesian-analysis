"""
Plain Bayesian linear/logistic MIXED model baseline for twin-pair data (e.g.
batss2) -- NO shrinkage, NO variable selection, just weakly-informative
Normal priors on every coefficient plus a pair-level random intercept.
Serves as the naive baseline that the spike-and-slab / horseshoe methods
elsewhere in this project are compared against.

  eta_i = alpha + X_i @ beta + u_{pair(i)}  (+ Z_i @ gamma if Z given)
  u_g ~ Normal(0, sigma_pair)          -- pair-level random intercept
  beta_j ~ Normal(0, prior_scale^2)    [same prior for every predictor]
  gamma_k ~ Normal(0, 5^2)             [forced-in covariates]
  normal:    y_i ~ Normal(eta_i, sigma_within)
  bernoulli: y_i ~ Bernoulli(logits=eta_i)
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def _make_model(family, prior_scale, G):
    def model(X, pair_idx, Z=None, y=None):
        n, p = X.shape
        with numpyro.plate("predictors", p):
            beta = numpyro.sample("beta", dist.Normal(0.0, prior_scale))

        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
        sigma_pair = numpyro.sample("sigma_pair", dist.HalfNormal(1.0))
        with numpyro.plate("pairs", G):
            u = numpyro.sample("u", dist.Normal(0.0, sigma_pair))

        eta = alpha + X @ beta + u[pair_idx]

        if Z is not None:
            q = Z.shape[1]
            with numpyro.plate("covariates", q):
                gamma = numpyro.sample("gamma", dist.Normal(0.0, 5.0))
            eta = eta + Z @ gamma

        if family == "normal":
            sigma_within = numpyro.sample("sigma_within", dist.HalfNormal(1.0))
            numpyro.sample("obs", dist.Normal(eta, sigma_within), obs=y)
        elif family == "bernoulli":
            numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)
        else:
            raise ValueError("family must be 'normal' or 'bernoulli'")

    return model


def fit_bayesian_lmm(X, y, pair_ids, Z=None, family="normal", prior_scale=2.5, seed=0,
                      num_warmup=2000, num_samples=3000, num_chains=4,
                      target_accept_prob=0.95, progress_bar=True):
    """
    X: (n,p) predictors, ASSUMED ALREADY STANDARDIZED -- not done internally.
    y: (n,) outcome, similarly assumed standardized for family="normal"
    (0/1 for family="bernoulli"). pair_ids: length-n twin-pair ids
    (singletons and complete pairs both fine). Z: optional forced-in covariates.
    prior_scale: SD of the (shared, non-shrinking) Normal prior on every
    beta_j -- 2.5 is a standard weakly-informative default.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z_arr = None if Z is None else jnp.array(np.asarray(Z, dtype=float))
    pair_codes, uniques = pd.factorize(pd.Series(pair_ids))
    G = len(uniques)

    model = _make_model(family, prior_scale, G)
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.array(X), pair_idx=jnp.array(pair_codes),
              Z=Z_arr, y=jnp.array(y))
    return mcmc, uniques


def posterior_summary(mcmc, level=0.95):
    """
    Per-predictor beta_mean, pd, and (level*100)% credible interval, all
    from the FULL posterior (no shrinkage conditioning applies here).
    pd_j = max(P(beta_j>0), P(beta_j<0)), range 0.5-1.0.
    Returns (beta_mean, pd, ci_lower, ci_upper).
    """
    beta = np.array(mcmc.get_samples()["beta"])
    beta_mean = beta.mean(axis=0)
    p_pos = (beta > 0).mean(axis=0)
    pd_ = np.maximum(p_pos, 1.0 - p_pos)
    a = (1 - level) / 2 * 100
    ci_lo, ci_hi = np.percentile(beta, [a, 100 - a], axis=0)
    return beta_mean, pd_, ci_lo, ci_hi


def savage_dickey_bf10(mcmc, prior_scale=2.5):
    """ORDINARY Bayes factor (Savage-Dickey density ratio), per predictor:
    BF10_j = prior_density(0) / posterior_density(0) (Gaussian KDE on the
    beta_j draws). See baseline_bayesian_lm.py."""
    from scipy.stats import gaussian_kde, norm
    beta = np.array(mcmc.get_samples()["beta"])
    p = beta.shape[1]
    prior_density_0 = norm.pdf(0.0, loc=0.0, scale=prior_scale)
    bf10 = np.zeros(p)
    for j in range(p):
        kde = gaussian_kde(beta[:, j])
        post_density_0 = max(kde(0.0)[0], 1e-300)
        bf10[j] = prior_density_0 / post_density_0
    return bf10


def icc_summary(mcmc, family="normal"):
    """ICC = sigma_pair^2 / (sigma_pair^2 + sigma_within^2). For
    family='bernoulli', sigma_within^2 is fixed at pi^2/3 (logistic latent
    variance) since there is no free residual-scale parameter."""
    samples = mcmc.get_samples()
    sp = np.array(samples["sigma_pair"])**2
    if family == "normal":
        sw = np.array(samples["sigma_within"])**2
    elif family == "bernoulli":
        sw = (np.pi**2) / 3.0
    else:
        raise ValueError("family must be 'normal' or 'bernoulli'")
    icc = sp / (sp + sw)
    return dict(mean=float(icc.mean()), ci5=float(np.percentile(icc, 5.5)), ci95=float(np.percentile(icc, 94.5)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    G = 150; p = 15
    true_idx = [2, 7]
    pair_ids, X_rows = [], []
    for g in range(G):
        n_members = 2 if rng.random() < 0.8 else 1
        for _ in range(n_members):
            pair_ids.append(g); X_rows.append(rng.normal(size=p))
    X = np.array(X_rows); pair_ids = np.array(pair_ids); n = X.shape[0]
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)

    b_true = np.zeros(p); b_true[true_idx] = [0.6, -0.5]
    sigma_pair_true, sigma_within_true = 0.6, 0.8   # ICC = 0.36/0.64 = 0.36
    u_true = rng.normal(0, sigma_pair_true, size=G)
    y = X @ b_true + u_true[pair_ids] + rng.normal(0, sigma_within_true, size=n)
    y = (y - y.mean()) / y.std()

    print("Fitting plain Bayesian LMM, family=normal (true signals at v2, v7) ...")
    mcmc, _ = fit_bayesian_lmm(Xs, y, pair_ids, family="normal", num_warmup=1000, num_samples=1500,
                                num_chains=2, progress_bar=False)
    beta_mean, pd_, ci_lo, ci_hi = posterior_summary(mcmc)
    icc = icc_summary(mcmc, family="normal")
    print(f"ICC = {icc['mean']:.3f} (true = 0.360)")
    order = np.argsort(-np.abs(beta_mean))[:6]
    for i in order:
        print(f"  idx={i:2d}  beta={beta_mean[i]:+.3f}  pd={pd_[i]:.3f}  CI=[{ci_lo[i]:.3f},{ci_hi[i]:.3f}]  {'<-- TRUE' if i in true_idx else ''}")
