"""
Plain Bayesian linear/logistic model baseline for single-cohort (non-twin)
data -- NO shrinkage, NO variable selection, just weakly-informative Normal
priors on every coefficient. Serves as the naive baseline that the
spike-and-slab / horseshoe methods elsewhere in this project are compared
against.

  eta_i = alpha + X_i @ beta  (+ Z_i @ gamma if Z given)
  beta_j ~ Normal(0, prior_scale^2)   [same prior for every predictor, no
                                        sparsity/shrinkage hierarchy]
  gamma_k ~ Normal(0, 5^2)            [forced-in covariates, as elsewhere]
  normal:    y_i ~ Normal(eta_i, sigma)
  bernoulli: y_i ~ Bernoulli(logits=eta_i)
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


def _make_model(family, prior_scale):
    def model(X, Z=None, y=None):
        n, p = X.shape
        with numpyro.plate("predictors", p):
            beta = numpyro.sample("beta", dist.Normal(0.0, prior_scale))

        eta = numpyro.sample("alpha", dist.Normal(0.0, 5.0)) + X @ beta

        if Z is not None:
            q = Z.shape[1]
            with numpyro.plate("covariates", q):
                gamma = numpyro.sample("gamma", dist.Normal(0.0, 5.0))
            eta = eta + Z @ gamma

        if family == "normal":
            sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
            numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
        elif family == "bernoulli":
            numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)
        else:
            raise ValueError("family must be 'normal' or 'bernoulli'")

    return model


def fit_bayesian_lm(X, y, Z=None, family="normal", prior_scale=2.5, seed=0,
                     num_warmup=2000, num_samples=3000, num_chains=4,
                     target_accept_prob=0.95, progress_bar=True):
    """
    X: (n,p) predictors, ASSUMED ALREADY STANDARDIZED -- not done internally.
    y: (n,) outcome, similarly assumed standardized for family="normal"
    (0/1 for family="bernoulli"). Z: optional forced-in covariates.
    prior_scale: SD of the (shared, non-shrinking) Normal prior on every
    beta_j -- 2.5 is a standard weakly-informative default for standardized
    predictors/outcome (Gelman et al.).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z_arr = None if Z is None else jnp.array(np.asarray(Z, dtype=float))

    model = _make_model(family, prior_scale)
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.array(X), Z=Z_arr, y=jnp.array(y))
    return mcmc


def posterior_summary(mcmc, level=0.95):
    """
    Per-predictor beta_mean, pd, and (level*100)% credible interval, all
    from the FULL posterior (no inclusion/shrinkage conditioning applies --
    every coefficient here already has a genuine, unshrunk marginal
    posterior). pd_j = max(P(beta_j>0), P(beta_j<0)), range 0.5-1.0.
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
    """
    Per-predictor ORDINARY Bayes factor (Savage-Dickey density ratio, not
    the inclusion-odds BF used for spike-and-slab/horseshoe): for testing
    H0: beta_j=0 vs H1: beta_j!=0,
        BF10_j = prior_density(0) / posterior_density(0)
    Posterior density at 0 estimated via Gaussian KDE on the beta_j draws
    (standard practice for Savage-Dickey; exact under a Gaussian-Gaussian
    conjugate model, which a plain linear model with Normal prior/likelihood
    approximately is). Valid here since the model has no spike/slab
    structure -- every beta_j already has a single, genuine marginal density.
    """
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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 88, 15
    X = rng.normal(size=(n, p))
    true_idx = [2, 7]
    b_true = np.zeros(p); b_true[true_idx] = [0.8, -0.6]
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)

    y = X @ b_true + rng.normal(0, 0.9, size=n)
    y = (y - y.mean()) / y.std()
    print("Fitting plain Bayesian LM, family=normal (true signals at v2, v7) ...")
    mcmc = fit_bayesian_lm(Xs, y, family="normal", num_warmup=1000, num_samples=1500,
                            num_chains=2, progress_bar=False)
    beta_mean, pd_, ci_lo, ci_hi = posterior_summary(mcmc)
    order = np.argsort(-np.abs(beta_mean))[:6]
    for i in order:
        print(f"  idx={i:2d}  beta={beta_mean[i]:+.3f}  pd={pd_[i]:.3f}  CI=[{ci_lo[i]:.3f},{ci_hi[i]:.3f}]  {'<-- TRUE' if i in true_idx else ''}")

    print("\nFitting plain Bayesian LM, family=bernoulli (true signals at v2, v7) ...")
    eta = X @ b_true
    yb = rng.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    mcmc_b = fit_bayesian_lm(Xs, yb, family="bernoulli", num_warmup=1000, num_samples=1500,
                              num_chains=2, progress_bar=False)
    beta_mean_b, pd_b, ci_lo_b, ci_hi_b = posterior_summary(mcmc_b)
    order_b = np.argsort(-np.abs(beta_mean_b))[:6]
    for i in order_b:
        print(f"  idx={i:2d}  beta={beta_mean_b[i]:+.3f}  pd={pd_b[i]:.3f}  CI=[{ci_lo_b[i]:.3f},{ci_hi_b[i]:.3f}]  {'<-- TRUE' if i in true_idx else ''}")
