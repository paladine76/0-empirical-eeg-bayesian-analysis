"""
Regularized horseshoe regression (Piironen & Vehtari 2017) for single-cohort (non-twin) 
data, e.g. ease2 / M1r. NUTS-native alternative to the spike-and-slab scripts: 
continuous global-local shrinkage prior, no discrete/mixture component, so it samples more 
naturally under NUTS.
 
  eta_i = alpha + X_i @ beta  (+ Z_i @ gamma if Z given)
  beta_j = tau * lambda_tilde_j * z_j,  z_j ~ Normal(0,1)   [non-centered]
  lambda_j ~ HalfCauchy(1)                                    [local shrinkage]
  tau ~ HalfCauchy(tau0)                                      [global shrinkage]
  c2 ~ InverseGamma(nu/2, nu/2 * slab_scale^2)                [regularizes large beta]
  lambda_tilde_j^2 = c2 * lambda_j^2 / (c2 + tau^2 * lambda_j^2)
  tau0 = (p0/(p-p0)) * (sigma_guess/sqrt(n))                  [targets p0 nonzero]
  normal:    y_i ~ Normal(eta_i, sigma)
  bernoulli: y_i ~ Bernoulli(logits=eta_i)
 
No exact zeros (unlike spike-and-slab), so "inclusion" and pd use the per-draw shrinkage 
factor kappa_j = 1/(1 + n*tau^2*lambda_tilde_j^2): kappa->0 = effectively unshrunk 
("included"), kappa->1 = fully shrunk.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(6)


def _make_model(family, p0, sigma_guess, nu=4.0, slab_scale=2.0):
    def model(X, Z=None, y=None):
        n, p = X.shape
        tau0 = (p0 / (p - p0)) * (sigma_guess / jnp.sqrt(n))
        
        tau = numpyro.sample("tau", dist.HalfCauchy(tau0))
        with numpyro.plate("local", p):
            lam = numpyro.sample("lam", dist.HalfCauchy(1.0))

        c2 = numpyro.sample("c2", dist.InverseGamma(nu / 2.0, nu / 2.0 * slab_scale**2))
        lam_tilde = jnp.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
        with numpyro.plate("predictors", p):
            z = numpyro.sample("z", dist.Normal(0.0, 1.0))

        beta = numpyro.deterministic("beta", z * lam_tilde * tau)
        numpyro.deterministic("lam_tilde", lam_tilde)
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 5.0))
        eta = alpha + X @ beta

        if Z is not None:
            q = Z.shape[1]
            with numpyro.plate("covariates", q):
                gamma = numpyro.sample("gamma", dist.Normal(0.0, 5.0))
            
            eta = eta + Z @ gamma
        
        if family == "normal":
            sigma = numpyro.sample("sigma", dist.HalfNormal(2.0 * sigma_guess))
            numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
        elif family == "bernoulli":
            numpyro.sample("obs", dist.Bernoulli(logits=eta), obs=y)
        else:
            raise ValueError("family must be 'normal' or 'bernoulli'")
        
    return model
    
def fit_horseshoe(X, y, Z=None, family="normal", p0=None, sigma_guess=None, seed=0,
                  num_warmup=2000, num_samples=3000, num_chains=4, target_accept_prob=0.95,
                  progress_bar=True):
    """
    X: (n,p) predictors, ASSUMED ALREADY STANDARDIZED (zero mean, unit
    variance per column) -- not done internally; standardize before calling.
    y: (n,) outcome, similarly assumed already standardized for
    family="normal" (0/1 for family="bernoulli", no standardization needed).
    Z: optional (n,q) forced-in, unshrunk covariates (not under the
    horseshoe prior; standardize these too if comparability of gamma across
    covariates matters).
    p0: prior target number of "relevant" predictors (default p/20, the
        settled sparsity level for this project).
    sigma_guess: prior noise-scale guess used in the tau0 calibration
        (default 1.0 for standardized/normal y, or 1.0 as a standard
        logistic-scale approximation for family="bernoulli").
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z_arr = None if Z is None else jnp.array(np.asarray(Z, dtype=float))

    p = X.shape[1]
    if p0 is None:
        p0 = p / 11.0
    
    if sigma_guess is None:
        sigma_guess = 1.0

    model = _make_model(family, p0, sigma_guess)
    kernel = NUTS(model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.array(X), Z=Z_arr, y=jnp.array(y))
    mcmc._p0 = p0
    
    return mcmc

def _kappa(mcmc, n):
    samples = mcmc.get_samples()
    tau = np.array(samples["tau"])[:, None]
    lam_tilde = np.array(samples["lam_tilde"])

    return 1.0 / (1.0 + n * tau**2 * lam_tilde**2)

def pseudo_inclusion_prob(mcmc, n):
    """
    pseudo-PIP_j = P(kappa_j < 0.5 | data): fraction of draws where predictor
    j is effectively unshrunk. Not a true posterior model-inclusion
    probability (horseshoe has no discrete indicator) -- a shrinkage-based
    analogue, comparable in spirit to spike-and-slab PIP.
    """
    kappa = _kappa(mcmc, n)
    beta = np.array(mcmc.get_samples()["beta"])
    pip = (kappa < 0.5).mean(axis=0)
    beta_mean = beta.mean(axis=0)

    return pip, beta_mean

def conditional_posterior_summary(mcmc, n, level=0.95):
    """
    Per-predictor conditional mean, pd, and (level*100)% credible interval,
    ALL CONDITIONAL ON INCLUSION (kappa_j < 0.5 for that draw -- effectively
    unshrunk), computed over the same included-draw subset in one pass --
    consistent estimand throughout (do not compare beta_mean_conditional
    against the unconditional mean from pseudo_inclusion_prob(), which is
    shrinkage-weighted and describes a different quantity).
    pd_j = max(P(beta_j>0 | included), P(beta_j<0 | included)). Range 0.5-1.0.
    Matches the conditional-on-inclusion definition used for spike-and-slab
    (script_univariate_ssvs.py) and BoomSpikeSlab (R), so pd is comparable
    across all three. NaN (all four) if zero included draws.
    Returns (beta_mean_conditional, pd, ci_lower, ci_upper, n_included).
    """
    kappa = _kappa(mcmc, n)
    beta = np.array(mcmc.get_samples()["beta"])
    included = kappa < 0.5
    p = beta.shape[1]
    beta_mean_cond = np.full(p, np.nan)
    pd_ = np.full(p, np.nan)
    lo = np.full(p, np.nan)
    hi = np.full(p, np.nan)
    n_included = included.sum(axis=0)
    a = (1 - level) / 2 * 100
    for j in range(p):
        if n_included[j] == 0:
            continue

        b_inc = beta[included[:, j], j]
        beta_mean_cond[j] = b_inc.mean()
        p_pos = (b_inc > 0).mean()
        pd_[j] = max(p_pos, 1.0 - p_pos)
        lo[j], hi[j] = np.percentile(b_inc, [a, 100 - a])

    return beta_mean_cond, pd_, lo, hi, n_included

def prior_inclusion_baseline(n, p, p0, sigma_guess=1.0, nu=4.0, slab_scale=2.0, S=20000, 
                             seed=0):
    """
    P(kappa < 0.5) under the PRIOR ALONE (no data), via prior-predictive
    simulation -- used as the baseline for the inclusion Bayes-factor
    analog, the same posterior-odds/prior-odds construction used for
    spike-and-slab's inclusion_bayes_factor().
    """
    rng = np.random.default_rng(seed)
    tau0 = (p0 / (p - p0)) * (sigma_guess / np.sqrt(n))
    tau_prior = np.abs(rng.standard_cauchy(S)) * tau0
    lam_prior = np.abs(rng.standard_cauchy(S))
    c2_prior = 1.0 / rng.gamma(nu / 2.0, 1.0 / (nu / 2.0 * slab_scale**2), size=S)
    lam_tilde_prior = np.sqrt(c2_prior * lam_prior**2 / (c2_prior + tau_prior**2 * lam_prior**2))
    kappa_prior = 1.0 / (1.0 + n * tau_prior**2 * lam_tilde_prior**2)

    return (kappa_prior < 0.5).mean()

def inclusion_bayes_factor(mcmc, n, sigma_guess=1.0, eps=1e-12):
    """
    Per-predictor inclusion Bayes factor: BF10_j = [pseudo_PIP_j/(1-pseudo_PIP_j)]
    / [p0_prior/(1-p0_prior)], where p0_prior = prior_inclusion_baseline(...)
    (P(kappa<0.5) under the prior alone). Same posterior-odds/prior-odds
    construction as script_univariate_ssvs.py's inclusion_bayes_factor().
    p0 is read from mcmc._p0 (stashed by fit_horseshoe).
    """
    p0 = mcmc._p0
    pip, _ = pseudo_inclusion_prob(mcmc, n)
    p = len(pip)
    prior_p = prior_inclusion_baseline(n, p, p0, sigma_guess=sigma_guess)
    prior_odds = prior_p / (1 - prior_p)
    post_odds = pip / np.clip(1 - pip, eps, None)
    
    return post_odds / prior_odds


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 88, 20
    X = rng.normal(size=(n, p))
    true_idx = [3, 11]
    b_true = np.zeros(p); b_true[true_idx] = [0.8, -0.7]

    y = X @ b_true + rng.normal(0, 0.9, size=n)
    y = (y - y.mean()) / y.std()
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)

    print("Fitting horseshoe on demo data (true signals at v3, v11) ...")
    mcmc = fit_horseshoe(X, y, family="normal", 
                         num_warmup=1000, num_samples=1500, num_chains=2)
    pip, beta_mean = pseudo_inclusion_prob(mcmc, n)
    beta_cond, pd_, ci_lo, ci_hi, n_included = conditional_posterior_summary(mcmc, n)
    order = np.argsort(-pip)[:10]
    for i in order:
        print(f"  idx={i:2d}  pseudo_PIP={pip[i]:.3f}  beta_cond={beta_cond[i]:.3f}  pd={pd_[i]:.3f}  CI=[{ci_lo[i]:.3f},{ci_hi[i]:.3f}]  {'<-- TRUE' if i in true_idx else ''}")