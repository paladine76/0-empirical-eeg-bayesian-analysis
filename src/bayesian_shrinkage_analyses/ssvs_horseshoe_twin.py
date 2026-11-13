"""
Twin-aware regularized horseshoe regression (Piironen & Vehtari 2017), adding a 
pair-level random intercept to horseshoe_ssvs.py for twin-pair data (e.g. batss2). 
NUTS-native, no discrete/mixture component.
 
  eta_i = alpha + X_i @ beta + u_{pair(i)}  (+ Z_i @ gamma if Z given)
  u_g ~ Normal(0, sigma_pair)                    -- pair-level random intercept
  beta_j = tau * lambda_tilde_j * z_j,  z_j ~ Normal(0,1)
  lambda_j ~ HalfCauchy(1);  tau ~ HalfCauchy(tau0);  c2 ~ InverseGamma(nu/2, nu/2*slab_scale^2)
  tau0 = (p0/(p-p0)) * (sigma_guess/sqrt(n))
  normal:    y_i ~ Normal(eta_i, sigma_within)
  bernoulli: y_i ~ Bernoulli(logits=eta_i)   [ICC uses fixed logistic variance pi^2/3]
 
pd is conditional on inclusion via the shrinkage factor kappa_j (same definition as 
horseshoe_ssvs.py), matching the spike-and-slab and BoomSpikeSlab conditional-pd convention.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd

numpyro.set_host_device_count(6)


def _make_model(family, p0, sigma_guess, G, nu=4.0, slab_scale=2.0):
    def model(X, pair_idx, Z=None, y=None):
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

def fit_twin_horseshoe(X, y, pair_ids, Z=None, family="normal", p0=None, sigma_guess=None,
                       seed=0, num_warmup=2000, num_samples=3000, num_chains=4, 
                       target_access_prob=0.95, progress_bar=True):
    """
    X: (n,p) predictors, ASSUMED ALREADY STANDARDIZED (zero mean, unit
    variance per column) -- not done internally; standardize before calling.
    y: (n,) outcome, similarly assumed already standardized for
    family="normal" (0/1 for family="bernoulli", no standardization needed).
    pair_ids: length-n twin-pair ids (singletons and complete pairs both
    fine). Z: optional forced-in covariates (standardize these too if
    comparability of gamma across covariates matters).
    p0: prior target number of "relevant" predictors (default p/20, the
        settled sparsity level for this project).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Z_arr = None if Z is None else jnp.array(np.asarray(Z, dtype=float))
    pair_codes, uniques = pd.factorize(pd.Series(pair_ids))
    G = len(uniques)
    p = X.shape[1]
    if p0 is None:
        p0 = p / 11.0
    
    if sigma_guess is None:
        sigma_guess = 1.0
    
    model = _make_model(family, p0, sigma_guess, G)
    kernel = NUTS(model, target_accept_prob=target_access_prob)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                progress_bar=progress_bar)
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.array(X), pair_idx=jnp.array(pair_codes), 
             Z=Z_arr, y=jnp.array(y))
    mcmc._p0 = p0

    return mcmc, uniques

def _kappa(mcmc, n):
    samples = mcmc.get_samples()
    tau = np.array(samples["tau"])[:, None]
    lam_tilde = np.array(samples["lam_tilde"])

    return 1.0 / (1.0 + n * tau**2 * lam_tilde**2)

def pseudo_inclusion_prob(mcmc, n):
    """
    pseudo-PIP_j = P(kappa_j < 0.5 | data). See horseshoe_ssvs.py for definition.
    """
    kappa = _kappa(mcmc, n)
    beta = np.array(mcmc.get_samples()["beta"])
    pip = (kappa < 0.5).mean(axis=0)
    beta_mean = beta.mean(axis=0)

    return pip, beta_mean

def conditional_posterior_summary(mcmc, n, level=0.95):
    """
    Per-predictor conditional mean, pd, and (level*100)% credible interval,
    ALL conditional on inclusion (kappa_j < 0.5), computed over the same
    included-draw subset in one pass -- consistent estimand throughout (do
    not compare beta_mean_conditional against the unconditional mean from
    pseudo_inclusion_prob(), which is shrinkage-weighted and describes a
    different quantity). pd_j = max(P(beta_j>0|included), P(beta_j<0|included)),
    range 0.5-1.0. Matches the conditional-on-inclusion convention used
    across script_twin_univariate_ssvs.py and BoomSpikeSlab (R). NaN (all
    four) if zero included draws.
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

def icc_summary(mcmc, family="normal"):
    """
    ICC = sigma_pair^2 / (sigma_pair^2 + sigma_within^2). For
    family='bernoulli', sigma_within^2 is fixed at pi^2/3 (logistic latent
    variance) since there is no free residual-scale parameter.
    """
    samples = mcmc.get_samples()
    sp = np.array(samples["sigma_pair"])**2
    if family == "normal":
        sw = np.array(samples["sigma_within"])**2
    elif family == "bernoulli":
        sw = (np.pi**2) / 3.0
    else:
        raise ValueError("family must be 'normal' or 'bernoulli'")
    
    icc = sp / (sp + sw)

    return dict(mean=float(icc.mean()), ci5=float(np.percentile(icc, 5.5)), 
                ci95=float(np.percentile(icc, 94.5)))

def prior_inclusion_baseline(n, p, p0, sigma_guess=1.0, nu=4.0, slab_scale=2.0, S=20000,
                             seed=0):
    """
    P(kappa < 0.5) under the PRIOR ALONE, via prior-predictive simulation --
    baseline for the inclusion Bayes-factor analog. See horseshoe_ssvs.py.
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
    / [p0_prior/(1-p0_prior)], where p0_prior = prior_inclusion_baseline(...).
    p0 is read from mcmc._p0 (stashed by fit_twin_horseshoe). See horseshoe_ssvs.py.
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
    G, p = 150, 20
    true_idx = [4, 12]
    pair_ids, X_rows = [], []
    for g in range(G):
        n_members = 2 if rng.random() < 0.8 else 1
        for _ in range(n_members):
            pair_ids.append(g) 
            X_rows.append(rng.normal(size=p))

    X = np.array(X_rows); pair_ids = np.array(pair_ids); 
    n = X.shape[0]
    b_true = np.zeros(p); b_true[true_idx] = [0.6, -0.5]
    sigma_pair_true, sigma_within_true = 0.6, 0.8
    u_true = rng.normal(0, sigma_pair_true, size=G)

    y = X @ b_true + u_true[pair_ids] + rng.normal(0, sigma_within_true, size=n)
    y = (y - y.mean()) / y.std()
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)

    print("Fitting twin-aware horseshoe on demo data (true signals at v4, v12) ...")
    mcmc, _ = fit_twin_horseshoe(X, y, pair_ids, family="normal", 
                                 num_warmup=1000, num_samples=1500, num_chains=2)
    pip, beta_mean = pseudo_inclusion_prob(mcmc, n)
    beta_cond, pd_, ci_lo, ci_hi, n_included = conditional_posterior_summary(mcmc, n)
    icc = icc_summary(mcmc, family="normal")
    print(f"ICC = {icc['mean']:.3f} (true = 0.360)")
    order = np.argsort(-pip)[:10]
    for i in order:
        print(f"  idx={i:2d}  pseudo_PIP={pip[i]:.3f}  beta_cond={beta_cond[i]:.3f}  pd={pd_[i]:.3f}  CI=[{ci_lo[i]:.3f},{ci_hi[i]:.3f}]  {'<-- TRUE' if i in true_idx else ''}")