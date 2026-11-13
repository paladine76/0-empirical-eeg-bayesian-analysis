"""Non-twin and twin horseshoe: does it run, recover obvious signal, and are
the reported summaries internally consistent?"""
import numpy as np
import pytest

from src.bayesian_shrinkage_analyses import ssvs_horseshoe as hs
from src.bayesian_shrinkage_analyses import ssvs_horseshoe_twin as hst
from conftest import MCMC_KW


@pytest.fixture(scope="module")
def fit(synth):
    return hs.fit_horseshoe(synth["X"], synth["y"], family="normal", seed=0, **MCMC_KW)


@pytest.fixture(scope="module")
def fit_twin(synth_twin):
    mcmc, _ = hst.fit_twin_horseshoe(synth_twin["X"], synth_twin["y"],
                                     synth_twin["pair_ids"], family="normal",
                                     seed=0, **MCMC_KW)
    return mcmc


def test_p0_default_is_p_over_11(synth, fit):
    """The sparsity target drives tau0 and hence all shrinkage. If this default
    silently changes, every downstream ranking changes with it."""
    assert fit._p0 == pytest.approx(synth["p"] / 11.0)


def test_recovers_true_signal(synth, fit):
    bf = hs.inclusion_bayes_factor(fit, synth["n"])
    top2 = set(np.argsort(-bf)[:2])
    assert top2 == set(synth["true_idx"])


def test_kappa_in_unit_interval(synth, fit):
    k = hs._kappa(fit, synth["n"])
    assert k.shape[1] == synth["p"]
    assert np.all((k >= 0) & (k <= 1))


def test_conditional_mean_lies_inside_its_own_ci(synth, fit):
    """beta_mean_cond, pd and the CI must all be computed on the same
    included-draw subset; if they drift apart the mean escapes its interval."""
    bmean, pd_, lo, hi, n_inc = hs.conditional_posterior_summary(fit, synth["n"])
    ok = n_inc > 0
    assert np.all(lo[ok] <= bmean[ok]) and np.all(bmean[ok] <= hi[ok])
    assert np.all((pd_[ok] >= 0.5) & (pd_[ok] <= 1.0))


def test_pip_and_bf_agree_on_ordering(synth, fit):
    """BF10 is a monotone rescaling of pseudo-PIP within a fit -- same ranking."""
    pip, _ = hs.pseudo_inclusion_prob(fit, synth["n"])
    bf = hs.inclusion_bayes_factor(fit, synth["n"])
    assert list(np.argsort(-pip)) == list(np.argsort(-bf))


def test_bernoulli_family_runs(synth):
    yb = (synth["y"] > 0).astype(float)
    mcmc = hs.fit_horseshoe(synth["X"], yb, family="bernoulli", seed=0, **MCMC_KW)
    bf = hs.inclusion_bayes_factor(mcmc, synth["n"])
    assert np.all(np.isfinite(bf))


def test_invalid_family_rejected(synth):
    with pytest.raises(Exception):
        hs.fit_horseshoe(synth["X"], synth["y"], family="poisson", seed=0, **MCMC_KW)


# ---- twin ----

def test_twin_recovers_true_signal(synth_twin, fit_twin):
    bf = hst.inclusion_bayes_factor(fit_twin, synth_twin["n"])
    assert set(np.argsort(-bf)[:2]) == set(synth_twin["true_idx"])


def test_twin_conditional_mean_inside_ci(synth_twin, fit_twin):
    bmean, pd_, lo, hi, n_inc = hst.conditional_posterior_summary(fit_twin, synth_twin["n"])
    ok = n_inc > 0
    assert np.all(lo[ok] <= bmean[ok]) and np.all(bmean[ok] <= hi[ok])


def test_twin_icc_recovers_nonzero_pair_variance(fit_twin):
    """Data were generated with a real pair-level intercept, so ICC must be
    bounded in [0,1] and clearly above zero."""
    icc = hst.icc_summary(fit_twin, family="normal")
    assert 0.0 <= icc["ci5"] <= icc["mean"] <= icc["ci95"] <= 1.0
    assert icc["mean"] > 0.1
