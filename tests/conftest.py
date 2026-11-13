"""Shared fixtures: one small synthetic dataset with known signal, reused
across tests so no test refits the sampler more than it needs to."""
import numpy as np
import pytest


@pytest.fixture(scope="session")
def synth():
    """n=60, p=12, two true signals at indices 2 and 7."""
    rng = np.random.default_rng(0)
    n, p = 60, 12
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[2, 7]] = [1.2, -1.0]
    y = X @ beta + rng.normal(0, 0.6, size=n)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std()
    return dict(X=Xs, y=ys, n=n, p=p, true_idx=[2, 7],
                names=[f"v{i}" for i in range(p)])


@pytest.fixture(scope="session")
def synth_twin():
    """60 pairs (n=120), p=12, two true signals, pair-level random intercept."""
    rng = np.random.default_rng(1)
    G, p = 60, 12
    pair_ids = np.repeat(np.arange(G), 2)
    n = len(pair_ids)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[2, 7]] = [1.2, -1.0]
    u = rng.normal(0, 0.7, size=G)
    y = X @ beta + u[pair_ids] + rng.normal(0, 0.6, size=n)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ys = (y - y.mean()) / y.std()
    return dict(X=Xs, y=ys, pair_ids=pair_ids, n=n, p=p,
                true_idx=[2, 7], names=[f"v{i}" for i in range(p)])


# small sampler budget -- these tests check correctness of plumbing and
# recovery of an obvious signal, not posterior precision
MCMC_KW = dict(num_warmup=300, num_samples=400, num_chains=1, progress_bar=False)
