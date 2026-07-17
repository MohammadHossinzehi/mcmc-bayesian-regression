"""Diagnostics for judging whether an MCMC chain has mixed well."""

from __future__ import annotations

import numpy as np


def autocorrelation(chain, max_lag=50):
    """Sample autocorrelation of a 1D chain at lags 0..max_lag.

    A fast-decaying autocorrelation indicates the chain is exploring the
    target distribution quickly (good mixing); slow decay means consecutive
    samples are highly dependent and more iterations are needed per
    effectively independent sample.
    """
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    centered = chain - chain.mean()
    var = np.dot(centered, centered) / n
    if var == 0:
        return np.zeros(max_lag + 1)

    acf = np.empty(max_lag + 1)
    for lag in range(max_lag + 1):
        cov = np.dot(centered[: n - lag], centered[lag:]) / n
        acf[lag] = cov / var
    return acf


def effective_sample_size(chain, max_lag=50):
    """Estimate the effective sample size (ESS) of a 1D chain using the
    initial positive sequence of the autocorrelation function:

        ESS = n / (1 + 2 * sum_{lag>=1} rho_lag)

    summing only while rho stays positive, which is the standard cutoff
    used to keep the estimate stable (Geyer's initial positive sequence).
    """
    chain = np.asarray(chain, dtype=float)
    n = len(chain)
    acf = autocorrelation(chain, max_lag=max_lag)

    total = 0.0
    for rho in acf[1:]:
        if rho <= 0:
            break
        total += rho

    denom = 1.0 + 2.0 * total
    return n / denom if denom > 0 else float(n)
