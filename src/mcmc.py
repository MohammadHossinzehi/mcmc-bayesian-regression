"""Metropolis-Hastings MCMC sampler implemented from scratch (NumPy only).

Given an unnormalized log-probability function for a target distribution,
this module draws approximate samples from that distribution using a
symmetric Gaussian random-walk proposal. It requires no gradient information
and works for a target of any dimensionality, using only ratios of the
target density.
"""

from __future__ import annotations

import numpy as np


def metropolis_hastings(
    log_prob_fn,
    initial,
    n_samples,
    proposal_std=0.1,
    burn_in=0,
    rng=None,
):
    """Run a Metropolis-Hastings sampler.

    Parameters
    ----------
    log_prob_fn : callable
        Function mapping a 1D numpy array of parameters to the
        (unnormalized) log-probability of the target distribution at that
        point. Only differences of log_prob_fn matter, so it need not be
        normalized.
    initial : array-like
        Starting point of the chain, shape (d,).
    n_samples : int
        Number of post-burn-in samples to keep.
    proposal_std : float or array-like
        Standard deviation of the isotropic Gaussian proposal. Since the
        proposal is symmetric, its density cancels out of the acceptance
        ratio (this is what makes it "Metropolis" rather than the more
        general "Metropolis-Hastings" case).
    burn_in : int
        Number of initial iterations to discard before collecting samples.
    rng : numpy.random.Generator, optional
        Source of randomness. A fresh default_rng() is created if omitted.

    Returns
    -------
    samples : numpy.ndarray, shape (n_samples, d)
    acceptance_rate : float
        Fraction of proposed moves (across burn-in + sampling) that were
        accepted. Useful for tuning proposal_std: too high wastes almost
        every proposal, too low creeps rather than mixes.
    """
    if rng is None:
        rng = np.random.default_rng()

    current = np.atleast_1d(np.array(initial, dtype=float))
    dim = current.shape[0]
    current_log_prob = log_prob_fn(current)

    total_iters = burn_in + n_samples
    samples = np.empty((n_samples, dim))
    n_accepted = 0

    for i in range(total_iters):
        proposal = current + rng.normal(0.0, proposal_std, size=dim)
        proposal_log_prob = log_prob_fn(proposal)

        log_accept_ratio = proposal_log_prob - current_log_prob
        if np.log(rng.uniform()) < log_accept_ratio:
            current = proposal
            current_log_prob = proposal_log_prob
            n_accepted += 1

        if i >= burn_in:
            samples[i - burn_in] = current

    acceptance_rate = n_accepted / total_iters
    return samples, acceptance_rate
