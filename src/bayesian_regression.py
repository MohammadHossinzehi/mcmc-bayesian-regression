"""Bayesian linear regression: synthetic data, log-posterior, and the
closed-form (conjugate) posterior used to check the MCMC sampler.

Model
-----
y_i = w * x_i + b + eps_i,   eps_i ~ Normal(0, sigma^2)

with a Normal(0, tau^2) prior independently on w and b, and known noise
variance sigma^2. Because the likelihood is Gaussian and the prior is
Gaussian, the posterior over (w, b) is also Gaussian and has a closed form,
which is what makes this a good sanity check for a general purpose sampler:
if Metropolis-Hastings didn't work, its estimated posterior mean/covariance
would disagree with the exact answer.
"""

from __future__ import annotations

import numpy as np


def generate_data(n=60, true_w=2.5, true_b=-1.0, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    noise = rng.normal(0.0, sigma, size=n)
    y = true_w * x + true_b + noise
    return x, y


def make_log_posterior(x, y, sigma=1.0, tau=10.0):
    """Return a function computing log p(w, b | x, y) up to a constant.

    theta is [w, b].
    """
    x = np.asarray(x)
    y = np.asarray(y)

    def log_posterior(theta):
        w, b = theta
        residuals = y - (w * x + b)
        log_likelihood = -0.5 * np.sum(residuals ** 2) / sigma ** 2
        log_prior = -0.5 * (w ** 2 + b ** 2) / tau ** 2
        return log_likelihood + log_prior

    return log_posterior


def closed_form_posterior(x, y, sigma=1.0, tau=10.0):
    """Exact posterior mean and covariance for (w, b) under the conjugate
    Gaussian model, via the standard Bayesian linear regression formula:

        Sigma_post = (X^T X / sigma^2 + I / tau^2)^-1
        mu_post    = Sigma_post @ X^T y / sigma^2

    where X is the design matrix [x_i, 1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    design = np.stack([x, np.ones_like(x)], axis=1)

    precision = design.T @ design / sigma ** 2 + np.eye(2) / tau ** 2
    cov_post = np.linalg.inv(precision)
    mean_post = cov_post @ design.T @ y / sigma ** 2
    return mean_post, cov_post
