"""Lognormal likelihood with record-specific left censoring.

All math in this module operates on Python floats at a single boundary.
Input concentrations are positive Decimals converted by the caller.

Parameterisation used for optimisation is ``(mu, eta)`` with
``eta = 1 / sigma > 0``, where ``mu`` and ``sigma`` are the parameters of
the underlying normal distribution of ``log(concentration)``.
"""

import math

LN_2PI = math.log(2.0 * math.pi)
SQRT2 = math.sqrt(2.0)


def log_standard_normal_pdf(z):
    return -0.5 * (z * z + LN_2PI)


def _log_survival_large(x):
    """log Q(x) for large positive x via the Mills asymptotic series."""
    x_squared = x * x
    total = 1.0
    term = 1.0
    previous_magnitude = 1.0
    # Alternating asymptotic series; best truncation is at the smallest term.
    for k in range(1, 101):
        term *= -(2.0 * k - 1.0) / x_squared
        magnitude = abs(term)
        if magnitude >= previous_magnitude:
            break
        total += term
        previous_magnitude = magnitude
    return -0.5 * x_squared - 0.5 * LN_2PI - math.log(x) + math.log(total)


def norm_logcdf(z):
    """Robust log of the standard normal cumulative distribution function."""
    if z >= 0.0:
        return math.log(0.5 * math.erfc(-z / SQRT2))
    x = -z
    if x <= 26.0:
        tail = math.erfc(x / SQRT2)
        if tail > 0.0:
            return math.log(0.5 * tail)
    return _log_survival_large(x)


def norm_cdf(z):
    return 0.5 * math.erfc(-z / SQRT2)


def mills_ratio(z):
    """Inverse Mills ratio lambda(z) = phi(z) / Phi(z), robust in both tails."""
    if z >= -10.0:
        return math.exp(log_standard_normal_pdf(z)) / norm_cdf(z)
    return math.exp(log_standard_normal_pdf(z) - norm_logcdf(z))


def exact_contribution(log_value, mu, eta):
    """Log density of a lognormal observation at an exact positive value."""
    w = (log_value - mu) * eta
    return math.log(eta) - log_value - 0.5 * (w * w + LN_2PI)


def censored_contribution(log_limit, mu, eta):
    """Log CDF mass below the censoring limit for a left-censored record."""
    return norm_logcdf((log_limit - mu) * eta)


def record_contribution(record, mu, eta):
    if record.exact:
        return exact_contribution(record.log_value, mu, eta)
    return censored_contribution(record.log_limit, mu, eta)


def total_log_likelihood(records, mu, eta):
    total = 0.0
    for record in records:
        total += record_contribution(record, mu, eta)
    return total


def gradient_and_hessian(records, mu, eta):
    """Analytic gradient and observed Hessian of the log likelihood.

    Returns ``(gradient, hessian)`` where each is a ``(mu, eta)`` pair.
    """
    g_mu = 0.0
    g_eta = 0.0
    h_mu_mu = 0.0
    h_mu_eta = 0.0
    h_eta_eta = 0.0
    for record in records:
        if record.exact:
            y = record.log_value
            w = (y - mu) * eta
            g_mu += w * eta
            g_eta += (1.0 - w * w) / eta
            h_mu_mu -= eta * eta
            h_mu_eta += 2.0 * w
            h_eta_eta -= (1.0 + w * w) / (eta * eta)
        else:
            z = (record.log_limit - mu) * eta
            lam = mills_ratio(z)
            g_mu -= eta * lam
            g_eta += z * lam / eta
            delta = lam * (z + lam)
            h_mu_mu -= eta * eta * delta
            h_mu_eta += lam * (z * z + z * lam - 1.0)
            h_eta_eta += z * lam * (1.0 - z * z - z * lam) / (eta * eta)
    return (g_mu, g_eta), (h_mu_mu, h_mu_eta, h_eta_eta)
