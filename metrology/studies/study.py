"""Censored concentration studies: separating capability from sample change.

First version: a pre-registered population is fit to a lognormal
distribution. Exact positive observations contribute log density;
left-censored ("below limit") observations contribute the lognormal CDF
mass below their own limit. Each record may carry a different detection
limit. Non-detects are never treated as missing, zero, or half the limit.

The study output is frozen: the original specification, every converted
record, every exclusion reason and every likelihood contribution is
embedded and fingerprinted, so re-running against newer certificates
cannot silently change a saved result.
"""

import math
from pathlib import Path
from decimal import Decimal
from ..errors import InputError, IntegrityError
from ..values import (
    canonical,
    digest,
    strict_json,
    number,
    decimal_text,
)
from .records import prepare_study
from .optimizer import fit_maximum_likelihood
from .likelihood import (
    total_log_likelihood,
    exact_contribution,
    censored_contribution,
    norm_cdf,
)
from .statistics import exact_only_summary, batch_bootstrap

SCHEMA = 1


def _estimands(mu, sigma, threshold):
    median = math.exp(mu)
    geometric_sd = math.exp(sigma)
    arithmetic_mean = math.exp(mu + 0.5 * sigma * sigma)
    probability_above = None
    if threshold is not None:
        threshold_log = math.log(float(threshold))
        probability_above = 1.0 - norm_cdf((threshold_log - mu) / sigma)
    return {
        "median": median,
        "geometric_sd": geometric_sd,
        "arithmetic_mean": arithmetic_mean,
        "probability_above": probability_above,
    }


def _identifiability(records):
    exact = [record for record in records if record.exact]
    distinct = {record.log_value for record in exact}
    if not records:
        return "no_accepted_records"
    if not exact:
        return "all_censored"
    if len(distinct) < 2:
        return "single_exact_observation"
    return "identified"


def _fit_callback(threshold):
    def callback(records):
        status = _identifiability(records)
        if status != "identified":
            return None
        fit = fit_maximum_likelihood(records)
        values = _estimands(fit["mu"], fit["sigma"], threshold)
        return {**fit, **values}

    return callback


def run_concentration_study(specification, registry=None):
    """Run the full study and return a fully traceable, fingerprinted report."""
    prepared = prepare_study(specification, registry)
    records = prepared["records"]
    status = _identifiability(records)
    exact = [record for record in records if record.exact]
    censored = [record for record in records if not record.exact]

    fit = None
    point_estimates = None
    if status == "identified":
        fit = fit_maximum_likelihood(records)
        values = _estimands(
            fit["mu"], fit["sigma"], prepared["threshold"]
        )
        point_estimates = {
            "median": decimal_text(Decimal(repr(values["median"]))),
            "geometric_standard_deviation": decimal_text(
                Decimal(repr(values["geometric_sd"]))
            ),
            "arithmetic_mean": decimal_text(
                Decimal(repr(values["arithmetic_mean"]))
            ),
            "probability_above_threshold": (
                decimal_text(Decimal(repr(values["probability_above"])))
                if values["probability_above"] is not None
                else None
            ),
            "log_mu": decimal_text(Decimal(repr(fit["mu"]))),
            "log_sigma": decimal_text(Decimal(repr(fit["sigma"]))),
        }

    bootstrap = None
    if status == "identified":
        bootstrap = batch_bootstrap(
            records,
            prepared["bootstrap_replicates"],
            prepared["bootstrap_seed"],
            float(prepared["confidence_level"]),
            _fit_callback(prepared["threshold"]),
        )

    report = {
        "schema": SCHEMA,
        "study_id": prepared["study_id"],
        "population": prepared["population"],
        "target_unit": prepared["target_unit"],
        "model": {
            "family": "lognormal",
            "exact_term": "log density of the exact positive concentration",
            "censored_term": "CDF mass below the record-specific censoring limit",
            "substitution": "none",
        },
        "threshold": (
            decimal_text(prepared["threshold"])
            if prepared["threshold"] is not None
            else None
        ),
        "confidence_level": decimal_text(prepared["confidence_level"]),
        "counts": {
            "accepted": len(records),
            "exact": len(exact),
            "left_censored": len(censored),
            "excluded": len(prepared["exclusions"]),
        },
        "identifiability": {
            "status": status,
            "unique_point_estimate": status == "identified",
            "explanation": _IDENTIFIABILITY_TEXT[status],
        },
        "fit": _fit_section(fit),
        "point_estimates": point_estimates,
        "bootstrap": bootstrap,
        "exact_only_descriptive": exact_only_summary(records),
        "records": [_record_trace(record, prepared["target_unit"]) for record in records],
        "exclusions": prepared["exclusions"],
        "handling_rules": prepared["rules"],
        "certificate_revisions": prepared["certificate_revisions"],
        "disclaimer": "this study characterises the concentration distribution and "
        "measurement capability; it does not replace existing conformity or "
        "acceptance decisions",
        "input_fingerprint": prepared["input_fingerprint"],
        "source_specification": specification,
    }
    report["report_fingerprint"] = digest(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    return report


_IDENTIFIABILITY_TEXT = {
    "identified": "at least two distinct exact observations anchor both lognormal "
    "parameters; censored records contribute CDF mass with record-specific limits",
    "all_censored": "every accepted record is left-censored; the likelihood has no "
    "unique interior maximum, so no unique mean is reported",
    "single_exact_observation": "only one distinct exact concentration is available; "
    "the log scale parameter is not separately identified, so no unique mean is "
    "reported",
    "no_accepted_records": "no record survived validation and duplicate handling",
}


def _fit_section(fit):
    if fit is None:
        return None
    return {
        "log_mu": decimal_text(Decimal(repr(fit["mu"]))),
        "log_sigma": decimal_text(Decimal(repr(fit["sigma"]))),
        "log_likelihood": decimal_text(Decimal(repr(fit["log_likelihood"]))),
        "converged": fit["converged"],
        "termination_reason": fit["termination_reason"],
        "gradient_inf_norm": (
            decimal_text(Decimal(repr(fit["gradient_inf_norm"])))
            if fit["gradient_inf_norm"] is not None
            else None
        ),
        "newton_attempts": fit["newton_attempts"],
        "nelder_mead_log_likelihood": decimal_text(
            Decimal(repr(fit["nelder_mead_log_likelihood"]))
        ),
        "optimiser_log_likelihood_gap": decimal_text(
            Decimal(repr(fit["optimiser_log_likelihood_gap"]))
        ),
        "information_eigenvalues": [
            decimal_text(Decimal(repr(value)))
            for value in fit["information_eigenvalues"]
        ],
        "boundary_solution": fit["boundary_solution"],
        "boundary_flags": fit["boundary_flags"],
        "flat_information_direction": fit["flat_information_direction"],
        "likelihood_contributions": fit["likelihood_contributions"],
    }


def _record_trace(record, target_unit):
    trace = record.as_trace()
    trace["unit"] = target_unit
    return trace


def save_study(report, destination):
    """Freeze a study report to disk; refuses to overwrite an existing file."""
    from .persistence import atomic_study_write

    return atomic_study_write(destination, canonical(report).encode("utf-8"))


def verify_study(path):
    """Reload a saved study and prove it still matches its embedded fingerprint."""
    raw = Path(path).read_bytes()
    if len(raw) > 64000000:
        raise InputError("study report exceeds supported size")
    try:
        report = strict_json(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise InputError("study report must be UTF-8") from exc
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise InputError("unsupported study schema")
    expected = report.get("report_fingerprint")
    recomputed = digest(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if expected != recomputed:
        raise IntegrityError("study report fingerprint mismatch")
    if digest(report["source_specification"]) != report["input_fingerprint"]:
        raise IntegrityError("embedded specification no longer matches its fingerprint")
    return {
        "study_id": report["study_id"],
        "report_fingerprint": expected,
        "input_fingerprint": report["input_fingerprint"],
        "identifiability": report["identifiability"]["status"],
        "accepted_records": report["counts"]["accepted"],
        "excluded_records": report["counts"]["excluded"],
        "frozen": True,
    }


def independent_likelihood_check(records, mu, sigma):
    """Recompute each likelihood term with standalone formulas.

    This deliberately bypasses the optimizer module: the log density of an
    exact observation and the CDF mass of a censored observation are built
    directly from the normal distribution, then compared term by term and
    in total against the shared likelihood implementation.
    """
    eta = 1.0 / sigma
    expected_total = 0.0
    terms = []
    for record in records:
        if record.exact:
            x = record.log_value
            standard = (x - mu) / sigma
            independent = (
                -x
                - 0.5 * math.log(2.0 * math.pi)
                - math.log(sigma)
                - 0.5 * standard * standard
            )
            shared = exact_contribution(x, mu, eta)
        else:
            standard = (record.log_limit - mu) / sigma
            independent = math.log(norm_cdf(standard))
            shared = censored_contribution(record.log_limit, mu, eta)
        expected_total += independent
        terms.append(
            {
                "record_id": record.record_id,
                "kind": record.kind,
                "independent_term": decimal_text(Decimal(repr(independent))),
                "shared_term": decimal_text(Decimal(repr(shared))),
                "absolute_difference": decimal_text(
                    Decimal(repr(abs(independent - shared)))
                ),
            }
        )
    shared_total = total_log_likelihood(records, mu, eta)
    return {
        "terms": terms,
        "independent_total": decimal_text(Decimal(repr(expected_total))),
        "shared_total": decimal_text(Decimal(repr(shared_total))),
        "absolute_difference": decimal_text(
            Decimal(repr(abs(expected_total - shared_total)))
        ),
    }


def independent_grid_check(records, grid_mu, grid_sigma):
    """Evaluate the likelihood on a caller-supplied grid without optimising.

    Used by tests as an independent small-sample cross-check of the MLE:
    the best grid cell must agree with the optimizer within the grid
    spacing.
    """
    grid_mu = [float(number(value)) for value in grid_mu]
    grid_sigma = [float(number(value, positive=True)) for value in grid_sigma]
    cells = []
    best = None
    for mu in grid_mu:
        for sigma in grid_sigma:
            value = total_log_likelihood(records, mu, 1.0 / sigma)
            cells.append(
                {
                    "mu": decimal_text(Decimal(repr(mu))),
                    "sigma": decimal_text(Decimal(repr(sigma))),
                    "log_likelihood": decimal_text(Decimal(repr(value))),
                }
            )
            if best is None or value > best["log_likelihood_float"]:
                best = {
                    "mu": mu,
                    "sigma": sigma,
                    "log_likelihood_float": value,
                }
    return {
        "cells": cells,
        "best_cell": {
            "mu": decimal_text(Decimal(repr(best["mu"]))),
            "sigma": decimal_text(Decimal(repr(best["sigma"]))),
            "log_likelihood": decimal_text(
                Decimal(repr(best["log_likelihood_float"]))
            ),
        },
    }
