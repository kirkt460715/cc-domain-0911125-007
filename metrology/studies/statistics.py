"""Exact-only descriptive statistics and batch-level bootstrap."""

import math
import random
from decimal import Decimal, localcontext
from ..values import decimal_text


def _sorted_decimals(values):
    return sorted(values)


def _empirical_percentile(sorted_values, probability):
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] + fraction * (
        sorted_values[high] - sorted_values[low]
    )


def exact_only_summary(records):
    """Descriptive statistics using exact observations only.

    This estimates the distribution *conditional on detection above the
    historical limits*, which is a different estimand from the censored
    maximum-likelihood estimate of the whole population.
    """
    exact = [record for record in records if record.exact]
    if not exact:
        return {
            "estimand": "distribution conditional on exact detection; excludes "
            "every left-censored record",
            "count": 0,
            "arithmetic_mean": None,
            "sample_standard_deviation": None,
            "median": None,
            "geometric_mean": None,
            "geometric_standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "quartiles": {"q1": None, "median": None, "q3": None},
            "note": "no exact observations available",
        }
    values = _sorted_decimals(record.value for record in exact)
    count = len(values)
    with localcontext() as ctx:
        ctx.prec = 40
        mean = sum(values, Decimal(0)) / Decimal(count)
        if count >= 2:
            variance = sum(
                ((value - mean) ** 2 for value in values), Decimal(0)
            ) / Decimal(count - 1)
            sample_sd = variance.sqrt()
        else:
            sample_sd = None
        logs = [Decimal(repr(math.log(float(value)))) for value in values]
        log_mean = sum(logs, Decimal(0)) / Decimal(count)
        geometric_mean = log_mean.exp()
        if count >= 2:
            log_variance = sum(
                ((value - log_mean) ** 2 for value in logs), Decimal(0)
            ) / Decimal(count - 1)
            geometric_sd = log_variance.sqrt().exp()
        else:
            geometric_sd = None
    quartiles = {
        "q1": decimal_text(_empirical_percentile(values, Decimal(".25"))),
        "median": decimal_text(_empirical_percentile(values, Decimal(".5"))),
        "q3": decimal_text(_empirical_percentile(values, Decimal(".75"))),
    }
    return {
        "estimand": "distribution conditional on exact detection; excludes every "
        "left-censored record",
        "count": count,
        "arithmetic_mean": decimal_text(mean),
        "sample_standard_deviation": (
            decimal_text(sample_sd) if sample_sd is not None else None
        ),
        "median": quartiles["median"],
        "geometric_mean": decimal_text(geometric_mean),
        "geometric_standard_deviation": (
            decimal_text(geometric_sd) if geometric_sd is not None else None
        ),
        "minimum": decimal_text(values[0]),
        "maximum": decimal_text(values[-1]),
        "quartiles": quartiles,
        "record_ids": [record.record_id for record in exact],
        "note": "censored records are not zero-filled; they are absent from this "
        "summary by construction",
    }


def _float_percentile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def batch_bootstrap(records, replicates, seed, confidence_level, fit_callback):
    """Resample whole sampling batches and refit on each replicate.

    Failed or unidentifiable rounds are counted honestly rather than
    silently replaced.
    """
    batches = {}
    for record in records:
        batches.setdefault(record.sampling_batch, []).append(record)
    batch_names = sorted(batches)
    rng = random.Random(seed)
    successes = []
    failures = 0
    failure_reasons = {}
    for _ in range(replicates):
        resampled = []
        for _ in range(len(batch_names)):
            chosen = rng.choice(batch_names)
            resampled.extend(batches[chosen])
        result = fit_callback(resampled)
        if result is None:
            failures += 1
            reason = "unidentifiable_resample"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        if not result.get("converged", False):
            failures += 1
            reason = result.get("termination_reason", "optimisation_failure")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        successes.append(result)
    output = {
        "method": "sampling batches resampled as intact clusters; all records of a "
        "drawn batch stay together",
        "replicates": replicates,
        "batches": batch_names,
        "successful_replicates": len(successes),
        "failed_replicates": failures,
        "failure_reasons": failure_reasons,
        "single_batch_warning": (
            "only one sampling batch exists; cluster bootstrap cannot reflect "
            "between-batch variation"
            if len(batch_names) == 1
            else None
        ),
    }
    if not successes:
        output["confidence_intervals"] = None
        return output
    alpha = 1.0 - float(confidence_level)
    interval = {}
    for key in ("median", "geometric_sd", "arithmetic_mean", "probability_above"):
        series = [item[key] for item in successes if item[key] is not None]
        if not series:
            interval[key] = None
            continue
        interval[key] = {
            "lower": repr(_float_percentile(series, alpha / 2.0)),
            "upper": repr(_float_percentile(series, 1.0 - alpha / 2.0)),
        }
    output["confidence_intervals"] = interval
    return output
