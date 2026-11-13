"""Synthetic cases with one shared true distribution and different limits.

These cases demonstrate the motivating problem: when the same underlying
lognormal population is observed through a high detection limit, naive
replacement of non-detects by zero (or by half the limit) biases the
arithmetic mean downward. The censored likelihood is invariant to that
substitution because non-detects enter through CDF mass alone.
"""

import math
import random
from ..values import integer, number


def lognormal_sample(mu, sigma, count, seed):
    """Draw exact latent concentrations from a lognormal distribution."""
    rng = random.Random(seed)
    return [math.exp(rng.gauss(mu, sigma)) for _ in range(count)]


def observe_with_limit(latent_values, detection_limit):
    """Turn latent values into exact or left-censored study records."""
    records = []
    for index, value in enumerate(latent_values):
        if value >= detection_limit:
            records.append(
                {
                    "record_id": f"s{index:04d}",
                    "result": {"kind": "exact", "value": repr(value)},
                }
            )
        else:
            records.append(
                {
                    "record_id": f"s{index:04d}",
                    "result": {"kind": "left_censored"},
                }
            )
    return records


def _wrap(records, detection_limit, unit, instrument, certificate, seed_tag):
    return [
        {
            "record_id": f"{seed_tag}-{item['record_id']}",
            "specimen": f"{seed_tag}-{item['record_id']}",
            "measurement_item": "trace-metal",
            "instrument": instrument,
            "certificate_revision": certificate,
            "sampling_batch": f"batch-{index % 4}",
            "detection_limit": repr(detection_limit),
            "unit": unit,
            "result": (
                {"kind": "exact", "value": item["result"]["value"]}
                if item["result"]["kind"] == "exact"
                else {"kind": "left_censored"}
            ),
        }
        for index, item in enumerate(records)
    ]


def build_synthetic_case(
    *,
    mu=0.0,
    sigma=0.5,
    count=400,
    seed=20261113,
    high_limit=1.2,
    low_limit=0.4,
    unit="mg_per_L",
):
    """Two instruments observe the *same* latent lognormal population."""
    mu = float(number(mu))
    sigma = float(number(sigma, positive=True))
    count = integer(count, "count", 4, 100000)
    high_limit = float(number(high_limit, positive=True))
    low_limit = float(number(low_limit, positive=True))
    if low_limit >= high_limit:
        from ..errors import InputError

        raise InputError("low_limit must be strictly below high_limit")
    latent = lognormal_sample(mu, sigma, count, seed)
    high_records = observe_with_limit(latent, high_limit)
    low_records = observe_with_limit(latent, low_limit)
    common = {
        "target_unit": unit,
        "population": "synthetic pre-registered lognormal population",
        "threshold": "2",
        "confidence_level": ".95",
        "bootstrap_replicates": 200,
        "bootstrap_seed": seed,
        "duplicate_policy": "exclude_specimen",
        "units": [
            {"symbol": unit, "dimension": "concentration", "scale": "1", "offset": "0"}
        ],
    }
    before = {
        **common,
        "study_id": "synthetic-before",
        "records": _wrap(
            high_records, high_limit, unit, "old-instrument", "cert-old", "before"
        ),
    }
    after = {
        **common,
        "study_id": "synthetic-after",
        "records": _wrap(
            low_records, low_limit, unit, "new-instrument", "cert-new", "after"
        ),
    }
    truth = {
        "log_mu": repr(mu),
        "log_sigma": repr(sigma),
        "median": repr(math.exp(mu)),
        "geometric_standard_deviation": repr(math.exp(sigma)),
        "arithmetic_mean": repr(math.exp(mu + 0.5 * sigma * sigma)),
        "latent_values": [repr(value) for value in latent],
    }
    return {
        "before": before,
        "after": after,
        "truth": truth,
        "detection_limits": {"high": repr(high_limit), "low": repr(low_limit)},
    }


def substitution_bias(latent_values, detection_limit, factor):
    """Naive mean obtained by replacing each non-detect with factor * limit.

    ``factor=0`` reproduces the zero substitution that made the old
    instrument appear to lower the mean; ``factor=0.5`` reproduces the
    half-limit convention. Returned purely as a documented illustration of
    the bias; the study itself never performs either substitution.
    """
    replaced = [
        value if value >= detection_limit else factor * detection_limit
        for value in latent_values
    ]
    return sum(replaced) / len(replaced)
