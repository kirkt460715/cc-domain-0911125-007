"""Maximum likelihood estimation for the censored lognormal model.

The log likelihood is optimised in the (mu, eta=1/sigma) parameterisation,
which keeps the scale parameter strictly positive. Analytic gradients and
Hessians drive a projected Newton method; a Nelder-Mead search on
(mu, log eta) is kept as an independent fallback and as a cross-check of
the Newton optimum.
"""

import math
from .likelihood import (
    total_log_likelihood,
    gradient_and_hessian,
    record_contribution,
)

ETA_FLOOR = 1e-8
ETA_CEILING = 1e8
GRADIENT_TOLERANCE = 1e-9
MAX_NEWTON_ITERATIONS = 200


def _solve_2x2(a, b, d, rhs1, rhs2):
    """Solve [[a,b],[b,d]] x = rhs for a symmetric 2x2 matrix."""
    determinant = a * d - b * b
    if determinant == 0.0 or not math.isfinite(determinant):
        return None
    x1 = (d * rhs1 - b * rhs2) / determinant
    x2 = (-b * rhs1 + a * rhs2) / determinant
    return x1, x2


def _eigenvalues_2x2(a, b, d):
    trace = a + d
    discriminant = max(0.0, trace * trace - 4.0 * (a * d - b * b))
    root = math.sqrt(discriminant)
    return 0.5 * (trace - root), 0.5 * (trace + root)


def _project(mu, eta, mu_low, mu_high):
    mu = min(max(mu, mu_low), mu_high)
    eta = min(max(eta, ETA_FLOOR), ETA_CEILING)
    return mu, eta


def _newton_step(records, start, mu_low, mu_high):
    mu, eta = start
    current = total_log_likelihood(records, mu, eta)
    initial = (mu, eta, current)
    for iteration in range(1, MAX_NEWTON_ITERATIONS + 1):
        (g_mu, g_eta), (h_mu_mu, h_mu_eta, h_eta_eta) = gradient_and_hessian(
            records, mu, eta
        )
        gradient_norm = max(abs(g_mu), abs(g_eta))
        direction = _solve_2x2(
            h_mu_mu, h_mu_eta, h_eta_eta, -g_mu, -g_eta
        )
        if direction is None:
            direction = (g_mu, g_eta)  # steepest ascent fallback
        step = 1.0
        accepted = False
        for _ in range(80):
            candidate_mu, candidate_eta = _project(
                mu + step * direction[0],
                eta + step * direction[1],
                mu_low,
                mu_high,
            )
            candidate = total_log_likelihood(records, candidate_mu, candidate_eta)
            if math.isfinite(candidate) and candidate > current:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return {
                "converged": False,
                "iterations": iteration,
                "mu": mu,
                "eta": eta,
                "log_likelihood": current,
                "gradient_inf_norm": gradient_norm,
                "reason": "line_search_failed",
                "start": initial,
            }
        mu, eta, current = candidate_mu, candidate_eta, candidate
        (g_mu, g_eta), _ = gradient_and_hessian(records, mu, eta)
        gradient_norm = max(abs(g_mu), abs(g_eta))
        if gradient_norm < GRADIENT_TOLERANCE:
            return {
                "converged": True,
                "iterations": iteration,
                "mu": mu,
                "eta": eta,
                "log_likelihood": current,
                "gradient_inf_norm": gradient_norm,
                "reason": "gradient_tolerance",
                "start": initial,
            }
    return {
        "converged": False,
        "iterations": MAX_NEWTON_ITERATIONS,
        "mu": mu,
        "eta": eta,
        "log_likelihood": current,
        "gradient_inf_norm": gradient_norm,
        "reason": "iteration_limit",
        "start": initial,
    }


def _nelder_mead(records, start, mu_low, mu_high):
    """Direct-search maximiser on (mu, log eta), independent of gradients."""
    mu0, eta0 = start
    t0 = math.log(eta0)

    def objective(point):
        mu, log_eta = point
        eta = math.exp(min(max(log_eta, math.log(ETA_FLOOR)), math.log(ETA_CEILING)))
        mu = min(max(mu, mu_low), mu_high)
        value = total_log_likelihood(records, mu, eta)
        if not math.isfinite(value):
            value = -1e300
        return -value  # minimise negative log likelihood

    simplex = [
        [mu0, t0],
        [mu0 + 1.0, t0],
        [mu0, t0 + 0.5],
    ]
    scores = [objective(point) for point in simplex]
    for _ in range(2000):
        order = sorted(range(3), key=lambda index: scores[index])
        best, second, worst = order
        spread = max(
            abs(simplex[worst][0] - simplex[best][0]),
            abs(simplex[worst][1] - simplex[best][1]),
        )
        if spread < 1e-10:
            break
        centroid = [
            0.5 * (simplex[best][0] + simplex[second][0]),
            0.5 * (simplex[best][1] + simplex[second][1]),
        ]
        reflection = [2 * centroid[0] - simplex[worst][0],
                      2 * centroid[1] - simplex[worst][1]]
        reflection_score = objective(reflection)
        if scores[best] <= reflection_score < scores[second]:
            simplex[worst], scores[worst] = reflection, reflection_score
            continue
        if reflection_score < scores[best]:
            expansion = [
                centroid[0] + 2 * (reflection[0] - centroid[0]),
                centroid[1] + 2 * (reflection[1] - centroid[1]),
            ]
            expansion_score = objective(expansion)
            if expansion_score < reflection_score:
                simplex[worst], scores[worst] = expansion, expansion_score
            else:
                simplex[worst], scores[worst] = reflection, reflection_score
            continue
        contraction = [
            centroid[0] + 0.5 * (simplex[worst][0] - centroid[0]),
            centroid[1] + 0.5 * (simplex[worst][1] - centroid[1]),
        ]
        contraction_score = objective(contraction)
        if contraction_score < scores[worst]:
            simplex[worst], scores[worst] = contraction, contraction_score
            continue
        for index in range(3):
            if index == best:
                continue
            simplex[index] = [
                simplex[best][0] + 0.5 * (simplex[index][0] - simplex[best][0]),
                simplex[best][1] + 0.5 * (simplex[index][1] - simplex[best][1]),
            ]
            scores[index] = objective(simplex[index])
    best_index = min(range(3), key=lambda index: scores[index])
    mu, log_eta = simplex[best_index]
    eta = math.exp(min(max(log_eta, math.log(ETA_FLOOR)), math.log(ETA_CEILING)))
    return {
        "converged": True,
        "mu": min(max(mu, mu_low), mu_high),
        "eta": eta,
        "log_likelihood": -scores[best_index],
    }


def _initial_points(records):
    exact_logs = [record.log_value for record in records if record.exact]
    all_logs = exact_logs + [record.log_limit for record in records if not record.exact]
    mu0 = sum(exact_logs) / len(exact_logs) if exact_logs else sum(all_logs) / len(all_logs)
    if len(exact_logs) >= 2:
        variance = sum((value - mu0) ** 2 for value in exact_logs) / (len(exact_logs) - 1)
        eta0 = 1.0 / math.sqrt(max(variance, 1e-12))
    else:
        eta0 = 1.0
    eta0 = min(max(eta0, ETA_FLOOR * 10), ETA_CEILING / 10)
    return [(mu0, eta0), (mu0 + 1.0, eta0 * 0.5), (mu0 - 1.0, eta0 * 2.0)]


def fit_maximum_likelihood(records):
    """Fit mu and sigma, reporting convergence, boundaries and curvature."""
    exact_logs = [record.log_value for record in records if record.exact]
    all_logs = exact_logs + [record.log_limit for record in records if not record.exact]
    span = max(all_logs) - min(all_logs)
    mu_low, mu_high = min(all_logs) - 50.0, max(all_logs) + 50.0

    attempts = []
    best = None
    for start in _initial_points(records):
        result = _newton_step(records, start, mu_low, mu_high)
        attempts.append(
            {
                "method": "newton",
                "start": {"mu": start[0], "eta": start[1]},
                "converged": result["converged"],
                "iterations": result["iterations"],
                "log_likelihood": result["log_likelihood"],
                "reason": result["reason"],
            }
        )
        if best is None or result["log_likelihood"] > best["log_likelihood"]:
            best = result

    direct = _nelder_mead(records, (best["mu"], best["eta"]), mu_low, mu_high)
    cross_check_gap = best["log_likelihood"] - direct["log_likelihood"]
    if direct["log_likelihood"] > best["log_likelihood"] + 1e-7:
        # The gradient search missed the direct-search optimum; trust the latter.
        best = {
            "converged": True,
            "iterations": best["iterations"],
            "mu": direct["mu"],
            "eta": direct["eta"],
            "log_likelihood": direct["log_likelihood"],
            "gradient_inf_norm": float("nan"),
            "reason": "nelder_mead_overrode_newton",
            "start": best["start"],
        }

    mu, eta = best["mu"], best["eta"]
    sigma = 1.0 / eta
    (g_mu, g_eta), (h_mu_mu, h_mu_eta, h_eta_eta) = gradient_and_hessian(
        records, mu, eta
    )
    information_eigenvalues = _eigenvalues_2x2(
        -h_mu_mu, -h_mu_eta, -h_eta_eta
    )
    boundary_mu = (mu - mu_low < 1e-6) or (mu_high - mu < 1e-6)
    boundary_eta = eta <= ETA_FLOOR * 100 or eta >= ETA_CEILING / 100
    scale_boundary = span > 0.0 and sigma > max(20.0 * span, 100.0)
    min_eigenvalue, max_eigenvalue = information_eigenvalues
    flat_direction = min_eigenvalue <= 1e-8 * max(1.0, abs(max_eigenvalue))
    contributions = [
        {
            "record_id": record.record_id,
            "kind": record.kind,
            "log_likelihood_contribution": record_contribution(record, mu, eta),
        }
        for record in records
    ]
    return {
        "mu": mu,
        "eta": eta,
        "sigma": sigma,
        "log_likelihood": best["log_likelihood"],
        "converged": best["converged"],
        "termination_reason": best["reason"],
        "gradient_inf_norm": (
            best["gradient_inf_norm"]
            if math.isfinite(best["gradient_inf_norm"])
            else None
        ),
        "newton_attempts": attempts,
        "nelder_mead_log_likelihood": direct["log_likelihood"],
        "optimiser_log_likelihood_gap": cross_check_gap,
        "information_eigenvalues": list(information_eigenvalues),
        "boundary_solution": bool(boundary_mu or boundary_eta or scale_boundary),
        "boundary_flags": {
            "mu_at_search_boundary": boundary_mu,
            "eta_at_search_boundary": boundary_eta,
            "sigma_dwarfs_data_span": scale_boundary,
        },
        "flat_information_direction": bool(flat_direction),
        "likelihood_contributions": contributions,
    }
