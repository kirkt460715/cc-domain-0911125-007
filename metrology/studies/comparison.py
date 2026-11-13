"""Before/after instrument comparison with a common-limit sensitivity control.

The two periods are estimated separately under the same specimen type and
sampling range. A sensitivity analysis then artificially censors *both*
datasets to the higher of the two effective detection limits; that control
equalises measurement capability and therefore isolates how much of any
apparent shift is driven by censoring rather than by the samples. It does
not recover the original unknown values and must never be described as
restoring the truth.
"""

from decimal import localcontext
from ..errors import InputError
from ..values import identifier, require_object, number, decimal_text
from ..calibration.units import Unit, UnitRegistry
from .records import prepare_study
from .study import run_concentration_study


def _common_limit(records):
    limits = [record.detection_limit for record in records]
    if not limits:
        return None
    return max(limits)


def _censor_to_common(record, common_limit):
    """Return a trace entry describing how a record is re-censored."""
    if not record.exact:
        return {
            "record_id": record.record_id,
            "original_kind": "left_censored",
            "sensitivity_kind": "left_censored",
            "sensitivity_limit": decimal_text(common_limit),
            "note": "already censored; limit raised to the common value",
        }
    if record.value < common_limit:
        return {
            "record_id": record.record_id,
            "original_kind": "exact",
            "sensitivity_kind": "left_censored",
            "sensitivity_limit": decimal_text(common_limit),
            "note": "exact value below the common limit; converted to a "
            "left-censored observation",
        }
    return {
        "record_id": record.record_id,
        "original_kind": "exact",
        "sensitivity_kind": "exact",
        "sensitivity_limit": None,
        "note": "exact value at or above the common limit; unchanged",
    }


def _sensitivity_specification(base, records, common_limit, suffix):
    sensitivity_records = []
    trace = []
    for record in records:
        entry = _censor_to_common(record, common_limit)
        trace.append(entry)
        original = dict(record.original)
        if entry["sensitivity_kind"] == "left_censored":
            original["result"] = {
                "kind": "left_censored",
                "limit": decimal_text(common_limit),
            }
            original["detection_limit"] = decimal_text(common_limit)
        sensitivity_records.append(original)
    specification = {
        "study_id": base["study_id"] + suffix,
        "target_unit": base["target_unit"],
        "population": base["population"],
        "records": sensitivity_records,
        "units": base.get("units", []),
        "duplicate_policy": base.get("duplicate_policy", "exclude_specimen"),
        "confidence_level": base.get("confidence_level", ".95"),
        "bootstrap_replicates": base.get("bootstrap_replicates", 1000),
        "bootstrap_seed": base.get("bootstrap_seed", 0),
    }
    if base.get("threshold") is not None:
        specification["threshold"] = base["threshold"]
    return specification, trace


def compare_instrument_periods(specification, registry=None):
    fields = {
        "comparison_id",
        "specimen_type",
        "sampling_range",
        "before",
        "after",
        "units",
        "threshold",
    }
    require_object(specification, fields, fields - {"units", "threshold"})
    comparison_id = identifier(specification["comparison_id"], "comparison_id")
    specimen_type = identifier(specification["specimen_type"], "specimen_type")
    sampling_range = require_object(specification["sampling_range"], required={"start", "end"})
    if registry is None:
        registry = UnitRegistry(
            UnitRegistry.builtins()
            + [Unit.parse(item) for item in specification.get("units", [])]
        )

    def normalize(side):
        base = dict(side)
        base.setdefault("units", specification.get("units", []))
        if "threshold" not in base and specification.get("threshold") is not None:
            base["threshold"] = specification["threshold"]
        return base

    before_spec = normalize(specification["before"])
    after_spec = normalize(specification["after"])
    if before_spec["target_unit"] != after_spec["target_unit"]:
        raise InputError("both periods must use the same target unit")
    target_unit = before_spec["target_unit"]

    before_report = run_concentration_study(before_spec, registry)
    after_report = run_concentration_study(after_spec, registry)
    before_records = prepare_study(before_spec, registry)["records"]
    after_records = prepare_study(after_spec, registry)["records"]
    before_limit = _common_limit(before_records)
    after_limit = _common_limit(after_records)
    common_limit = max(value for value in (before_limit, after_limit) if value is not None)

    before_sens_spec, before_trace = _sensitivity_specification(
        before_spec, before_records, common_limit, "-common-limit-before"
    )
    after_sens_spec, after_trace = _sensitivity_specification(
        after_spec, after_records, common_limit, "-common-limit-after"
    )
    before_sensitivity = run_concentration_study(before_sens_spec, registry)
    after_sensitivity = run_concentration_study(after_sens_spec, registry)

    return {
        "schema": 1,
        "comparison_id": comparison_id,
        "specimen_type": specimen_type,
        "sampling_range": sampling_range,
        "target_unit": target_unit,
        "periods": {
            "before": _period_summary(before_report),
            "after": _period_summary(after_report),
        },
        "detection_limits": {
            "before_max": decimal_text(before_limit),
            "after_max": decimal_text(after_limit),
            "common_sensitivity_limit": decimal_text(common_limit),
        },
        "sensitivity_control": {
            "common_limit": decimal_text(common_limit),
            "before": _period_summary(before_sensitivity),
            "after": _period_summary(after_sensitivity),
            "record_censoring_trace": {
                "before": before_trace,
                "after": after_trace,
            },
            "warning": "artificial common-limit censoring equalises measurement "
            "capability to isolate censoring effects; it discards information and "
            "does not recover original unknown concentrations or ground truth",
        },
        "interpretation": "compare period estimates at native limits to see the "
        "observed effect of the more sensitive instrument, then compare the "
        "common-limit estimates: differences that survive common censoring are "
        "attributable to the samples rather than to detection capability",
        "disclaimer": "this comparison does not replace existing conformity or "
        "acceptance decisions",
        "study_fingerprints": {
            "before": before_report["report_fingerprint"],
            "after": after_report["report_fingerprint"],
            "before_sensitivity": before_sensitivity["report_fingerprint"],
            "after_sensitivity": after_sensitivity["report_fingerprint"],
        },
    }


def _period_summary(report):
    return {
        "study_id": report["study_id"],
        "identifiability": report["identifiability"],
        "counts": report["counts"],
        "point_estimates": report["point_estimates"],
        "exact_only_descriptive": report["exact_only_descriptive"],
        "fit": None if report["fit"] is None else {
            "converged": report["fit"]["converged"],
            "boundary_solution": report["fit"]["boundary_solution"],
            "log_likelihood": report["fit"]["log_likelihood"],
            "termination_reason": report["fit"]["termination_reason"],
        },
        "bootstrap": None if report["bootstrap"] is None else {
            "successful_replicates": report["bootstrap"]["successful_replicates"],
            "failed_replicates": report["bootstrap"]["failed_replicates"],
            "failure_reasons": report["bootstrap"]["failure_reasons"],
            "confidence_intervals": report["bootstrap"]["confidence_intervals"],
        },
    }
