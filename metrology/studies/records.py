"""Input records for concentration studies with left-censored results.

A non-detect is a *left-censored observation* ("strictly below a limit"),
never missing data and never substituted with zero or half the limit.
This module only normalises and validates input; it performs no
substitution of any kind.
"""

import math
from dataclasses import dataclass, field
from decimal import localcontext
from ..errors import InputError
from ..values import (
    identifier,
    label,
    number,
    require_object,
    integer,
    decimal_text,
    digest,
)
from ..calibration.units import Unit, UnitRegistry

DUPLICATE_POLICIES = ("exclude_specimen", "first", "last")
RESULT_KINDS = ("exact", "left_censored")


@dataclass(frozen=True)
class PreparedRecord:
    """A record converted into the common study unit."""

    record_id: str
    specimen: str
    measurement_item: str
    instrument: str
    certificate_revision: object  # identifier string or None
    sampling_batch: str
    kind: str  # "exact" | "left_censored"
    value: object = None  # Decimal, exact concentration (exact only)
    limit: object = None  # Decimal, censoring limit (censored only)
    detection_limit: object = None  # Decimal, capability metadata
    log_value: float = None
    log_limit: float = None
    original: dict = field(default_factory=dict)
    flags: tuple = ()

    @property
    def exact(self):
        return self.kind == "exact"

    def as_trace(self):
        """Per-record traceability entry used in the numeric report."""
        return {
            "record_id": self.record_id,
            "specimen": self.specimen,
            "measurement_item": self.measurement_item,
            "instrument": self.instrument,
            "certificate_revision": self.certificate_revision,
            "certificate_applicable": self.certificate_revision is not None,
            "sampling_batch": self.sampling_batch,
            "kind": self.kind,
            "value": decimal_text(self.value) if self.value is not None else None,
            "limit": decimal_text(self.limit) if self.limit is not None else None,
            "detection_limit": (
                decimal_text(self.detection_limit)
                if self.detection_limit is not None
                else None
            ),
            "unit": None,
            "flags": list(self.flags),
        }


@dataclass(frozen=True)
class Exclusion:
    record_id: str
    specimen: str
    reason: str
    detail: str
    policy: bool = False


# Exclusion reason vocabulary. Each reason is documented in the report so a
# reviewer can trace why an observation never reached the likelihood.
REASON_MISSING_FIELD = "missing_field"
REASON_LIMIT_ZERO = "zero_detection_limit"
REASON_NON_POSITIVE_EXACT = "non_positive_exact_value"
REASON_CENSOR_LIMIT_INVALID = "censoring_limit_not_positive"
REASON_CENSOR_LIMIT_MISMATCH = "censoring_limit_mismatch"
REASON_DUPLICATE_SPECIMEN = "duplicate_specimen_measurement"
REASON_SPECIMEN_GROUP_DROPPED = "specimen_group_dropped"


def _result_payload(value):
    require_object(value, {"kind", "value", "limit"}, {"kind"})
    kind = value["kind"]
    if kind not in RESULT_KINDS:
        raise InputError(
            "result.kind must be 'exact' or 'left_censored'", field="result.kind"
        )
    if kind == "exact" and "value" not in value:
        raise InputError("exact result requires value", field="result.value")
    if kind == "left_censored" and "limit" in value and "value" in value:
        raise InputError("left-censored result cannot also carry a value")
    return kind


def _parse_raw_record(raw, index):
    fields = {
        "record_id",
        "specimen",
        "measurement_item",
        "instrument",
        "certificate_revision",
        "sampling_batch",
        "detection_limit",
        "result",
        "unit",
    }
    require_object(raw, fields, fields - {"certificate_revision"})
    record_id = identifier(raw["record_id"], "record_id")
    specimen = identifier(raw["specimen"], "specimen")
    certificate = raw["certificate_revision"]
    if certificate is not None:
        certificate = identifier(certificate, "certificate_revision")
    kind = _result_payload(raw["result"])
    return {
        "record_id": record_id,
        "specimen": specimen,
        "measurement_item": identifier(raw["measurement_item"], "measurement_item"),
        "instrument": identifier(raw["instrument"], "instrument"),
        "certificate_revision": certificate,
        "sampling_batch": identifier(raw["sampling_batch"], "sampling_batch"),
        "detection_limit_text": raw["detection_limit"],
        "kind": kind,
        "result": raw["result"],
        "unit": identifier(raw["unit"], "unit"),
        "raw": raw,
        "input_index": index,
    }


def _float_log(decimal_value):
    return math.log(float(decimal_value))


def prepare_study(specification, registry=None):
    """Validate a study specification and convert every concentration to one unit.

    Returns a dict containing prepared accepted records, exclusions with
    explicit reasons and the frozen handling rules applied to the input.
    """
    fields = {
        "study_id",
        "target_unit",
        "population",
        "records",
        "units",
        "duplicate_policy",
        "threshold",
        "confidence_level",
        "bootstrap_replicates",
        "bootstrap_seed",
    }
    require_object(
        specification,
        fields,
        {"study_id", "target_unit", "population", "records"},
    )
    study_id = identifier(specification["study_id"], "study_id")
    population = label(specification["population"], "population")
    if registry is None:
        registry = UnitRegistry(
            UnitRegistry.builtins()
            + [Unit.parse(item) for item in specification.get("units", [])]
        )
    else:
        for item in specification.get("units", []):
            registry.add(item)
    target_unit = identifier(specification["target_unit"], "target_unit")
    registry.get(target_unit)  # raises when unknown
    duplicate_policy = specification.get("duplicate_policy", "exclude_specimen")
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise InputError("unsupported duplicate_policy", field="duplicate_policy")
    confidence_level = number(
        specification.get("confidence_level", ".95"), "confidence_level"
    )
    if not 0 < confidence_level < 1:
        raise InputError("confidence_level must be strictly between zero and one")
    replicates = integer(
        specification.get("bootstrap_replicates", 1000),
        "bootstrap_replicates",
        1,
        100000,
    )
    seed = integer(specification.get("bootstrap_seed", 0), "bootstrap_seed", 0)
    threshold = (
        number(specification["threshold"], "threshold", positive=True)
        if "threshold" in specification and specification["threshold"] is not None
        else None
    )
    raw_records = specification["records"]
    if not isinstance(raw_records, list) or not 1 <= len(raw_records) <= 10000:
        raise InputError("records must be a list of one to ten thousand items")

    prepared = []
    exclusions = []
    seen_ids = set()
    for index, raw in enumerate(raw_records):
        parsed = _parse_raw_record(raw, index)
        if parsed["record_id"] in seen_ids:
            raise InputError("record_id must be unique", field="record_id")
        seen_ids.add(parsed["record_id"])
        record_id, specimen = parsed["record_id"], parsed["specimen"]

        def exclude(reason, detail):
            exclusions.append(
                Exclusion(record_id, specimen, reason, detail).__dict__.copy()
            )

        if not registry.compatible(parsed["unit"], target_unit):
            # A dimension mismatch is a configuration error, not a data issue.
            raise InputError(
                "record unit is incompatible with the target unit",
                field="unit",
                details={"record_id": record_id},
            )
        with localcontext() as ctx:
            ctx.prec = 40
            detection = number(parsed["detection_limit_text"], "detection_limit")
            if detection < 0:
                raise InputError("detection_limit cannot be negative")
            detection_target = registry.convert(
                detection, parsed["unit"], target_unit, delta=True
            )
            if parsed["kind"] == "exact":
                exact = number(parsed["result"]["value"], "result.value")
                value_target = registry.convert(
                    exact, parsed["unit"], target_unit
                )
                if detection_target == 0:
                    exclude(
                        REASON_LIMIT_ZERO,
                        "detection limit converts to zero; measurement capability is "
                        "undefined so the record cannot support a censored study",
                    )
                    continue
                if value_target <= 0:
                    exclude(
                        REASON_NON_POSITIVE_EXACT,
                        "lognormal model requires strictly positive exact concentrations",
                    )
                    continue
                flags = []
                if value_target < detection_target:
                    # Reported as detected yet below the stated capability: keep,
                    # but flag the internal contradiction for reviewers.
                    flags.append("exact_below_detection_limit")
                prepared.append(
                    PreparedRecord(
                        record_id=record_id,
                        specimen=specimen,
                        measurement_item=parsed["measurement_item"],
                        instrument=parsed["instrument"],
                        certificate_revision=parsed["certificate_revision"],
                        sampling_batch=parsed["sampling_batch"],
                        kind="exact",
                        value=value_target,
                        detection_limit=detection_target,
                        log_value=_float_log(value_target),
                        original=parsed["raw"],
                        flags=tuple(flags),
                    )
                )
            else:
                result = parsed["result"]
                if detection_target == 0:
                    exclude(
                        REASON_LIMIT_ZERO,
                        "a left-censored record needs a positive limit; a zero "
                        "detection limit cannot bound the non-detect",
                    )
                    continue
                if "limit" in result:
                    censoring = number(result["limit"], "result.limit")
                    if censoring <= 0:
                        exclude(
                            REASON_CENSOR_LIMIT_INVALID,
                            "censoring limit must be strictly positive",
                        )
                        continue
                    censoring_target = registry.convert(
                        censoring, parsed["unit"], target_unit, delta=True
                    )
                    if censoring_target != detection_target:
                        exclude(
                            REASON_CENSOR_LIMIT_MISMATCH,
                            "explicit censoring limit must equal the record "
                            "detection limit after unit conversion",
                        )
                        continue
                else:
                    censoring_target = detection_target
                prepared.append(
                    PreparedRecord(
                        record_id=record_id,
                        specimen=specimen,
                        measurement_item=parsed["measurement_item"],
                        instrument=parsed["instrument"],
                        certificate_revision=parsed["certificate_revision"],
                        sampling_batch=parsed["sampling_batch"],
                        kind="left_censored",
                        limit=censoring_target,
                        detection_limit=detection_target,
                        log_limit=_float_log(censoring_target),
                        original=parsed["raw"],
                    )
                )

    prepared, duplicate_exclusions = _apply_duplicate_policy(
        prepared, duplicate_policy
    )
    exclusions.extend(duplicate_exclusions)

    certificates = sorted(
        {
            record.certificate_revision
            for record in prepared
            if record.certificate_revision is not None
        }
    )
    return {
        "study_id": study_id,
        "population": population,
        "target_unit": target_unit,
        "records": prepared,
        "exclusions": exclusions,
        "rules": {
            "duplicate_policy": duplicate_policy,
            "duplicate_policies": list(DUPLICATE_POLICIES),
            "censoring_substitution": "none; non-detects enter through CDF mass "
            "only, never zero or half the detection limit",
            "zero_detection_limit": "excluded with reason zero_detection_limit",
            "certificate_not_applicable": "certificate_revision may be null; such "
            "records are kept and marked certificate_applicable=false",
            "unit_conversion": (
                "all concentrations and limits converted to target unit before "
                "modelling"
            ),
        },
        "confidence_level": confidence_level,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "threshold": threshold,
        "certificate_revisions": certificates,
        "input_fingerprint": digest(specification),
    }


def _apply_duplicate_policy(prepared, policy):
    groups = {}
    for record in prepared:
        groups.setdefault(record.specimen, []).append(record)
    if policy == "exclude_specimen":
        kept, dropped = [], []
        for specimen, group in groups.items():
            if len(group) == 1:
                kept.append(group[0])
                continue
            for record in group:
                dropped.append(
                    {
                        "record_id": record.record_id,
                        "specimen": specimen,
                        "reason": REASON_DUPLICATE_SPECIMEN,
                        "detail": (
                            f"specimen has {len(group)} repeated measurements; "
                            "exclude_specimen drops the whole group"
                        ),
                        "policy": True,
                    }
                )
        kept.sort(key=lambda item: item.record_id)
        return kept, dropped

    kept, dropped = [], []
    for specimen, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        ordered = sorted(group, key=lambda item: item.record_id)
        chosen = ordered[0] if policy == "first" else ordered[-1]
        kept.append(chosen)
        for record in ordered:
            if record.record_id == chosen.record_id:
                continue
            dropped.append(
                {
                    "record_id": record.record_id,
                    "specimen": specimen,
                    "reason": REASON_DUPLICATE_SPECIMEN,
                    "detail": f"duplicate specimen measurement; {policy} policy "
                    f"kept {chosen.record_id}",
                    "policy": True,
                }
            )
    kept.sort(key=lambda item: item.record_id)
    return kept, dropped
