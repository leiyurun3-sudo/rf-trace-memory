"""Strict response parsing for the local judge; no inference or label defaults.

Each public function returns a plain dict and raises ValueError with retry-ready
details on failure. Validators accept one optional rubric/ratings wrapper, drop
extra keys, and never mutate their inputs. Raw responses remain the caller's
responsibility. Rubric fields follow the six-field contract in judge.py.
"""

import json


RUBRIC_KEYS = (
    "requested_edit", "protected_requirements", "subject_applicable",
    "subject_reason", "pattern_applicable", "pattern_reason",
)
CRITERIA = ("edit", "subject", "pattern", "artifact")
RATING_KEYS = ("value", "applicable", "reason")

RUBRIC_SCHEMA_EXAMPLE = '''{
  "requested_edit": ["Describe the actual requested edit from the task."],
  "protected_requirements": [],
  "subject_applicable": null,
  "subject_reason": "Source-only subject applicability is uncertain.",
  "pattern_applicable": null,
  "pattern_reason": "Source-only pattern applicability is uncertain."
}'''

RATINGS_SCHEMA_EXAMPLE = '''{
  "edit": {"value": null, "applicable": true, "reason": "Edit achievement is uncertain."},
  "subject": {"value": null, "applicable": null, "reason": "Frozen subject applicability is unknown."},
  "pattern": {"value": null, "applicable": null, "reason": "Frozen pattern applicability is unknown."},
  "artifact": {"value": null, "applicable": true, "reason": "Artifact assessment is uncertain."}
}'''


class ResponseSchemaError(ValueError):
    """A response requires correction by the model, without imputed fields."""


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseSchemaError("JSON contains duplicate key: " + repr(key))
        result[key] = value
    return result


def _invalid_constant(value):
    raise ResponseSchemaError("JSON contains nonstandard numeric constant: " + value)


def parse_json_object(text):
    """Extract one JSON object, optionally fenced or surrounded by prose.

    Multiple objects (even identical ones), duplicate keys, and NaN/Infinity are
    errors. Nested dictionaries belong to their enclosing object and do not count
    as separate candidates. A complete JSON array/scalar is not an object.
    """
    if not isinstance(text, str) or not text.strip():
        raise ResponseSchemaError("Response must be nonempty text containing one JSON object.")
    text = text.strip()
    decoder = json.JSONDecoder(
        object_pairs_hook=_object_pairs, parse_constant=_invalid_constant,
    )
    try:
        complete = decoder.decode(text)
    except json.JSONDecodeError:
        pass
    else:
        if not isinstance(complete, dict):
            raise ResponseSchemaError("Expected one JSON object, not an array or scalar.")
        return complete

    objects = []
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        objects.append(obj)
        if len(objects) > 1:
            raise ResponseSchemaError(
                "Response contains multiple JSON objects; return exactly one, without alternatives."
            )
        position = end
    if not objects:
        raise ResponseSchemaError("No valid JSON object found; return one object using valid JSON syntax.")
    return objects[0]


def _required_object(data, required, path, wrapper=None):
    if not isinstance(data, dict):
        raise ResponseSchemaError(path + " must be a JSON object.")
    if wrapper is not None and wrapper in data:
        if any(key in data for key in required):
            raise ResponseSchemaError(path + " mixes wrapped and unwrapped fields; return one form only.")
        data = data[wrapper]
        if not isinstance(data, dict):
            raise ResponseSchemaError(path + " wrapper must contain a JSON object.")
    missing = [key for key in required if key not in data]
    if missing:
        raise ResponseSchemaError(path + " missing required fields: " + ", ".join(missing) + ".")
    return data


def _reason(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ResponseSchemaError(path + " must be a nonempty explanatory string.")
    return value


def _applicability(value, path):
    if value is not None and type(value) is not bool:
        raise ResponseSchemaError(path + " must be JSON true, false, or null (not a number/string).")
    return value


def validate_rubric(data):
    """Return the six canonical source-only rubric fields; allow empty protection.

    requested_edit is a nonempty list of nonblank strings. An empty
    protected_requirements list is valid and never filled with constraints.
    """
    data = _required_object(data, RUBRIC_KEYS, "rubric", "rubric")
    result = {}
    for key in ("requested_edit", "protected_requirements"):
        values = data[key]
        if not isinstance(values, list) or (key == "requested_edit" and not values):
            qualifier = "nonempty " if key == "requested_edit" else "possibly empty "
            raise ResponseSchemaError("rubric." + key + " must be a " + qualifier + "array of nonempty strings.")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ResponseSchemaError("rubric." + key + " must contain only nonempty strings.")
        result[key] = list(values)
    for criterion in ("subject", "pattern"):
        key = criterion + "_applicable"
        result[key] = _applicability(data[key], "rubric." + key)
        key = criterion + "_reason"
        result[key] = _reason(data[key], "rubric." + key)
    return result


def validate_ratings(data, frozen_rubric):
    """Return exactly four nested ratings, checked against the frozen rubric.

    Each rating requires value, applicable, and a nonblank reason. value must be
    an actual integer 0/1 or null; bools and floats are rejected. edit/artifact
    applicability is true. subject/pattern applicability must equal the frozen
    value, including null; non-true applicability requires a null rating value.
    """
    frozen = validate_rubric(frozen_rubric)
    data = _required_object(data, CRITERIA, "ratings", "ratings")
    result = {}
    for criterion in CRITERIA:
        path = "ratings." + criterion
        if not isinstance(data[criterion], dict):
            raise ResponseSchemaError(
                path + " must be a nested object with required fields value, applicable, reason; "
                "flat scores are invalid. Return all three fields explicitly, with null for an uncertain value."
            )
        rating = _required_object(data[criterion], RATING_KEYS, path)
        value = rating["value"]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ResponseSchemaError(path + ".value must be integer 0, integer 1, or null (not bool/float/string).")
        applicable = _applicability(rating["applicable"], path + ".applicable")
        expected = frozen[criterion + "_applicable"] if criterion in ("subject", "pattern") else True
        if applicable is not expected:
            raise ResponseSchemaError(
                path + ".applicable must be " + json.dumps(expected)
                + (" to match the frozen source-only rubric." if criterion in ("subject", "pattern")
                   else "; use value null with a reason for uncertainty.")
            )
        if applicable is not True and value is not None:
            raise ResponseSchemaError(path + ".value must be null when applicable is false or null.")
        result[criterion] = {
            "value": value, "applicable": applicable,
            "reason": _reason(rating["reason"], path + ".reason"),
        }
    return result


def active_ratings_example(frozen_rubric):
    """Only model decisions: applicability is fixed metadata, not a prediction."""
    frozen = validate_rubric(frozen_rubric)
    return {key: {"value": None, "reason": "Explain the visible evidence or uncertainty."}
            for key in CRITERIA
            if key in ("edit", "artifact") or frozen[key + "_applicable"] is True}


def validate_active_ratings(data, frozen_rubric):
    """Validate applicable decisions, then attach source-only metadata.

    Inapplicable and uncertain-applicability preservation items are excluded
    from output prediction. Their canonical null is not an inferred pass/fail.
    Extra model fields, including applicability, remain in the raw record only.
    This accepts old nested records for explicitly audited cache migration.
    """
    frozen = validate_rubric(frozen_rubric)
    active = list(active_ratings_example(frozen))
    data = _required_object(data, active, "ratings", "ratings")
    result = {}
    for key in CRITERIA:
        expected = frozen[key + "_applicable"] if key in ("subject", "pattern") else True
        if expected is not True:
            result[key] = {"value": None, "applicable": expected,
                           "reason": "Frozen source-only rubric: " + frozen[key + "_reason"]}
            continue
        path = "ratings." + key
        rating = _required_object(data[key], ("value", "reason"), path)
        value = rating["value"]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ResponseSchemaError(path + ".value must be integer 0, integer 1, or null.")
        result[key] = {"value": value, "applicable": True,
                       "reason": _reason(rating["reason"], path + ".reason")}
    return result
