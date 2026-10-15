"""Schema inference for GrowthLens personal-inquiry analysis.

This module inspects a pandas DataFrame (an uploaded CSV/Excel file) and infers,
in a transparent and inspectable way:

* the column data types,
* the analytical role of each column (identifier, timestamp, event, dimension,
  measure, currency, ...),
* which columns can be used as dimensions vs measures,
* a first guess at the row grain,

and builds a *sanitized* payload that can be sent to an optional LLM provider.

Privacy note: :func:`public_payload` deliberately strips raw rows and any
values from identifier-like columns. Only column metadata, low-cardinality
category labels and aggregate counts ever leave the process, and only when the
caller has obtained explicit user consent.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

ID_PATTERN = re.compile(
    r"(^|_)(id|uuid|guid|user|customer|cust|visitor|session|email|e-mail|phone|"
    r"mobile|msisdn|account|acct|cookie|device_id|anon|hash)($|_)",
    re.IGNORECASE,
)
# Concatenated identifiers such as "visitorid", "itemid", "transactionid",
# "userid", "orderid" - common in raw event exports (e.g. Retailrocket).
CONCAT_ID_PATTERN = re.compile(
    r"(visitor|user|customer|cust|session|account|acct|order|item|product|article|"
    r"sku|transaction|trans|event|cart|basket|device|member|customer|client)id$",
    re.IGNORECASE,
)
TIMESTAMP_HINT = re.compile(
    r"(time|timestamp|date|datetime|_ts$|^ts$|created|occurred|event_at|at$|day)",
    re.IGNORECASE,
)
EVENT_HINT = re.compile(
    r"(event|action|activity|stage|step|behaviour|behavior|funnel|type_of_event)",
    re.IGNORECASE,
)
CURRENCY_HINT = re.compile(
    r"(revenue|amount|amt|price|value|spend|cost|sales|gmv|mrr|arr|total|payment|"
    r"charge|fee|basket|order_value|aov|ltv)",
    re.IGNORECASE,
)
GROUP_HINT = re.compile(
    r"(group|segment|variant|variation|bucket|experiment|test|cohort|arm|cell)",
    re.IGNORECASE,
)
DEVICE_HINT = re.compile(r"(device|platform|os|browser|channel|source|medium|campaign)", re.IGNORECASE)

# Columns whose values must never be shared with an external model.
NEVER_SHARE = re.compile(
    r"(email|e-mail|phone|mobile|msisdn|name|address|ssn|passport|iban|card|"
    r"ip|url|latitude|longitude|lat|lon|geo|postcode|zip)",
    re.IGNORECASE,
)


def _numeric_like(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def to_utc_naive(series: pd.Series, epoch_unit: Optional[str] = None) -> pd.Series:
    """Parse a column to tz-naive UTC datetimes.

    Epoch ``ms``/``s`` values are interpreted as UTC. Timezone-aware inputs are
    converted to UTC; naive inputs are treated as already-UTC. This is the single
    source of truth for "which instant is this row" across every report.
    """
    if epoch_unit == "ms":
        parsed = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ms")
    elif epoch_unit == "s":
        parsed = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def utc_week_start(series: pd.Series, epoch_unit: Optional[str] = None) -> pd.Series:
    """Monday 00:00 of the UTC calendar week for each timestamp.

    Monday is ISO weekday 0, so subtracting ``weekday`` days floors every instant
    to the Monday that starts its UTC week. This is used for BOTH cohort weeks and
    active weeks, so the two can never drift apart.
    """
    stamps = to_utc_naive(series, epoch_unit)
    return (stamps - pd.to_timedelta(stamps.dt.weekday, unit="D")).dt.normalize()


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Best-effort numeric coercion for amount-like columns.

    Handles thousands separators and both decimal conventions
    (``"1,234.50"`` and ``"105,5"``). Non-parseable values become NaN.
    """
    direct = pd.to_numeric(series, errors="coerce")
    if direct.notna().mean() >= 0.9 or not _numeric_like_strings(series):
        return direct

    def parse(value: str):
        cleaned = re.sub(r"[^\d,.\-]", "", str(value).strip())
        if not cleaned:
            return None
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            if cleaned.count(",") == 1 and len(cleaned.split(",")[-1]) in (1, 2):
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    parsed = series.astype(str).map(parse)
    return pd.to_numeric(parsed, errors="coerce")


def _numeric_like_strings(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _looks_like_datetime_strings(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(1000)
    if sample.empty:
        return False
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
        except Exception:  # noqa: BLE001
            return False
    return parsed.notna().mean() >= 0.9


def _epoch_unit(series: pd.Series) -> Optional[str]:
    """Detect Unix epoch columns and return 'ms', 's' or None."""
    if not _numeric_like(series):
        return None
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric.empty:
        return None
    median = float(numeric.abs().median())
    if median >= 1e12:  # milliseconds since 1970 (~2001+)
        return "ms"
    if median >= 1e8:  # seconds since 1970 (~1973+)
        return "s"
    return None


def _classify_dtype(series: pd.Series, name: str) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _numeric_like(series):
        # A numeric concatenated id (e.g. itemid) is an identifier, not a measure.
        if CONCAT_ID_PATTERN.search(name) or ID_PATTERN.search(name):
            return "identifier"
        return "numeric"
    # object / string / category
    if TIMESTAMP_HINT.search(name) and _looks_like_datetime_strings(series):
        return "datetime"
    if CONCAT_ID_PATTERN.search(name) or ID_PATTERN.search(name):
        return "identifier"
    # Amount-like text columns (e.g. "105,5" or "1,234.50") are numeric.
    if CURRENCY_HINT.search(name):
        non_null = series.replace("", pd.NA).dropna()
        if len(non_null) and coerce_numeric(non_null).notna().mean() >= 0.9:
            return "numeric"
    return "text"


def _guess_role(name: str, dtype: str, nunique: int, n_non_null: int) -> str:
    if dtype == "datetime":
        return "timestamp"
    if dtype == "identifier":
        return "identifier"
    if dtype == "numeric":
        if CURRENCY_HINT.search(name):
            return "currency"
        if TIMESTAMP_HINT.search(name):
            return "timestamp_numeric"
        if GROUP_HINT.search(name):
            return "dimension"
        return "measure"
    if dtype == "boolean":
        return "dimension"
    # text
    if EVENT_HINT.search(name):
        return "event"
    if DEVICE_HINT.search(name) or GROUP_HINT.search(name):
        return "dimension"
    if CONCAT_ID_PATTERN.search(name) or ID_PATTERN.search(name):
        return "identifier"
    return "dimension"


def infer_column(series: pd.Series, name: str, row_count: int) -> Dict[str, Any]:
    dtype = _classify_dtype(series, name)
    nunique = int(series.nunique(dropna=True))
    non_null = int(series.notna().sum())
    role = _guess_role(name, dtype, nunique, non_null)
    is_identifierish = bool(ID_PATTERN.search(name)) or bool(CONCAT_ID_PATTERN.search(name)) or dtype == "identifier"
    low_cardinality = nunique <= 30 and nunique > 0
    epoch_unit = _epoch_unit(series) if role in ("timestamp_numeric", "timestamp") else None
    if epoch_unit and role == "timestamp_numeric":
        role = "timestamp"
    return {
        "name": name,
        "dtype": dtype,
        "role": role,
        "nunique": nunique,
        "non_null": non_null,
        "row_count": row_count,
        "is_identifierish": is_identifierish,
        "low_cardinality": low_cardinality,
        "epoch_unit": epoch_unit,
    }


def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Return an inspectable schema description for ``df``."""
    row_count = int(len(df))
    columns: List[Dict[str, Any]] = [infer_column(df[col], str(col), row_count) for col in df.columns]

    dimensions = [c["name"] for c in columns if c["role"] in ("dimension", "event") and not c["is_identifierish"]]
    measures = [c["name"] for c in columns if c["role"] in ("measure", "currency")]
    identifiers = [c["name"] for c in columns if c["role"] == "identifier" or c["is_identifierish"]]
    timestamps = [c["name"] for c in columns if c["role"] == "timestamp"]
    events = [c["name"] for c in columns if c["role"] == "event"]
    currency = [c["name"] for c in columns if c["role"] == "currency"]

    grain = _guess_grain(columns, identifiers, timestamps, events)

    return {
        "row_count": row_count,
        "columns": columns,
        "dimensions": dimensions,
        "measures": measures,
        "identifiers": identifiers,
        "timestamps": timestamps,
        "events": events,
        "currency": currency,
        "grain_guess": grain,
    }


def _guess_grain(columns, identifiers, timestamps, events) -> str:
    if identifiers and timestamps and events:
        return "one row per user event (event log)"
    if identifiers and timestamps:
        return "one row per user per period (activity/appearance)"
    if identifiers and not timestamps:
        return "one row per user (snapshot)"
    if timestamps:
        return "one row per period (aggregated time series)"
    return "one row per record (unknown grain)"


def column_by_name(schema: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for col in schema.get("columns", []):
        if col["name"] == name:
            return col
    return None


def public_payload(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    include_categories: bool = False,
    low_card_threshold: int = 20,
) -> Dict[str, Any]:
    """Build a sanitized payload safe to send to an optional LLM provider.

    * No raw rows are included.
    * Identifier-like columns are reduced to type/role/distinct-count only.
    * Columns matching :data:`NEVER_SHARE` (email/name/phone/address/...) never
      expose their values.
    * Low-cardinality category labels are shared only when
      ``include_categories`` is explicitly True AND the column is not
      identifier-like or sensitive. The default is False so that any new caller
      must opt in to sharing even non-sensitive category labels.
    """
    cols: List[Dict[str, Any]] = []
    for col in schema["columns"]:
        entry: Dict[str, Any] = {
            "name": col["name"],
            "role": col["role"],
            "type": col["dtype"],
            "distinct_count": col["nunique"],
        }
        if col.get("epoch_unit"):
            entry["epoch_unit"] = col["epoch_unit"]
        share_values = (
            include_categories
            and col["role"] in ("dimension", "event")
            and col["low_cardinality"]
            and col["nunique"] <= low_card_threshold
            and not col["is_identifierish"]
            and not NEVER_SHARE.search(col["name"])
        )
        if share_values:
            counts = df[col["name"]].astype(str).value_counts().head(low_card_threshold)
            entry["top_values"] = [str(v) for v in counts.index.tolist()]
        cols.append(entry)

    return {
        "row_count": schema["row_count"],
        "grain_guess": schema["grain_guess"],
        "columns": cols,
        "available_dimensions": schema["dimensions"],
        "available_measures": schema["measures"],
        "identifiers_present": bool(schema["identifiers"]),
        "timestamps_present": bool(schema["timestamps"]),
        "events_present": bool(schema["events"]),
    }


# ---------------------------------------------------------------------------
# Strict outbound sanitization (planning + explanation)
# ---------------------------------------------------------------------------

# A value column is unsafe to forward when it is an identifier, matches a
# sensitive-name pattern, or has too many distinct values to be a "category".
_REDACT = "<redacted>"


def column_is_sensitive(schema: Dict[str, Any], name: str, low_card_threshold: int = 20) -> bool:
    """True when a column's *values* must never be forwarded to an LLM.

    Result tables contain both source columns (grouping keys, which may be
    identifiers/PII) and derived metric columns (``rows``, ``sum(amount)``, ...).
    Only source columns can be sensitive; a name that is not in the schema is a
    derived metric/label and is safe to forward.
    """
    col = column_by_name(schema, name)
    if col is None:
        return False
    if col.get("is_identifierish") or col.get("role") == "identifier":
        return True
    if NEVER_SHARE.search(name):
        return True
    if col.get("role") in ("dimension", "event") or col.get("dtype") in ("text", "boolean"):
        return int(col.get("nunique", 0)) > low_card_threshold
    return False


def sanitize_text(text: str, sensitive_values: List[str]) -> str:
    if not text:
        return text
    text_out = text
    for s in sensitive_values or []:
        if not s:
            continue
        text_out = re.sub(re.escape(str(s)), "<redacted>", text_out)
    # Mask email-like strings that may appear in notes/errors.
    text_out = re.sub(r"[\w\.-]+@[\w\.-]+\.[\w]+", "<redacted>", text_out)
    return text_out


def sanitize_notes(notes: List[Any], plan: Optional[Dict[str, Any]] = None) -> List[str]:
    sensitive: List[str] = []
    if plan:
        for f in plan.get("filters") or []:
            v = f.get("value")
            if isinstance(v, (list, tuple)):
                for b in v:
                    if b is not None and str(b) != "":
                        sensitive.append(str(b))
            elif v is not None and str(v) != "":
                sensitive.append(str(v))
    out: List[str] = []
    for n in notes or []:
        out.append(sanitize_text(str(n), sensitive))
    return out


def sanitize_table_preview(
    table: pd.DataFrame,
    schema: Dict[str, Any],
    max_rows: int = 10,
    low_card_threshold: int = 20,
) -> Dict[str, Any]:
    """Return a rows-only preview with sensitive/identifier value columns redacted.
    """
    if table is None or table.empty:
        return {"rows": [], "redacted_columns": []}
    preview = table.head(max_rows).copy()
    redacted: List[str] = []
    for col in preview.columns:
        if column_is_sensitive(schema, str(col), low_card_threshold):
            preview[col] = _REDACT
            redacted.append(str(col))
    rows = json.loads(preview.to_json(orient="records", date_format="iso")) if not preview.empty else []
    return {"rows": rows, "redacted_columns": redacted}


def sanitize_plan(plan: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a copy of an analysis plan with literal values redacted.

    Column *names* are kept (the explanation needs them), but user-supplied
    literal values (filter values, and any value on a sensitive/identifier
    column) are replaced with ``<redacted>`` so a typed-in email or visitor id
    cannot leak out through the plan.
    """
    schema = schema or {}
    safe = dict(plan)
    if safe.get("filters"):
        safe["filters"] = [
            {**f, "value": _REDACT} for f in safe["filters"]
        ]
    if schema:
        for key in ("x", "y"):
            if safe.get(key) and column_is_sensitive(schema, safe[key]):
                safe[key] = _REDACT
    return safe
