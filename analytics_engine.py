"""Structured, safe execution engine for GrowthLens personal-inquiry analysis.

The engine receives a *validated, strictly structured* analysis plan (a plain
dict, normally proposed by an optional LLM) and computes real metrics against
the uploaded DataFrame using pandas.

Hard safety rules enforced here:

* Only whitelisted intents, aggregations and filter operators are accepted.
* No ``eval``/``exec``, no arbitrary Python/SQL/shell is ever executed.
* Every referenced column must exist in the dataset; otherwise the plan is
  reported as *unanswerable* with the list of missing fields.
* Revenue-type metrics are refused unless a valid amount column is present.
* Correlation is reported as association only - never causation.

The module depends only on pandas (no Streamlit), so it is unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import analytics_schema
import standard_reports

ALLOWED_INTENTS = {
    "groupby_aggregate",
    "distinct_count",
    "time_trend",
    "event_stages",
    "funnel",
    "cohort",
    "correlation",
    "unsupported",
}

ALLOWED_AGGREGATIONS = {"sum", "mean", "median", "count", "nunique", "min", "max"}
ALLOWED_FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "in", "not_in", "contains", "between"}
NUMERIC_AGGS = {"sum", "mean", "median", "min", "max"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(schema: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for c in schema.get("columns", []):
        if c["name"] == name:
            return c
    return None


def _is_numeric(schema: Dict[str, Any], name: str) -> bool:
    c = _col(schema, name)
    return bool(c and c["dtype"] in ("numeric",) or (c and c["role"] in ("measure", "currency")))


def _is_identifierish(schema: Dict[str, Any], name: str) -> bool:
    c = _col(schema, name)
    return bool(c and (c.get("is_identifierish") or c.get("role") == "identifier"))


def _coerce_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_errors(f: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Return validation errors for a single filter (fail-closed, no skipping).

    Error strings deliberately name only the *column* and *operator* - never the
    user-supplied literal value - so a filter value can never leak through an
    error message that later reaches an external explanation call.
    """
    col, op, value = f.get("column"), f.get("op"), f.get("value")
    if op not in ALLOWED_FILTER_OPS:
        return [f"Unsupported filter operator: {op!r}."]
    if op in (">", "<", ">=", "<="):
        if not _is_numeric(schema, col):
            return [f"Filter '{op}' needs a numeric column, but '{col}' is not numeric."]
        if _coerce_number(value) is None:
            return [f"Filter '{col} {op}' needs a numeric value."]
    elif op == "between":
        if not _is_numeric(schema, col):
            return [f"Filter 'between' needs a numeric column, but '{col}' is not numeric."]
        bounds = value if isinstance(value, (list, tuple)) else None
        if bounds is None or len(bounds) != 2 or any(_coerce_number(b) is None for b in bounds):
            return [f"Filter '{col} between' needs two numeric bounds."]
    elif op in ("in", "not_in"):
        if not isinstance(value, (list, tuple)):
            return [f"Filter '{col} {op}' needs a list of values."]
    return []


def _time_series(df: pd.DataFrame, name: str, schema: Dict[str, Any]) -> pd.Series:
    col = _col(schema, name)
    unit = col.get("epoch_unit") if col else None
    return analytics_schema.to_utc_naive(df[name], unit)


def _period_floor(series: pd.Series, gran: str) -> pd.Series:
    if gran == "W":
        return analytics_schema.utc_week_start(series)
    if gran == "M":
        return series.dt.to_period("M").dt.to_timestamp()
    return series.dt.normalize()  # daily


def _period_index(start: pd.Series, end: pd.Series, gran: str) -> pd.Series:
    if gran == "W":
        return (end - start).dt.days // 7
    if gran == "M":
        return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    return (end - start).dt.days


def _apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    work = df
    for f in filters or []:
        name, op, value = f.get("column"), f.get("op"), f.get("value")
        if name not in work.columns:
            raise ValueError(f"filter column '{name}' is not present in the data")
        if op not in ALLOWED_FILTER_OPS:
            raise ValueError(f"filter operator '{op}' is not supported")
        series = work[name]
        try:
            if op == "==":
                mask = series == value
            elif op == "!=":
                mask = series != value
            elif op in (">", "<", ">=", "<="):
                numeric = pd.to_numeric(series, errors="coerce")
                rhs = float(value)
                mask = {
                    ">": numeric > rhs,
                    "<": numeric < rhs,
                    ">=": numeric >= rhs,
                    "<=": numeric <= rhs,
                }[op]
            elif op == "in":
                mask = series.astype(str).isin([str(v) for v in (value or [])])
            elif op == "not_in":
                mask = ~series.astype(str).isin([str(v) for v in (value or [])])
            elif op == "contains":
                mask = series.astype(str).str.contains(str(value), case=False, na=False)
            elif op == "between":
                bounds = list(value)
                numeric = pd.to_numeric(series, errors="coerce")
                mask = (numeric >= float(bounds[0])) & (numeric <= float(bounds[1]))
            else:  # pragma: no cover - guarded above
                raise ValueError(f"filter operator '{op}' is not supported")
            work = work[mask.fillna(False)]
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            # Do NOT include the raw exception text: it can embed the
            # user-supplied literal (e.g. float("alice@example.com")).
            raise ValueError(f"filter on '{name}' ({op}) could not be applied") from exc
    return work, notes


def _aggregate(series: pd.Series, op: str, numeric_col: bool) -> Any:
    if op == "count":
        return int(series.count())
    if op == "nunique":
        return int(series.nunique(dropna=True))
    if op in NUMERIC_AGGS and not numeric_col:
        raise ValueError("numeric aggregation on a non-numeric column")
    numeric = pd.to_numeric(series, errors="coerce")
    return float(getattr(numeric, op)())


def _empty(intent: str, note: str) -> Dict[str, Any]:
    return {
        "intent": intent,
        "table": pd.DataFrame(),
        "scalar": None,
        "chart": {"kind": "none"},
        "notes": [note],
        "meta": {},
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_plan(plan: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """Return (ok, errors, missing_columns)."""
    errors: List[str] = []
    missing: List[str] = []

    if not isinstance(plan, dict):
        return False, ["The analysis plan was not a structured object."], []

    intent = plan.get("intent")
    if intent not in ALLOWED_INTENTS:
        errors.append(f"Unsupported analysis intent: {intent!r}.")

    names = {c["name"] for c in schema.get("columns", [])}

    def require(col):
        if col and col not in names and col not in missing:
            missing.append(col)

    metric = plan.get("metric") or {}
    require(metric.get("column"))
    op = metric.get("operation")
    if op and op not in ALLOWED_AGGREGATIONS:
        errors.append(f"Unsupported aggregation: {op!r}.")
    if op in NUMERIC_AGGS and metric.get("column"):
        if _is_identifierish(schema, metric["column"]):
            errors.append(
                f"Aggregation '{op}' is not valid on identifier column "
                f"'{metric['column']}' (use a distinct count instead)."
            )
        elif not _is_numeric(schema, metric["column"]):
            errors.append(f"Aggregation '{op}' needs numeric column '{metric['column']}'.")

    for g in plan.get("group_by") or []:
        require(g)
    for f in plan.get("filters") or []:
        col = f.get("column")
        if not col:
            missing.append("a filter column")
            continue
        if col not in names:
            require(col)
            continue
        errors.extend(_filter_errors(f, schema))

    time = plan.get("time") or {}
    require(time.get("column"))

    require(plan.get("user_id"))
    require(plan.get("event_column"))
    require(plan.get("timestamp"))

    for stage in plan.get("stages") or []:
        if not stage.get("event_value"):
            errors.append("A funnel stage is missing its event value.")

    x = plan.get("x")
    y = plan.get("y")
    if intent == "correlation":
        require(x)
        require(y)

    if intent == "funnel":
        if not plan.get("user_id"):
            missing.append("a user/visitor identifier column")
        if not plan.get("timestamp"):
            missing.append("a timestamp column")
        if len(plan.get("stages") or []) < 2:
            errors.append("A funnel needs at least two ordered stages.")

    if intent == "cohort":
        if not plan.get("user_id"):
            missing.append("a user/visitor identifier column")
        if not plan.get("timestamp"):
            missing.append("a timestamp column")

    ok = not errors and not missing
    return ok, errors, missing


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_plan(df: pd.DataFrame, plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a validated plan and return a result dict.

    Result keys: ``intent``, ``table`` (DataFrame), ``scalar``, ``chart``
    (a spec the UI turns into Plotly), ``notes`` (list[str]), ``meta`` (dict).
    """
    ok, errors, missing = validate_plan(plan, schema)
    if not ok:
        note = "This question cannot be answered from the uploaded data."
        if missing:
            note += " Missing fields: " + ", ".join(missing) + "."
        if errors:
            note += " " + " ".join(errors)
        return _empty(plan.get("intent", "unsupported"), note)

    intent = plan["intent"]
    try:
        work, filter_notes = _apply_filters(df, plan.get("filters") or [])
    except ValueError as exc:
        return _empty(intent, f"The analysis could not be run because a filter was invalid: {exc}")

    if intent == "groupby_aggregate":
        result = _run_groupby(work, plan)
    elif intent == "distinct_count":
        result = _run_distinct(work, plan)
    elif intent == "time_trend":
        result = _run_time_trend(work, plan, schema)
    elif intent == "event_stages":
        result = _run_event_stages(work, plan)
    elif intent == "funnel":
        result = _run_funnel(work, plan, schema)
    elif intent == "cohort":
        result = _run_cohort(work, plan, schema)
    elif intent == "correlation":
        result = _run_correlation(work, plan)
    else:
        result = _empty(intent, plan.get("reason") or "Unsupported analysis.")

    result["notes"] = filter_notes + result.get("notes", [])
    return result


def _metric(plan):
    m = plan.get("metric") or {}
    return m.get("operation", "count"), m.get("column")


def _run_groupby(work: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    group_by = [g for g in (plan.get("group_by") or []) if g in work.columns]
    op, col = _metric(plan)
    if not group_by:
        return _empty("groupby_aggregate", "No valid grouping column was provided.")
    try:
        work = work.copy()
        if op in NUMERIC_AGGS and col in work.columns:
            work[col] = analytics_schema.coerce_numeric(work[col])
        if op in ("count", "nunique") and (col is None or col not in work.columns):
            grouped = work.groupby(group_by, dropna=False).size()
            label = "rows"
        else:
            grouped = work.groupby(group_by, dropna=False)[col].agg(op)
            label = f"{op}({col})"
    except Exception as exc:  # noqa: BLE001
        return _empty("groupby_aggregate", f"Could not compute the grouping: {exc}")
    table = grouped.reset_index()
    table.columns = group_by + [label]
    table = table.sort_values(label, ascending=False).reset_index(drop=True)
    return {
        "intent": "groupby_aggregate",
        "table": table,
        "scalar": None,
        "chart": {"kind": "bar", "x": group_by, "y": label, "title": f"{label} by " + ", ".join(group_by)},
        "notes": [],
        "meta": {"metric_label": label, "group_by": group_by},
    }


def _run_distinct(work: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    op, col = _metric(plan)
    target = plan.get("user_id") or col
    if target not in work.columns:
        return _empty("distinct_count", "No valid identifier column was provided.")
    group_by = [g for g in (plan.get("group_by") or []) if g in work.columns]
    label = f"distinct {target}"
    if group_by:
        table = work.groupby(group_by, dropna=False)[target].nunique().reset_index()
        table.columns = group_by + [label]
        table = table.sort_values(label, ascending=False).reset_index(drop=True)
        chart = {"kind": "bar", "x": group_by, "y": label, "title": f"Distinct users by " + ", ".join(group_by)}
        scalar = None
    else:
        scalar = int(work[target].nunique(dropna=True))
        table = pd.DataFrame({label: [scalar]})
        chart = {"kind": "none"}
    return {"intent": "distinct_count", "table": table, "scalar": scalar, "chart": chart, "notes": [], "meta": {}}


def _run_time_trend(work: pd.DataFrame, plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    time = plan.get("time") or {}
    col = time.get("column")
    gran = time.get("period", "W")
    if col not in work.columns:
        return _empty("time_trend", "No valid time column was provided.")
    op, metric_col = _metric(plan)
    group_by = [g for g in (plan.get("group_by") or []) if g in work.columns]
    ts = _time_series(work, col, schema)
    tmp = work.assign(__period=_period_floor(ts, gran))
    tmp = tmp[tmp["__period"].notna()]
    if op in NUMERIC_AGGS and metric_col in tmp.columns:
        tmp[metric_col] = analytics_schema.coerce_numeric(tmp[metric_col])
    if group_by:
        if op in ("count", "nunique") and (metric_col is None or metric_col not in tmp.columns):
            table = tmp.groupby(["__period"] + group_by, dropna=False).size().reset_index(name="rows")
            label = "rows"
        else:
            table = tmp.groupby(["__period"] + group_by, dropna=False)[metric_col].agg(op).reset_index()
            label = f"{op}({metric_col})"
        table = table.sort_values("__period")
        chart = {"kind": "line", "x": "__period", "y": label, "series": group_by[0], "title": f"{label} over time"}
    else:
        if op in ("count", "nunique") and (metric_col is None or metric_col not in tmp.columns):
            table = tmp.groupby("__period").size().reset_index(name="rows")
            label = "rows"
        else:
            table = tmp.groupby("__period")[metric_col].agg(op).reset_index()
            label = f"{op}({metric_col})"
        table = table.sort_values("__period")
        chart = {"kind": "line", "x": "__period", "y": label, "title": f"{label} over time"}
    table = table.rename(columns={"__period": "period"}).reset_index(drop=True)
    chart["x"] = "period"
    return {"intent": "time_trend", "table": table, "scalar": None, "chart": chart, "notes": [], "meta": {}}


def _run_event_stages(work: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    event_col = plan.get("event_column")
    user = plan.get("user_id")
    if event_col not in work.columns:
        return _empty("event_stages", "No valid event column was provided.")
    label = "unique users"
    if user and user in work.columns:
        table = work.groupby(event_col, dropna=False)[user].nunique().reset_index()
    else:
        table = work.groupby(event_col, dropna=False).size().reset_index(name=label)
    table.columns = [event_col, label]
    table = table.sort_values(label, ascending=False).reset_index(drop=True)
    total = int(table[label].sum()) or 1
    table["share_of_stage_total_%"] = (table[label] / total * 100).round(2)
    return {
        "intent": "event_stages",
        "table": table,
        "scalar": None,
        "chart": {"kind": "bar", "x": event_col, "y": label, "title": f"{label} per event stage"},
        "notes": [
            "Counts are unique visitors per event, not conversion between ordered steps. "
            "A sequential funnel requires user ID, timestamps and stage order."
        ],
        "meta": {},
    }


def _run_funnel(work: pd.DataFrame, plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    user = plan.get("user_id")
    event_col = plan.get("event_column")
    ts_col = plan.get("timestamp")
    stages = plan.get("stages") or []
    if user not in work.columns or event_col not in work.columns or ts_col not in work.columns:
        return _empty("funnel", "A funnel requires user ID, event and timestamp columns.")
    tmp = work[[user, event_col, ts_col]].copy()
    tmp["__ts"] = _time_series(tmp, ts_col, schema)
    tmp = tmp.dropna(subset=[user, event_col, "__ts"])
    tmp = tmp.sort_values([user, "__ts"])

    first_users = int(tmp[user].nunique())
    counts: List[int] = []
    last_time: Dict[Any, Any] = {}
    for stage_index, stage in enumerate(stages):
        value = stage.get("event_value")
        stage_ts = tmp[tmp[event_col].astype(str) == str(value)]
        accepted = 0
        # Only visitors who completed the *immediately previous* stage remain
        # eligible. A visitor who misses a stage is dropped from the funnel and
        # can never be counted at a later stage (strictly sequential).
        new_last: Dict[Any, Any] = {}
        for uid, group in stage_ts.groupby(user):
            threshold = last_time.get(uid)
            if stage_index > 0 and threshold is None:
                continue
            times = group["__ts"]
            if threshold is None:
                earliest = times.min()
            else:
                after = times[times > threshold]
                if after.empty:
                    continue
                earliest = after.min()
            new_last[uid] = earliest
            accepted += 1
        last_time = new_last
        counts.append(accepted)

    table = pd.DataFrame(
        {
            "stage": [s.get("label") or s.get("event_value") for s in stages],
            "users": counts,
        }
    )
    base = counts[0] if counts else 0
    table["% of first stage"] = [round(c / base * 100, 2) if base else 0.0 for c in counts]
    table["step conversion %"] = [
        round(counts[i] / counts[i - 1] * 100, 2) if i > 0 and counts[i - 1] else (100.0 if i == 0 else 0.0)
        for i in range(len(counts))
    ]
    return {
        "intent": "funnel",
        "table": table,
        "scalar": None,
        "chart": {
            "kind": "funnel",
            "x": "stage",
            "y": "users",
            "title": "Ordered funnel (each stage after the previous, per user)",
        },
        "notes": [
            f"Started from {first_users} unique visitors in the filtered data.",
            "Each stage only counts users who reached it *after* the previous stage.",
        ],
        "meta": {},
    }


def _run_cohort(work: pd.DataFrame, plan: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    user = plan.get("user_id")
    ts_col = plan.get("timestamp")
    gran = (plan.get("time") or {}).get("period", "W")
    if user not in work.columns or ts_col not in work.columns:
        return _empty("cohort", "Cohort retention requires a user ID and a timestamp column.")
    # Delegate to the exact same core the standard report uses, so the AI answer
    # and the standard report can never disagree.
    report = standard_reports.cohort_retention(work, user, ts_col, schema, gran=gran)
    if not report.get("available"):
        return _empty("cohort", report.get("reason") or "Cohort retention is unavailable.")
    return {
        "intent": "cohort",
        "table": report["table"],
        "scalar": None,
        "chart": {"kind": "heatmap", "title": report["chart"]["title"]},
        "notes": list(report.get("notes") or []),
        "meta": {"granularity": gran},
    }


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    # Deliberately excludes min/max: on small samples those equal an individual
    # raw value, which we do not want leaving the process.
    return {
        "count": int(clean.count()),
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4) if clean.count() > 1 else 0.0,
    }


def _run_correlation(work: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    x, y = plan.get("x"), plan.get("y")
    if x not in work.columns or y not in work.columns:
        return _empty("correlation", "Both numeric columns are required for a correlation.")
    pair = work[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        return _empty("correlation", "Not enough numeric rows to compute a correlation.")
    value = float(pair[x].corr(pair[y]))
    # The raw observation pairs stay ONLY in ``table`` (used for the local
    # scatterplot and download). Only the aggregate summary below is ever
    # eligible to be forwarded for an explanation - never the pairs themselves.
    summary = {
        "method": "pearson",
        "correlation": round(value, 4),
        "valid_pairs": int(len(pair)),
        x: _numeric_summary(pair[x]),
        y: _numeric_summary(pair[y]),
    }
    return {
        "intent": "correlation",
        "table": pd.DataFrame({x: pair[x], y: pair[y]}).head(1000),
        "scalar": round(value, 4),
        "chart": {"kind": "scatter", "x": x, "y": y, "title": f"{y} vs {x}"},
        "notes": ["This is an association (Pearson correlation), NOT evidence of causation."],
        "meta": {"summary": summary},
    }
