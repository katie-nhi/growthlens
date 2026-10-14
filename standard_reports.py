"""Real, deterministic standard reports for the "Analyse My Data" page.

These replace the old "coming in Phase 2" placeholders. Everything is computed
with pandas on the GrowthLens Streamlit server from the uploaded file - nothing
is fabricated and no raw records are forwarded to any external AI/LLM API.

Four reports:

* Executive Overview        - genuine dataset metrics (events, visitors, items,
                              date span, event-type breakdown).
* Visitor-level funnel      - first view -> earliest cart strictly after that
                              view -> earliest transaction strictly after that
                              cart, per visitor (not global minimums).
* Weekly cohort retention   - unique visitor-week records, Monday-start UTC
                              weeks, first observed activity as the cohort, and
                              N/A for periods a cohort could not have observed yet.
* Browsing segmentation     - visitors bucketed by view count (1 / 2-4 / 5+)
                              with the ordered-funnel outcomes per bucket.

The module is pandas-only (no Streamlit / no Plotly), so it is unit-testable.
Callers receive plain ``DataFrame`` results plus a small chart *spec* they turn
into an interactive Plotly figure.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import analytics_schema

# Canonical funnel stages, in order, matched against event values.
_STAGE_PATTERNS: List[Tuple[str, str]] = [
    ("transaction", r"trans|purchase|buy|order|checkout|complete|paid"),
    ("cart", r"cart|basket|add.?to.?cart"),
    ("view", r"view|detail|browse"),
]
_STAGE_ORDER = ["view", "cart", "transaction"]
_STAGE_LABELS = {"view": "View", "cart": "Cart", "transaction": "Transaction"}


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _first(names: List[str], pattern: str) -> Optional[str]:
    for name in names:
        if re.search(pattern, name, re.IGNORECASE):
            return name
    return None


def detect_columns(df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Optional[str]]:
    ids = schema.get("identifiers", [])
    dims = schema.get("dimensions", [])
    events = schema.get("events", [])
    timestamps = schema.get("timestamps", [])
    currency = schema.get("currency", [])

    event_col = events[0] if events else _detect_event_column(df, schema)
    return {
        "visitor": _first(ids, r"visitor|user|customer|cust|member|client|session") or (ids[0] if ids else None),
        "item": _first(ids, r"item|product|article|sku|stock|goods"),
        "transaction": _first(ids, r"transaction|order|invoice|purchase"),
        "event": event_col,
        "timestamp": timestamps[0] if timestamps else None,
        "amount": currency[0] if currency else None,
        "channel": _first(dims, r"channel|source|medium|campaign|acquisition|attribution"),
        "device": _first(dims, r"device|platform|os|browser"),
    }


def _detect_event_column(df: pd.DataFrame, schema: Dict[str, Any]) -> Optional[str]:
    """Fallback: a text/dimension column whose values look like funnel events."""
    for col in schema.get("columns", []):
        if col["dtype"] not in ("text", "boolean") and col["role"] not in ("dimension", "event"):
            continue
        values = df[col["name"]].dropna().astype(str).unique()[:50]
        if any(re.search(pat, v, re.IGNORECASE) for v in values for _k, pat in _STAGE_PATTERNS):
            return col["name"]
    return None


def stage_values(df: pd.DataFrame, event_col: str) -> Dict[str, Any]:
    """Map canonical stages to the actual event values present."""
    values = df[event_col].dropna().astype(str).unique()
    mapping: Dict[str, Any] = {}
    for value in values:
        for key, pattern in _STAGE_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                mapping.setdefault(key, value)
                break
    return mapping


def _to_datetime(df: pd.DataFrame, col: str, schema: Dict[str, Any]) -> pd.Series:
    c = analytics_schema.column_by_name(schema, col)
    unit = c.get("epoch_unit") if c else None
    return analytics_schema.to_utc_naive(df[col], unit)


def _na(report: str, reason: str) -> Dict[str, Any]:
    return {"report": report, "available": False, "reason": reason, "metrics": [], "table": pd.DataFrame(), "chart": {"kind": "none"}, "notes": []}


# ---------------------------------------------------------------------------
# Executive overview
# ---------------------------------------------------------------------------

def executive_overview(df: pd.DataFrame, cols: Dict[str, Optional[str]], schema: Dict[str, Any]) -> Dict[str, Any]:
    metrics: List[Tuple[str, Any]] = [("Total events (rows)", f"{len(df):,}")]
    visitor = cols["visitor"]
    metrics.append(("Unique visitors", f"{int(df[visitor].nunique()):,}" if visitor else "n/a"))
    if cols["item"]:
        metrics.append(("Unique items", f"{int(df[cols['item']].nunique()):,}"))
    if cols["timestamp"]:
        ts = _to_datetime(df, cols["timestamp"], schema).dropna()
        if not ts.empty:
            metrics.append(("Date range", f"{ts.min():%Y-%m-%d} \u2192 {ts.max():%Y-%m-%d}"))
            metrics.append(("Span (days)", int((ts.max() - ts.min()).days) + 1))
    if visitor and len(df):
        metrics.append(("Events per visitor", round(len(df) / max(df[visitor].nunique(), 1), 2)))

    table = pd.DataFrame()
    chart: Dict[str, Any] = {"kind": "none"}
    if cols["event"]:
        ev = df[cols["event"]].dropna().astype(str)
        if visitor:
            grouped = df.dropna(subset=[cols["event"]]).groupby(cols["event"])
            table = grouped.agg(rows=(cols["event"], "size"), unique_visitors=(visitor, "nunique")).reset_index()
            table = table.rename(columns={cols["event"]: "event"})
        else:
            table = ev.value_counts().rename_axis("event").reset_index(name="rows")
        table["share_of_events_%"] = (table["rows"] / table["rows"].sum() * 100).round(2)
        table = table.sort_values("rows", ascending=False).reset_index(drop=True)
        chart = {"kind": "bar", "x": "event", "y": "rows", "title": "Events by type"}
    return {"report": "overview", "available": True, "reason": "", "metrics": metrics, "table": table, "chart": chart, "notes": []}


# ---------------------------------------------------------------------------
# Sequential funnel
# ---------------------------------------------------------------------------

def sequential_funnel(df: pd.DataFrame, cols: Dict[str, Optional[str]], schema: Dict[str, Any]) -> Dict[str, Any]:
    visitor, event_col, ts_col = cols["visitor"], cols["event"], cols["timestamp"]
    if not (visitor and event_col and ts_col):
        return _na("funnel", "A visitor id, event column and timestamp are required.")
    mapping = stage_values(df, event_col)
    present = [s for s in _STAGE_ORDER if s in mapping]
    if len(present) < 2:
        return _na("funnel", "Fewer than two recognised funnel stages were found in the event column.")

    ev = df[[visitor, event_col, ts_col]].copy()
    ev["__ts"] = _to_datetime(ev, ts_col, schema)
    ev = ev.dropna(subset=[visitor, event_col, "__ts"])

    view_value = mapping.get("view")
    view_times = ev[ev[event_col] == view_value].groupby(visitor)["__ts"].min() if view_value else pd.Series(dtype="datetime64[ns]")

    cart_times: Dict[Any, Any] = {}
    if "cart" in present and view_value:
        carts = ev[ev[event_col] == mapping["cart"]].merge(
            view_times.rename("__view_ts"), left_on=visitor, right_index=True, how="inner"
        )
        carts = carts[carts["__ts"] > carts["__view_ts"]]
        cart_times = carts.groupby(visitor)["__ts"].min()

    trx_times: Dict[Any, Any] = {}
    if "transaction" in present and len(cart_times):
        trx = ev[ev[event_col] == mapping["transaction"]].merge(
            pd.Series(cart_times).rename("__cart_ts"), left_on=visitor, right_index=True, how="inner"
        )
        trx = trx[trx["__ts"] > trx["__cart_ts"]]
        trx_times = trx.groupby(visitor)["__ts"].min()

    reached = {
        "view": int(view_times.shape[0]),
        "cart": int(len(cart_times)),
        "transaction": int(len(trx_times)),
    }
    table = pd.DataFrame(
        {
            "stage": [_STAGE_LABELS[s] for s in present],
            "visitors": [reached[s] for s in present],
        }
    )
    base = reached[present[0]] if present else 0
    table["% of first stage"] = [round(v / base * 100, 2) if base else 0.0 for v in table["visitors"]]
    table["step conversion %"] = [
        round(table["visitors"][i] / table["visitors"][i - 1] * 100, 2) if i > 0 and table["visitors"][i - 1] else (100.0 if i == 0 else 0.0)
        for i in range(len(table))
    ]
    notes = [
        "Each visitor is counted at most once per stage.",
        "A later stage only counts visitors who reached it strictly AFTER the previous stage for that same visitor.",
    ]
    return {
        "report": "funnel",
        "available": True,
        "reason": "",
        "metrics": [(f"{_STAGE_LABELS[s]} reached", f"{reached[s]:,}") for s in present],
        "table": table,
        "chart": {"kind": "funnel", "x": "stage", "y": "visitors", "title": "Visitor-level sequential funnel"},
        "notes": notes,
        "_reached": reached,
        "_present": present,
        "_cart_visitors": set(cart_times.index) if len(cart_times) else set(),
        "_trx_visitors": set(trx_times.index) if len(trx_times) else set(),
        "_view_counts": ev[ev[event_col] == view_value].groupby(visitor).size() if view_value else pd.Series(dtype="int64"),
    }


# ---------------------------------------------------------------------------
# Weekly cohort retention
# ---------------------------------------------------------------------------

def weekly_cohort(df: pd.DataFrame, cols: Dict[str, Optional[str]], schema: Dict[str, Any]) -> Dict[str, Any]:
    visitor, ts_col = cols["visitor"], cols["timestamp"]
    if not (visitor and ts_col):
        return _na("cohort", "A visitor id and timestamp are required for cohort retention.")
    work = df[[visitor, ts_col]].copy()
    work["__ts"] = _to_datetime(work, ts_col, schema)
    work = work.dropna(subset=[visitor, "__ts"])
    if work.empty:
        return _na("cohort", "No valid timestamped rows were found.")

    # Monday-start UTC week (shared helper, so active/cohort weeks cannot drift).
    c = analytics_schema.column_by_name(schema, ts_col)
    work["__week"] = analytics_schema.utc_week_start(work[ts_col], c.get("epoch_unit") if c else None)

    # Unique visitor-week records (a visitor active many times in a week counts once).
    visitor_week = work[[visitor, "__week"]].drop_duplicates()
    first_week = visitor_week.groupby(visitor)["__week"].min()
    visitor_week = visitor_week.assign(__cohort=visitor_week[visitor].map(first_week))
    visitor_week["__index"] = ((visitor_week["__week"] - visitor_week["__cohort"]).dt.days // 7).astype(int)

    pivot = visitor_week.groupby(["__cohort", "__index"]).size().unstack(fill_value=0).sort_index()
    last_week = work["__week"].max()
    last_ts = work["__ts"].max()
    # Include every observable week index, so an observable week in which nobody
    # returned is a real 0% (not a missing column). Unobservable cells become N/A below.
    max_col = int((last_week - pivot.index.min()).days // 7) if len(pivot.index) else 0
    pivot = pivot.reindex(columns=range(0, max_col + 1), fill_value=0)
    sizes = pivot[0]
    retention = pivot.div(sizes.replace(0, pd.NA), axis=0) * 100

    for cohort_start in retention.index:
        max_index = int((last_week - cohort_start).days // 7)
        future = [c for c in retention.columns if c > max_index]
        if future:
            retention.loc[cohort_start, future] = pd.NA
        # The dataset ends mid-week, so its last week is only partially observed. A
        # 0% there is ambiguous (a visitor could still return later that week), so it
        # must be N/A. A non-zero cell is a real observed return and is kept.
        if max_index in retention.columns:
            week_start = cohort_start + pd.Timedelta(days=7 * max_index)
            fully_observed = last_ts >= week_start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)
            if not fully_observed and retention.at[cohort_start, max_index] == 0:
                retention.at[cohort_start, max_index] = pd.NA

    retention = retention.round(1)
    retention.columns = [f"W{int(c)}" for c in retention.columns]
    table = retention.reset_index().rename(columns={"__cohort": "cohort_start"})
    table.insert(1, "cohort_size", sizes.reindex(retention.index).astype("Int64").values)
    table["cohort_start"] = pd.to_datetime(table["cohort_start"]).dt.strftime("%Y-%m-%d")

    notes = [
        "Weeks start on Monday (UTC). One visitor counts at most once per week (unique visitor-week records).",
        "Cohort size is the number of unique visitors first observed in that week.",
        "Blank (N/A) cells are weeks a cohort could not have observed yet, or the dataset's "
        "final partial week where a 0% would be ambiguous - they are not zero.",
    ]
    return {
        "report": "cohort",
        "available": True,
        "reason": "",
        "metrics": [
            ("Cohorts", int(len(sizes))),
            ("Weeks observed", int(retention.shape[1])),
            ("Unique visitor-week records", int(len(visitor_week))),
        ],
        "table": table,
        "chart": {"kind": "heatmap", "title": "Weekly cohort retention % (N/A = not yet observable)"},
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Browsing segmentation
# ---------------------------------------------------------------------------

def cohort_heatmap_labels(table: pd.DataFrame) -> List[str]:
    """Y-axis labels for the cohort heatmap.

    Derived from exactly the ``cohort_start`` values shown in the output table,
    rendered as an explicit non-date string (``"05 May 2025"``) so Plotly cannot
    treat the axis as a date and shift it into the viewer's timezone.
    """
    return [pd.Timestamp(value).strftime("%d %b %Y") for value in table["cohort_start"]]


def browsing_segmentation(funnel: Dict[str, Any]) -> Dict[str, Any]:
    if not funnel.get("available"):
        return _na("segmentation", "Segmentation reuses the ordered funnel, which is unavailable.")
    view_counts: pd.Series = funnel.get("_view_counts")
    if view_counts is None or view_counts.empty:
        return _na("segmentation", "No view events were found to segment visitors.")
    cart_visitors = funnel.get("_cart_visitors", set())
    trx_visitors = funnel.get("_trx_visitors", set())

    def bucket(n: int) -> str:
        if n <= 1:
            return "1 view"
        if n <= 4:
            return "2-4 views"
        return "5+ views"

    labels = ["1 view", "2-4 views", "5+ views"]
    buckets = view_counts.map(bucket)
    rows = []
    for label in labels:
        members = buckets[buckets == label].index
        if len(members) == 0:
            rows.append({"segment": label, "visitors": 0, "reached_cart": 0, "reached_transaction": 0,
                         "cart_rate_%": 0.0, "transaction_rate_%": 0.0})
            continue
        reached_cart = len(set(members) & cart_visitors)
        reached_trx = len(set(members) & trx_visitors)
        rows.append({
            "segment": label,
            "visitors": int(len(members)),
            "reached_cart": reached_cart,
            "reached_transaction": reached_trx,
            "cart_rate_%": round(reached_cart / len(members) * 100, 2),
            "transaction_rate_%": round(reached_trx / len(members) * 100, 2),
        })
    table = pd.DataFrame(rows)
    return {
        "report": "segmentation",
        "available": True,
        "reason": "",
        "metrics": [
            ("Segments", len(table)),
            ("Most visitors in", table.loc[table["visitors"].idxmax(), "segment"] if table["visitors"].max() else "n/a"),
            ("Best transaction rate", table.loc[table["transaction_rate_%"].idxmax(), "segment"] if table["visitors"].max() else "n/a"),
        ],
        "table": table,
        "chart": {
            "kind": "grouped_bar",
            "x": "segment",
            "series": ["visitors", "reached_cart", "reached_transaction"],
            "title": "Visitors and funnel outcomes by browsing segment",
        },
        "notes": [
            "Segments are defined by the visitor's number of view events: 1, 2-4, or 5+.",
            "reached_cart / reached_transaction use the same ordered-funnel outcomes (each stage after the previous).",
        ],
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(df: pd.DataFrame, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    schema = schema or analytics_schema.infer_schema(df)
    cols = detect_columns(df, schema)
    funnel = sequential_funnel(df, cols, schema)
    return {
        "overview": executive_overview(df, cols, schema),
        "funnel": funnel,
        "cohort": weekly_cohort(df, cols, schema),
        "segmentation": browsing_segmentation(funnel),
    }
