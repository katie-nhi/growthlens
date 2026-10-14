"""Fifth section for the "Analyse My Data" page:

    Personal Inquiry Analysis - Ask GrowthLens AI

This is a real analysis workflow, not a chatbot. The optional LLM only
*interprets* the question into a strict JSON plan and *explains* the already
computed results. All metrics are computed with pandas on the GrowthLens
Streamlit server in ``analytics_engine``; the model never runs code and never
sees raw rows.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics_engine
import analytics_schema
import llm

ROLE_OPTIONS = [
    "identifier",
    "timestamp",
    "event",
    "dimension",
    "measure",
    "currency",
    "text",
]
GRAIN_OPTIONS = [
    "one row per user event (event log)",
    "one row per user per period (activity/appearance)",
    "one row per user (snapshot)",
    "one row per period (aggregated time series)",
    "one row per record (unknown grain)",
]


# ---------------------------------------------------------------------------
# Example questions (adapt to the columns that are actually available)
# ---------------------------------------------------------------------------

def _visitor_id(schema: Dict[str, Any]) -> Optional[str]:
    """Pick the most likely visitor/user identifier column."""
    ids = schema.get("identifiers", [])
    if not ids:
        return None
    for name in ids:
        if _re_name(name, r"visitor|user|customer|cust|member|client|session"):
            return name
    return ids[0]


def _re_name(name: str, pattern: str) -> bool:
    import re

    return bool(re.search(pattern, name, re.IGNORECASE))


def _channel_column(dims: List[str]) -> Optional[str]:
    for name in dims:
        if _re_name(name, r"channel|source|medium|campaign|acquisition|attribution"):
            return name
    return None


def _channel_attribution_kind(df: Optional[pd.DataFrame], visitor: Optional[str], channel: str) -> str:
    """Return 'acquisition' if the channel looks constant per visitor, else 'event'."""
    if df is None or visitor is None or channel not in df.columns or visitor not in df.columns:
        return "event"
    sample = df[[visitor, channel]].dropna()
    if len(sample) > 200_000:
        sample = sample.sample(200_000, random_state=0)
    try:
        per_visitor = sample.groupby(visitor)[channel].nunique(dropna=True)
        if per_visitor.empty:
            return "event"
        return "acquisition" if float((per_visitor <= 1).mean()) >= 0.99 else "event"
    except Exception:  # noqa: BLE001
        return "event"


def _is_identifierish_name(schema: Dict[str, Any], name: str) -> bool:
    for c in schema.get("columns", []):
        if c["name"] == name:
            return bool(c.get("is_identifierish") or c.get("role") == "identifier")
    return False


def suggest_examples(schema: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> List[str]:
    """Adaptive example questions that never aggregate identifier columns.

    Rules enforced here:
    * Identifiers (visitorid, itemid, ...) are only ever requested as a
      DISTINCT COUNT of visitors - never summed or averaged.
    * Average order value is only suggested when a real amount/currency column
      exists (and the plan later uses its non-null transaction amounts).
    * A channel column that varies within a visitor is described as an
      *event-level* channel, not as customer acquisition attribution.
    """
    dims = list(schema.get("dimensions", []))
    currency = list(schema.get("currency", []))
    measures = [m for m in schema.get("measures", []) if not _is_identifierish_name(schema, m)]
    events = schema.get("events", [])
    timestamps = schema.get("timestamps", [])
    visitor = _visitor_id(schema)

    examples: List[str] = []
    if visitor:
        examples.append(f"How many unique visitors are in the dataset (distinct count of {visitor})?")
        if events:
            examples.append(f"How many unique visitors reach each event stage ({visitor})?")
    if events and visitor:
        channel = _channel_column(dims)
        if channel:
            kind = _channel_attribution_kind(df, visitor, channel)
            if kind == "acquisition":
                examples.append(f"Which acquisition channel ({channel}) brings the most unique visitors?")
            else:
                examples.append(
                    f"Do results differ across the event-level {channel} field? "
                    f"(event-level, not customer acquisition attribution)"
                )
    if timestamps and visitor:
        examples.append("What are the weekly behaviour patterns of visitors?")
    if currency:
        examples.append(f"What is the average order value for completed transactions ({currency[0]}, non-null amounts only)?")
    if timestamps and visitor:
        examples.append("How does visitor retention change week over week?")
    safe_values = currency or measures
    if dims and safe_values:
        examples.append(f"Compare average {safe_values[0]} across {dims[0]} segments.")
    return examples[:5]


# ---------------------------------------------------------------------------
# Schema inspection / clarification
# ---------------------------------------------------------------------------

def _schema_frame(schema: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": c["name"],
                "role": c["role"] if c["role"] in ROLE_OPTIONS else "text",
                "type": c["dtype"],
                "distinct": c["nunique"],
            }
            for c in schema["columns"]
        ]
    )


def _apply_role_overrides(schema: Dict[str, Any], edited: pd.DataFrame) -> Dict[str, Any]:
    override = {row["column"]: row["role"] for _, row in edited.iterrows()}
    for col in schema["columns"]:
        if col["name"] in override and override[col["name"]] in ROLE_OPTIONS:
            col["role"] = override[col["name"]]
    schema["dimensions"] = [c["name"] for c in schema["columns"] if c["role"] in ("dimension", "event") and not c["is_identifierish"]]
    schema["measures"] = [c["name"] for c in schema["columns"] if c["role"] in ("measure", "currency")]
    schema["identifiers"] = [c["name"] for c in schema["columns"] if c["role"] == "identifier" or c["is_identifierish"]]
    schema["timestamps"] = [c["name"] for c in schema["columns"] if c["role"] == "timestamp"]
    schema["events"] = [c["name"] for c in schema["columns"] if c["role"] == "event"]
    schema["currency"] = [c["name"] for c in schema["columns"] if c["role"] == "currency"]
    return schema


def render_schema_clarifier(schema: Dict[str, Any]) -> Dict[str, Any]:
    with st.expander("Detected schema - inspect or clarify column roles", expanded=False):
        st.caption(
            "GrowthLens inferred these roles. Correct anything that looks wrong "
            "before asking a question; this affects how the question is interpreted."
        )
        edited = st.data_editor(
            _schema_frame(schema),
            hide_index=True,
            width="stretch",
            disabled=["column", "type", "distinct"],
            column_config={
                "role": st.column_config.SelectboxColumn("role", options=ROLE_OPTIONS, required=True),
            },
            key="inquiry_schema_editor",
        )
        grain = st.selectbox(
            "Row grain (what does one row represent?)",
            GRAIN_OPTIONS,
            index=GRAIN_OPTIONS.index(schema["grain_guess"]) if schema["grain_guess"] in GRAIN_OPTIONS else len(GRAIN_OPTIONS) - 1,
            key="inquiry_grain",
        )
        schema = _apply_role_overrides(schema, edited)
        schema["grain_guess"] = grain
    return schema


# ---------------------------------------------------------------------------
# Chart + result rendering
# ---------------------------------------------------------------------------

def _make_chart(result: Dict[str, Any]) -> Optional[Any]:
    spec = result.get("chart") or {}
    kind = spec.get("kind")
    table = result.get("table")
    if kind in (None, "none") or table is None or table.empty:
        return None
    try:
        if kind == "bar":
            x = spec.get("x")
            x = x[0] if isinstance(x, list) and len(x) == 1 else x
            return px.bar(table, x=x, y=spec.get("y"), title=spec.get("title"))
        if kind == "line":
            return px.line(table, x=spec.get("x"), y=spec.get("y"), color=spec.get("series"), title=spec.get("title"))
        if kind == "funnel":
            return px.funnel(table, x=spec.get("y"), y=spec.get("x"), title=spec.get("title"))
        if kind == "scatter":
            return px.scatter(table, x=spec.get("x"), y=spec.get("y"), title=spec.get("title"))
        if kind == "heatmap":
            matrix = table.copy()
            drop = [c for c in ("cohort_start", "cohort_size") if c in matrix.columns]
            index_col = "cohort_start" if "cohort_start" in matrix.columns else matrix.columns[0]
            matrix = matrix.set_index(index_col).drop(columns=[c for c in drop if c != index_col], errors="ignore")
            matrix = matrix.apply(pd.to_numeric, errors="coerce")
            return px.imshow(matrix, text_auto=True, aspect="auto", title=spec.get("title"), color_continuous_scale="Blues")
    except Exception:  # noqa: BLE001 - a chart failure must not break the page
        return None
    return None


def _plan_definitions(plan: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    metric = plan.get("metric") or {}
    if metric:
        rows.append({"item": "Metric", "definition": f"{metric.get('operation', 'count')} of {metric.get('column') or 'rows'}"
                     + (f" ({metric.get('name')})" if metric.get("name") else "")})
    if plan.get("group_by"):
        rows.append({"item": "Group by", "definition": ", ".join(plan["group_by"])})
    if plan.get("filters"):
        rows.append({"item": "Filters", "definition": "; ".join(
            f"{f.get('column')} {f.get('op')} {f.get('value')}" for f in plan["filters"])})
    if plan.get("time"):
        rows.append({"item": "Time", "definition": f"{plan['time'].get('period', 'W')} by {plan['time'].get('column')}"})
    if plan.get("user_id"):
        rows.append({"item": "Unique visitor id", "definition": plan["user_id"]})
    if plan.get("stages"):
        rows.append({"item": "Ordered stages", "definition": " -> ".join(
            f"{s.get('label')} ({s.get('event_value')})" for s in plan["stages"])})
    if plan.get("x") and plan.get("y"):
        rows.append({"item": "Association", "definition": f"{plan['y']} vs {plan['x']}"})
    return pd.DataFrame(rows)


def _digest(result: Dict[str, Any]) -> Dict[str, Any]:
    table = result.get("table")
    preview = []
    if table is not None and not table.empty:
        preview = json.loads(table.head(25).to_json(orient="records", date_format="iso"))
    return {
        "intent": result.get("intent"),
        "scalar": result.get("scalar"),
        "row_count": 0 if table is None else int(len(table)),
        "table_preview": preview,
        "notes": result.get("notes", []),
    }


def _render_result(bundle: Dict[str, Any]) -> None:
    plan = bundle["plan"]
    result = bundle["result"]

    st.markdown("**Interpreted question**")
    st.write(plan.get("interpreted_question") or bundle["question"])

    defs = _plan_definitions(plan)
    if not defs.empty:
        st.markdown("**Proposed metric definitions**")
        st.dataframe(defs, hide_index=True, width="stretch")

    st.markdown("**Executed results**")
    if result.get("scalar") is not None and (result.get("table") is None or result["table"].empty):
        st.metric("Result", result["scalar"])
    elif result.get("scalar") is not None and result.get("intent") == "correlation":
        st.metric("Pearson correlation", result["scalar"])
    table = result.get("table")
    if table is not None and not table.empty:
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download analysis result (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="growthlens_analysis.csv",
            mime="text/csv",
            key="inquiry_download",
        )
    else:
        st.info(result.get("notes", ["No rows were returned."])[0] if result.get("notes") else "No rows were returned.")

    fig = _make_chart(result)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Business interpretation (AI, from computed results - no invented numbers)**")
    st.write(bundle.get("explanation") or "No interpretation was returned.")

    limitations = list(plan.get("limitations") or []) + list(result.get("notes") or [])
    if limitations:
        st.markdown("**Limitations**")
        for item in limitations:
            st.write(f"- {item}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(df: pd.DataFrame, schema: Dict[str, Any], question: str) -> Dict[str, Any]:
    payload = analytics_schema.public_payload(df, schema)
    plan = llm.interpret_question(question, payload, grain_hint=schema.get("grain_guess"))

    ok, errors, missing = analytics_engine.validate_plan(plan, schema)
    unsupported = plan.get("intent") == "unsupported" or not ok

    if unsupported:
        result = analytics_engine.execute_plan(df, plan, schema)
        reason = plan.get("reason") or ""
        if missing:
            reason = (reason + " Missing fields: " + ", ".join(missing) + ".").strip()
        if errors:
            reason = (reason + " " + " ".join(errors)).strip()
        if not reason:
            reason = "This question cannot be answered from the uploaded data."
        result = dict(result)
        result["notes"] = [reason] + result.get("notes", [])
    else:
        result = analytics_engine.execute_plan(df, plan, schema)

    explanation = ""
    try:
        explanation = llm.explain_findings(question, plan, _digest(result))
    except llm.LLMError as exc:
        explanation = f"(Interpretation unavailable: {exc})"

    return {"question": question, "plan": plan, "result": result, "explanation": explanation, "unsupported": unsupported}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_inquiry(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Personal Inquiry Analysis - Ask GrowthLens AI")
    st.caption(
        "Ask a custom business question about this dataset. The file is processed by "
        "the GrowthLens app server. GrowthLens does not send raw rows or visitor IDs "
        "to the AI provider - only a sanitized schema and the aggregated results, and "
        "only after you give consent."
    )

    schema = analytics_schema.infer_schema(df)
    schema = render_schema_clarifier(schema)

    examples = suggest_examples(schema, df)
    if examples:
        st.markdown("**Example questions (adapt to your columns):**")
        cols = st.columns(min(len(examples), 3))
        for i, example in enumerate(examples):
            if cols[i % len(cols)].button(example, key=f"inquiry_example_{i}"):
                st.session_state["inquiry_question"] = example

    question = st.text_area(
        "Your question",
        key="inquiry_question",
        placeholder="e.g. Which acquisition channel converts best by device?",
        height=80,
    )

    if not llm.is_llm_configured():
        st.info(
            "AI connection is not configured, so custom natural-language analysis "
            "is disabled in this build. The example questions above show what will "
            "be available. No rule-based result is presented as AI-generated."
        )
        st.button("Analyse with AI", disabled=True, key="inquiry_run_disabled")
        return

    consent = st.checkbox(
        "I consent to sending my question, the sanitized schema (column names, "
        "types, roles and low-cardinality category labels) and the aggregated "
        "results to the configured AI provider ("
        + llm.provider_name()
        + "). No raw rows or visitor IDs are sent.",
        key="inquiry_consent",
    )
    run = st.button("Analyse with AI", type="primary", key="inquiry_run")

    if run:
        if not question.strip():
            st.warning("Please type a question first.")
        elif not consent:
            st.warning("Please tick the consent box to send the sanitized request to the AI provider.")
        else:
            with st.spinner("Interpreting your question and computing results..."):
                try:
                    st.session_state["inquiry_result"] = _run_pipeline(df, schema, question.strip())
                except llm.LLMError as exc:
                    st.session_state.pop("inquiry_result", None)
                    st.error(f"AI request failed: {exc}")

    bundle = st.session_state.get("inquiry_result")
    if bundle:
        st.markdown("### Analysis")
        _render_result(bundle)
