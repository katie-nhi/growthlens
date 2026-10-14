"""Optional LLM client for GrowthLens "Ask GrowthLens AI".

This module is only ever used by the *personal inquiry* workflow. It follows
three hard rules:

1. **Never hard-code API keys.** Keys are read from Streamlit Secrets or the
   environment (``GROWTHLENS_LLM_API_KEY``), never from source.
2. **No raw data leaves the process without consent.** Callers must pass only a
   *sanitized* schema payload (see :func:`analytics_schema.public_payload`) or
   aggregated result digests, and must have obtained explicit user consent.
3. **No code execution.** The model returns a strict JSON *plan* which is then
   validated and executed locally by ``analytics_engine``. The model never
   runs Python/SQL/shell.

If no provider is configured, :func:`is_llm_configured` returns False and the UI
shows a clear "AI connection required" state - it never presents a rule-based
result as AI-generated.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:  # Streamlit is optional so this module can be unit-tested standalone.
    import streamlit as st
except Exception:  # noqa: BLE001
    st = None


class LLMError(RuntimeError):
    """Raised when the configured LLM call or its response cannot be used."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _secret(name: str) -> Optional[str]:
    if st is not None:
        try:
            if name in st.secrets:  # type: ignore[attr-defined]
                return str(st.secrets[name])
        except Exception:  # noqa: BLE001 - no secrets file configured
            pass
    return os.environ.get(name)


def is_llm_configured() -> bool:
    """True only when an API key is available via Secrets or environment."""
    return bool(_secret("GROWTHLENS_LLM_API_KEY"))


def provider_name() -> str:
    return _secret("GROWTHLENS_LLM_PROVIDER") or "openai-compatible"


def model_name() -> str:
    return _secret("GROWTHLENS_LLM_MODEL") or "gpt-4o-mini"


def _base_url() -> str:
    return _secret("GROWTHLENS_LLM_BASE_URL") or "https://api.openai.com/v1/chat/completions"


def _call_model(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    """Call the configured chat-completions endpoint and return message text.

    Kept deliberately small and swappable; tests monkeypatch this function.
    """
    key = _secret("GROWTHLENS_LLM_API_KEY")
    if not key:
        raise LLMError("AI connection required: no API key is configured.")
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        raise LLMError("The 'requests' package is required for LLM calls.") from exc

    payload = {"model": model_name(), "messages": messages, "temperature": temperature}
    try:
        response = requests.post(
            _base_url(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"LLM request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

PLAN_FIELDS = {
    "intent": "one of: groupby_aggregate, distinct_count, time_trend, event_stages, funnel, cohort, correlation, unsupported",
    "title": "short title for the analysis",
    "interpreted_question": "your restatement of what the user is asking",
    "metric": {"name": "human readable metric name", "operation": "one of: sum, mean, median, count, nunique, min, max", "column": "column name or null"},
    "group_by": ["column names to group by"],
    "filters": [{"column": "col", "op": "one of ==,!=,>,<,>=,<=,in,not_in,contains,between", "value": "literal"}],
    "time": {"column": "timestamp col or null", "period": "D or W or M"},
    "user_id": "identifier column or null",
    "event_column": "event/stage column or null",
    "timestamp": "timestamp column or null",
    "stages": [{"label": "step name", "event_value": "value in the event column"}],
    "x": "numeric column for correlation or null",
    "y": "numeric column for correlation or null",
    "required_columns": ["every column your plan references"],
    "reasoning": "why this plan answers the question",
    "limitations": ["known limitations of this analysis"],
    "reason": "only when intent is unsupported: why it cannot be answered",
}

_SYSTEM_PLAN = (
    "You are a careful analytics planner for a pandas-based tool. "
    "Given a natural-language business question and a sanitized dataset schema, "
    "produce EXACTLY ONE JSON analysis plan and nothing else (no prose, no code). "
    "Use only the columns listed in the schema. Never invent columns or values. "
    "Rules: distinguish event rows from unique visitors (use distinct_count or "
    "event_stages with a user id to count people); sequential funnels require a "
    "user id, a timestamp and ordered stages; cohort retention needs first "
    "observed activity; never claim causation from association; never plan a "
    "revenue calculation unless a numeric amount/currency column exists. "
    "If the question cannot be answered with the given columns, return "
    "intent=\"unsupported\" and explain in 'reason'. "
    "Return a JSON object with exactly these keys: "
    + json.dumps(PLAN_FIELDS)
)

_SYSTEM_EXPLAIN = (
    "You are a data analyst. Using ONLY the aggregated results provided, explain "
    "the findings for the user in 3-6 sentences, then list limitations. "
    "Do not invent or extrapolate numbers that are not in the provided results. "
    "State associations as associations, never as causation. If the results are "
    "empty or unsupported, say so plainly and list the missing fields."
)


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("The model did not return a JSON object.")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Could not parse the model's JSON plan: {exc}") from exc


def interpret_question(question: str, payload: Dict[str, Any], grain_hint: Optional[str] = None) -> Dict[str, Any]:
    """Ask the model to turn a question + sanitized schema into a structured plan."""
    if not is_llm_configured():
        raise LLMError("AI connection required: no API key is configured.")
    user_content = (
        f"Dataset schema (sanitized, no raw rows):\n{json.dumps(payload, indent=2)}\n\n"
        f"Confirmed row grain: {grain_hint or payload.get('grain_guess')}\n\n"
        f"Business question: {question}\n\n"
        "Return only the JSON plan."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PLAN},
        {"role": "user", "content": user_content},
    ]
    return _extract_json(_call_model(messages))


def explain_findings(
    question: str,
    plan: Dict[str, Any],
    results_digest: Dict[str, Any],
) -> str:
    """Ask the model to narrate the already-computed, aggregated results."""
    if not is_llm_configured():
        raise LLMError("AI connection required: no API key is configured.")
    user_content = (
        f"Question: {question}\n\n"
        f"Interpreted plan: {json.dumps(plan, default=str)}\n\n"
        f"Computed aggregated results (no raw rows):\n{json.dumps(results_digest, indent=2, default=str)}\n\n"
        "Explain the findings and limitations. Use only these numbers."
    )
    messages = [
        {"role": "system", "content": _SYSTEM_EXPLAIN},
        {"role": "user", "content": user_content},
    ]
    return _call_model(messages, temperature=0.2).strip()


# ---------------------------------------------------------------------------
# Backwards-compatible placeholder (Phase 1 blueprint mode)
# ---------------------------------------------------------------------------

def generate_recommendations_with_llm(*_args, **_kwargs):
    raise NotImplementedError(
        "Blueprint recommendations remain transparent and rule-based. "
        "The personal-inquiry workflow uses interpret_question/explain_findings."
    )
