"""Tests for the personal-inquiry example suggestions (no network, no LLM).

These import ``inquiry`` (which imports Streamlit/Plotly). If those are not
installed, the tests are skipped so the rest of the suite still runs.

Run directly:  python tests/test_inquiry_suggestions.py
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

try:
    import inquiry  # noqa: E402
    import analytics_schema  # noqa: E402
    _READY = True
except Exception:  # noqa: BLE001
    _READY = False


def _df(channel_constant_per_visitor=True):
    rows = [
        # visitor, event, ts, item, amount, channel
        (1, "view", "2024-01-01", 10, None, "social"),
        (1, "addtocart", "2024-01-02", 10, None, "social" if channel_constant_per_visitor else "email"),
        (1, "transaction", "2024-01-03", 10, "25", "social"),
        (2, "view", "2024-01-01", 11, None, "paid_search"),
        (2, "view", "2024-01-02", 12, None, "paid_search"),
    ]
    df = pd.DataFrame(rows, columns=["visitorid", "event", "timestamp", "itemid", "order_value_sgd", "channel"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_suggests_distinct_visitors_not_identifier_sums():
    if not _READY:
        return
    df = _df()
    examples = inquiry.suggest_examples(analytics_schema.infer_schema(df), df)
    blob = " ".join(examples)
    assert re.search(r"unique visitors|distinct count", blob, re.IGNORECASE), blob
    # never propose summing/averaging an identifier
    assert not re.search(r"(sum|average|mean|total)[^.]*(visitorid|itemid)", blob, re.IGNORECASE), blob


def test_average_order_value_only_with_amount_column():
    if not _READY:
        return
    df = _df()
    with_amount = " ".join(inquiry.suggest_examples(analytics_schema.infer_schema(df), df)).lower()
    assert "average order value" in with_amount

    no_amount = df.drop(columns=["order_value_sgd"])
    without = " ".join(inquiry.suggest_examples(analytics_schema.infer_schema(no_amount), no_amount)).lower()
    assert "average order value" not in without


def test_channel_attribution_vs_event_level():
    if not _READY:
        return
    # channel constant per visitor -> acquisition attribution
    acquisition = " ".join(
        inquiry.suggest_examples(analytics_schema.infer_schema(_df(True)), _df(True))
    ).lower()
    assert "acquisition channel" in acquisition

    # channel varies within a visitor -> event-level, not acquisition
    event_level = " ".join(
        inquiry.suggest_examples(analytics_schema.infer_schema(_df(False)), _df(False))
    ).lower()
    assert "event-level" in event_level
    assert "acquisition channel" not in event_level


def test_digest_redacts_identifier_and_pii_grouping_columns():
    if not _READY:
        return
    df = pd.DataFrame(
        {
            "visitorid": ["VISITOR_SECRET_1", "VISITOR_SECRET_1", "VISITOR_SECRET_2"],
            "email": ["alice@example.com", "alice@example.com", "bob@example.com"],
        }
    )
    schema = analytics_schema.infer_schema(df)
    for group in ("visitorid", "email"):
        result = {
            "intent": "groupby_aggregate",
            "table": pd.DataFrame({group: sorted(df[group].unique()), "rows": [2, 1]}),
            "scalar": None,
            "notes": [],
        }
        digest = inquiry._digest(result, schema)
        blob = str(digest)
        assert "VISITOR_SECRET_1" not in blob
        assert "alice@example.com" not in blob
        assert group in digest["redacted_columns"]


def test_digest_keeps_safe_dimension_values():
    if not _READY:
        return
    df = _df()
    schema = analytics_schema.infer_schema(df)
    result = {
        "intent": "groupby_aggregate",
        "table": pd.DataFrame({"channel": ["social", "paid_search"], "rows": [3, 2]}),
        "scalar": None,
        "notes": [],
    }
    digest = inquiry._digest(result, schema)
    assert digest["redacted_columns"] == []
    assert {r["channel"] for r in digest["table_preview"]} == {"social", "paid_search"}


def test_plan_definitions_hide_empty_time():
    if not _READY:
        return
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"}, "time": {"column": None, "period": "W"}}
    defs = inquiry._plan_definitions(plan)
    assert "Time" not in set(defs["item"])
    # no definition should contain the literal string "None"
    assert not any("None" in d for d in defs["definition"])


def test_plan_definitions_show_populated_time():
    if not _READY:
        return
    plan = {"intent": "time_trend", "metric": {"operation": "count"}, "time": {"column": "timestamp", "period": "W"}}
    defs = inquiry._plan_definitions(plan).set_index("item")["definition"]
    assert defs["Time"] == "W by timestamp"


def test_method_notes_are_auditable():
    if not _READY:
        return
    df = _df()
    schema = analytics_schema.infer_schema(df)
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "order_value_sgd"},
            "group_by": ["channel"], "filters": [{"column": "channel", "op": "==", "value": "social"}]}
    result = {"intent": "groupby_aggregate", "table": pd.DataFrame({"channel": ["social"], "sum": [25.0]}), "notes": []}
    notes = " ".join(inquiry._method_notes(plan, result, schema)).lower()
    assert "method:" in notes
    assert "numerator/denominator" in notes
    assert "filters applied" in notes


def test_field_availability_classifies_columns():
    if not _READY:
        return
    df = _df()
    schema = analytics_schema.infer_schema(df)
    plan = {"intent": "distinct_count", "user_id": "visitorid", "metric": {"operation": "nunique", "column": "visitorid"}}
    fa = inquiry._field_availability(plan, schema)
    assert "visitorid" in fa["used_columns"]
    assert "visitorid" in fa["uncertain_identifier_columns"]
    assert "event" in fa["available_unused"]


def test_pipeline_never_sends_identifier_values_to_explanation():
    if not _READY:
        return
    import llm as llm_mod

    df = pd.DataFrame(
        {
            "visitorid": ["VISITOR_SECRET_1", "VISITOR_SECRET_1", "VISITOR_SECRET_2"],
            "email": ["alice@example.com", "alice@example.com", "bob@example.com"],
            "event": ["view", "view", "cart"],
        }
    )
    schema = analytics_schema.infer_schema(df)
    captured = {}

    orig_secret, orig_call = llm_mod._secret, llm_mod._call_model
    try:
        llm_mod._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"}, "group_by": ["visitorid"]}

        def fake_call(messages, temperature=0.0):
            joined = " ".join(m["content"] for m in messages)
            if "plan" in joined.lower() and "return only the json plan" in joined.lower():
                import json as _json

                return _json.dumps(plan)
            captured["explain"] = joined
            return "done"

        llm_mod._call_model = fake_call
        llm_mod.reset_limits()
        bundle = inquiry._run_pipeline(df, schema, "break down by visitor")
        assert "VISITOR_SECRET_1" not in captured["explain"]
        assert "alice@example.com" not in captured["explain"]
        assert bundle["result"]["intent"] == "groupby_aggregate"
    finally:
        llm_mod._secret, llm_mod._call_model = orig_secret, orig_call
        llm_mod.reset_limits()


# ---------------------------------------------------------------------------
# Privacy regressions - correlation digest and invalid-filter pipeline
# ---------------------------------------------------------------------------

def test_digest_for_correlation_omits_raw_pairs():
    if not _READY:
        return
    df = pd.DataFrame({
        "a": [111.111, 222.222, 333.333, 444.444, 555.555],
        "b": [0.987654, 1.876543, 2.765432, 3.654321, 4.543210],
    })
    schema = analytics_schema.infer_schema(df)
    import analytics_engine as engine

    result = engine.execute_plan(df, {"intent": "correlation", "x": "a", "y": "b"}, schema)
    digest = inquiry._digest(result, schema)
    blob = str(digest)
    for token in ("111.111", "555.555", "0.987654", "4.54321"):
        assert token not in blob, token
    assert digest["table_preview"] == []
    assert digest["correlation_summary"]["valid_pairs"] == 5


def test_pipeline_invalid_filter_never_sends_literal():
    if not _READY:
        return
    import llm as llm_mod

    SENTINEL = "alice@example.com"
    df = pd.DataFrame({"visitorid": ["v1", "v1", "v2"], "amount": [1.0, 2.0, 3.0]})
    schema = analytics_schema.infer_schema(df)
    captured = {}
    calls = {"explain": 0}

    orig_secret, orig_call = llm_mod._secret, llm_mod._call_model
    try:
        llm_mod._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
                "filters": [{"column": "amount", "op": ">", "value": SENTINEL}]}

        def fake_call(messages, temperature=0.0):
            joined = " ".join(m["content"] for m in messages)
            captured["all"] = captured.get("all", "") + " " + joined
            if "return only the json plan" in joined.lower():
                import json as _json

                return _json.dumps(plan)
            calls["explain"] += 1
            return "should not be reached"

        llm_mod._call_model = fake_call
        llm_mod.reset_limits()
        bundle = inquiry._run_pipeline(df, schema, "count where amount is high")
        assert SENTINEL not in captured.get("all", "")
        assert calls["explain"] == 0, "invalid plan must not trigger an explanation call"
        assert bundle["unsupported"] is True
    finally:
        llm_mod._secret, llm_mod._call_model = orig_secret, orig_call
        llm_mod.reset_limits()


def test_pipeline_correlation_never_sends_raw_pairs():
    if not _READY:
        return
    import llm as llm_mod

    df = pd.DataFrame({
        "visitorid": ["v1", "v2", "v3", "v4", "v5"],
        "amount": [111.111, 222.222, 333.333, 444.444, 555.555],
        "score": [0.987654, 1.876543, 2.765432, 3.654321, 4.543210],
    })
    schema = analytics_schema.infer_schema(df)
    captured = {"all": ""}

    orig_secret, orig_call = llm_mod._secret, llm_mod._call_model
    try:
        llm_mod._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        plan = {"intent": "correlation", "x": "amount", "y": "score"}

        def fake_call(messages, temperature=0.0):
            joined = " ".join(m["content"] for m in messages)
            captured["all"] += " " + joined
            if "return only the json plan" in joined.lower():
                import json as _json

                return _json.dumps(plan)
            return "done"

        llm_mod._call_model = fake_call
        llm_mod.reset_limits()
        bundle = inquiry._run_pipeline(df, schema, "is amount correlated with score?")
        blob = captured["all"]
        for token in ("111.111", "555.555", "0.987654", "4.54321"):
            assert token not in blob, token
        assert bundle["result"]["intent"] == "correlation"
    finally:
        llm_mod._secret, llm_mod._call_model = orig_secret, orig_call
        llm_mod.reset_limits()


def test_input_fingerprint_changes_with_content():
    if not _READY:
        return
    base = {"visitorid": [1, 1, 2], "amount": [1.0, 2.0, 3.0]}
    df1 = pd.DataFrame(base)
    df2 = pd.DataFrame({"visitorid": [1, 1, 2], "amount": [1.0, 2.0, 999.0]})
    schema = analytics_schema.infer_schema(df1)
    fp1 = inquiry._input_fingerprint(df1, schema, "how many visitors")
    fp2 = inquiry._input_fingerprint(df2, schema, "how many visitors")
    assert fp1 != fp2
    # stable for identical content
    assert fp1 == inquiry._input_fingerprint(df1.copy(), schema, "how many visitors")


def _run_all():
    if not _READY:
        print("SKIP  inquiry suggestions (streamlit/plotly not importable)")
        return 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    for test in tests:
        try:
            test()
            print("PASS  " + test.__name__)
            passed += 1
        except AssertionError as exc:
            print("FAIL  " + test.__name__ + " -> " + str(exc))
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print("ERROR " + test.__name__ + " -> " + repr(exc))
            failed += 1
    print("\n{} passed, {} failed, {} total".format(passed, failed, passed + failed))
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
