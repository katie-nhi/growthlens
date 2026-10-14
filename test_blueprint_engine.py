"""Unit tests for blueprint_engine (standard library only, plus this module).

Run directly:  python tests/test_blueprint_engine.py
Or with pytest: pytest tests/test_blueprint_engine.py

These tests use only synthetic example metadata - never real customer data.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blueprint_engine as engine  # noqa: E402


def _example_schema():
    return engine.normalize_schema(engine.EXAMPLE_SCHEMA)


# ---------------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------------

def test_normalize_schema_skips_blank_and_lowercases():
    raw = [
        {"name": " UserID ", "data_type": "IDENTIFIER", "role": "Identifier"},
        {"name": "", "data_type": "text", "role": "ignore"},
        {"name": "Amount", "data_type": "Currency", "role": "Measure", "meaning": "x"},
    ]
    schema = engine.normalize_schema(raw)
    assert len(schema) == 2, "blank column should be skipped"
    assert schema[0]["name"] == "UserID"
    assert schema[0]["data_type"] == "identifier"
    assert schema[0]["role"] == "identifier"
    assert schema[1]["data_type"] == "currency"


def test_capabilities_example_schema():
    caps = engine.analyze_capabilities(_example_schema())
    assert caps["identifier"] is True
    assert caps["timestamp"] is True
    assert caps["event"] is True
    assert caps["currency"] is False


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_funnel_recommended_for_example_schema():
    chosen, blocked = engine.recommend_analyses(
        _example_schema(), "find where users drop off in the signup conversion funnel"
    )
    assert "funnel" in chosen, "funnel should be recommended"
    assert blocked == [], "nothing should be blocked for the example schema"


def test_revenue_blocked_without_currency():
    chosen, blocked = engine.recommend_analyses(
        _example_schema(), "analyse revenue and ROAS by channel"
    )
    blocked_keys = {b["key"] for b in blocked}
    assert "revenue" in blocked_keys, "revenue should be blocked without a currency column"
    assert "revenue" not in chosen
    revenue_block = next(b for b in blocked if b["key"] == "revenue")
    assert "currency" in revenue_block["missing"]


def test_cohort_blocked_without_identifier():
    schema = engine.normalize_schema([
        {"name": "country", "data_type": "categorical", "role": "dimension"},
        {"name": "when", "data_type": "datetime", "role": "timestamp"},
    ])
    _chosen, blocked = engine.recommend_analyses(schema, "retention and churn analysis")
    blocked_keys = {b["key"] for b in blocked}
    assert "cohort" in blocked_keys
    cohort_block = next(b for b in blocked if b["key"] == "cohort")
    assert "identifier" in cohort_block["missing"]


def test_objective_controls_selection_not_everything():
    chosen, _ = engine.recommend_analyses(
        _example_schema(), "compare average session duration between A/B test variants"
    )
    # A/B test needs group+measure which the example schema lacks, so the
    # engine should NOT dump every supported technique.
    assert len(chosen) <= 4
    assert "revenue" not in chosen


def test_funnel_formulas_use_user_column_names():
    result = engine.generate_blueprint(
        columns=[
            {"name": "person_key", "data_type": "identifier", "role": "identifier"},
            {"name": "occurred_at", "data_type": "datetime", "role": "timestamp"},
            {"name": "action", "data_type": "categorical", "role": "event"},
        ],
        objective="signup funnel drop-off",
        granularity="one row = one action by one person",
    )
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    joined = " ".join(f["formula"] for f in funnel["formulas"])
    assert "person_key" in joined, "formulas must use the user's column names"
    assert "action" in joined
    assert "visitorid" not in joined, "must not leak example column names"


def test_funnel_and_cohort_include_power_query():
    result = engine.generate_blueprint(
        _example_schema(), "conversion funnel and retention", "one row = one event"
    )
    for key in ("funnel", "cohort"):
        analysis = next((a for a in result["analyses"] if a["key"] == key), None)
        assert analysis is not None, key + " should be recommended"
        assert analysis["power_query"], key + " needs Power Query guidance"


def test_segmentation_uses_sumifs_and_dimension():
    schema = engine.normalize_schema([
        {"name": "channel", "data_type": "categorical", "role": "dimension"},
        {"name": "spend", "data_type": "numeric", "role": "measure", "unit": "USD"},
    ])
    result = engine.generate_blueprint(schema, "breakdown by channel", "one row = one order")
    seg = next(a for a in result["analyses"] if a["key"] == "segmentation")
    formulas = " ".join(f["formula"] for f in seg["formulas"])
    assert "SUMIFS" in formulas
    assert "Data[channel]" in formulas
    assert "Data[spend]" in formulas


def test_ab_test_formulas_reference_group_and_measure():
    schema = engine.normalize_schema([
        {"name": "variant", "data_type": "text", "role": "group"},
        {"name": "conversion_time", "data_type": "numeric", "role": "measure"},
    ])
    result = engine.generate_blueprint(schema, "A/B experiment comparison", "one row = one user")
    ab = next(a for a in result["analyses"] if a["key"] == "ab_test")
    formulas = " ".join(f["formula"] for f in ab["formulas"])
    assert "AVERAGEIFS" in formulas
    assert "Data[variant]" in formulas
    assert "Data[conversion_time]" in formulas


# ---------------------------------------------------------------------------
# Anti-fabrication + dashboard
# ---------------------------------------------------------------------------

def test_dashboard_uses_placeholders_only():
    result = engine.generate_blueprint(
        _example_schema(), "conversion funnel", "one row = one event"
    )
    for kpi in result["dashboard"]["kpis"]:
        assert kpi["placeholder"] == "?", "KPIs must not contain invented numbers"
    for chart in result["dashboard"]["charts"]:
        assert chart["note"], "each blank chart must explain it is unpopulated"


def test_result_is_labelled_rule_based():
    result = engine.generate_blueprint(_example_schema(), "funnel", "one row = one event")
    assert result["source"] == "rule-based"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_inputs_reports_missing_required_fields():
    errors = engine.validate_inputs("", "", _example_schema())
    assert any("objective" in e.lower() for e in errors)
    assert any("granularity" in e.lower() for e in errors)


def test_validate_inputs_detects_duplicates():
    schema = engine.normalize_schema([
        {"name": "id", "data_type": "identifier", "role": "identifier"},
        {"name": "id", "data_type": "text", "role": "ignore"},
    ])
    errors = engine.validate_inputs("obj", "one row = one thing", schema)
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_inputs_empty_schema():
    errors = engine.validate_inputs("obj", "grain", [])
    assert any("column" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Timestamp UNIT awareness
# ---------------------------------------------------------------------------

def _event_schema(unit="datetime"):
    return engine.normalize_schema([
        {"name": "ts", "data_type": "datetime", "role": "timestamp", "unit": unit},
        {"name": "visitorid", "data_type": "identifier", "role": "identifier"},
        {"name": "event", "data_type": "categorical", "role": "event"},
    ])


def test_unix_ms_converted_before_text_and_grouping():
    result = engine.generate_blueprint(
        _event_schema("Unix ms"), "conversion funnel and retention", "one row = one event"
    )
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    joined = " ".join(f["formula"] for f in cohort["formulas"])
    assert "/86400000" in joined, "Unix ms must be divided by 86,400,000"
    assert "DATE(1970,1,1)" in joined, "must offset from the Unix epoch"
    assert "ts_UTC" in joined, "downstream formulas must use the converted column"
    # No TEXT() should run on the RAW unix column.
    for f in cohort["formulas"]:
        if "TEXT(" in f["formula"] and "FirstObserved" not in f["formula"]:
            assert "[@ts]" not in f["formula"], "must not TEXT() the raw Unix column"


def test_unix_seconds_converted_before_grouping():
    result = engine.generate_blueprint(
        _event_schema("Unix seconds"), "retention", "one row = one event"
    )
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    joined = " ".join(f["formula"] for f in cohort["formulas"])
    assert "/86400" in joined and "/86400000" not in joined


def test_datetime_unit_needs_no_conversion():
    result = engine.generate_blueprint(
        _event_schema("datetime"), "retention", "one row = one event"
    )
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    joined = " ".join(f["formula"] for f in cohort["formulas"])
    assert "/86400000" not in joined
    assert "ts_UTC" not in joined


# ---------------------------------------------------------------------------
# Time granularity + cohort correctness
# ---------------------------------------------------------------------------

def test_weekly_cohort_uses_relative_weeks():
    result = engine.generate_blueprint(
        _event_schema(), "weekly retention by cohort", "one row = one event",
        time_granularity="weekly",
    )
    assert result["time_granularity"] == "weekly"
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    assert "weekly" in cohort["title"].lower()
    joined = " ".join(f["formula"] for f in cohort["formulas"])
    assert "/7" in joined and "INT(" in joined, "relative week index must divide day-gap by 7"
    assert any("W0" in c for c in cohort["pivot"]["columns"]), "pivot columns must be W0..W8"


def test_auto_granularity_infers_weekly_from_objective():
    assert engine.resolve_time_granularity("show me weekly retention") == "weekly"
    assert engine.resolve_time_granularity("monthly growth trend") == "monthly"
    assert engine.resolve_time_granularity("daily active users") == "daily"
    assert engine.resolve_time_granularity("something vague") == "monthly"


def test_cohort_dedupes_and_defines_denominator():
    result = engine.generate_blueprint(
        _event_schema(), "weekly retention", "one row = one event", time_granularity="weekly"
    )
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    pq = " ".join(cohort["power_query"]).lower()
    assert "deduplicate" in pq or "remove duplicates" in pq
    assert "distinct count" in cohort["pivot"]["aggregation"].lower()
    denominator = " ".join(m["denominator"] for m in cohort["metrics"] if m).lower()
    assert "cohort size" in denominator
    assert any("denominator" in m["name"].lower() for m in cohort["metrics"] if m)


def test_cohort_explains_incomplete_observation():
    result = engine.generate_blueprint(
        _event_schema(), "weekly retention", "one row = one event", time_granularity="weekly"
    )
    cohort = next(a for a in result["analyses"] if a["key"] == "cohort")
    notes = " ".join(cohort["notes"]).lower()
    assert "censored" in notes or "incomplete" in notes


# ---------------------------------------------------------------------------
# Ordered funnel correctness
# ---------------------------------------------------------------------------

def test_funnel_is_ordered_not_global_minimum():
    result = engine.generate_blueprint(
        _event_schema(), "checkout funnel drop-off", "one row = one event",
        funnel_stages="view, cart, purchase",
    )
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    joined = " ".join(f["formula"] for f in funnel["formulas"])
    assert "step1_time" in joined and "step2_time" in joined
    assert '"&' in joined, "later steps must be '>' the previous step's time"
    pq = " ".join(funnel["power_query"]).lower()
    assert "after" in pq
    notes = " ".join(funnel["notes"]).lower()
    assert "global minimum" in notes


def test_funnel_stage_labels_are_used():
    result = engine.generate_blueprint(
        _event_schema(), "funnel", "one row = one event",
        funnel_stages="view, cart, purchase",
    )
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    assert funnel["stages"] == ["view", "cart", "purchase"]
    joined = " ".join(f["formula"] for f in funnel["formulas"])
    assert '"view"' in joined and '"purchase"' in joined


def test_funnel_participation_labelled_not_conversion():
    result = engine.generate_blueprint(_event_schema(), "funnel", "one row = one event")
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    names = " ".join(m["name"] for m in funnel["metrics"] if m)
    assert "participation" in names.lower()
    assert "NOT conversion" in names


def test_repeated_events_use_distinct_counts():
    result = engine.generate_blueprint(_event_schema(), "funnel", "one row = one event")
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    joined = " ".join(f["formula"] for f in funnel["formulas"])
    assert "UNIQUE" in joined, "repeat events must be de-duplicated per user"


# ---------------------------------------------------------------------------
# Summary + ranking + options
# ---------------------------------------------------------------------------

def test_summary_reports_readiness_and_missing():
    result = engine.generate_blueprint(
        _example_schema(), "analyse revenue and ROAS", "one row = one event"
    )
    summary = result["summary"]
    assert summary["top"], "summary must list top recommendations"
    assert any("identifier" in p for p in summary["readiness_present"])
    assert any("revenue" in m["title"].lower() for m in summary["missing"])


def test_analyses_are_ranked_with_primary_flag():
    result = engine.generate_blueprint(
        _example_schema(), "funnel and retention and trend", "one row = one event"
    )
    ranks = [a["rank"] for a in result["analyses"]]
    assert ranks == list(range(1, len(ranks) + 1))
    assert result["analyses"][0]["primary"] is True


def test_generate_without_stages_does_not_crash():
    result = engine.generate_blueprint(
        _event_schema(), "funnel", "one row = one event", funnel_stages=None
    )
    funnel = next(a for a in result["analyses"] if a["key"] == "funnel")
    assert len(funnel["stages"]) >= 2



# ---------------------------------------------------------------------------
# Simple runner (works without pytest)
# ---------------------------------------------------------------------------

def _run_all():
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
