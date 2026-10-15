"""Unit tests for analytics_engine using synthetic data only.

Run directly:  python tests/test_analytics_engine.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import analytics_engine as engine  # noqa: E402
import analytics_schema  # noqa: E402


def _event_log():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u4"],
            "event": ["view", "cart", "purchase", "view", "purchase", "view", "cart"],
            "device": ["ios", "ios", "ios", "android", "android", "ios", "android"],
            "channel": ["paid", "paid", "paid", "organic", "organic", "paid", "organic"],
            "amount": [0.0, 0.0, 25.0, 0.0, 40.0, 0.0, 0.0],
            "order_ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-05", "2024-01-08", "2024-01-09"]
            ),
        }
    )


def _schema(df):
    return analytics_schema.infer_schema(df)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_ok_for_groupby():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "amount"}, "group_by": ["device"]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert ok, (errors, missing)


def test_validate_flags_missing_column():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "revenue"}, "group_by": ["device"]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert not ok
    assert "revenue" in missing


def test_revenue_without_amount_column_is_unanswerable():
    df = pd.DataFrame({"user_id": ["u1", "u2"], "event": ["a", "b"], "spend": ["x", "y"]})
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "revenue"}, "group_by": ["event"]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert not ok and "revenue" in missing


def test_numeric_aggregation_on_text_is_error():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "event"}, "group_by": ["device"]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert not ok and any("numeric" in e for e in errors)


def test_disallowed_operation_rejected():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "drop_table", "column": "amount"}, "group_by": ["device"]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("Unsupported aggregation" in e for e in errors)


# ---------------------------------------------------------------------------
# Execution - grouping / distinct
# ---------------------------------------------------------------------------

def test_groupby_sum_amount_by_device():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "amount"}, "group_by": ["device"]}
    result = engine.execute_plan(df, plan, _schema(df))
    table = result["table"].set_index("device")["sum(amount)"]
    assert table["ios"] == 25.0
    assert table["android"] == 40.0


def test_distinct_users_by_channel():
    df = _event_log()
    plan = {"intent": "distinct_count", "metric": {"operation": "nunique", "column": "user_id"}, "user_id": "user_id", "group_by": ["channel"]}
    result = engine.execute_plan(df, plan, _schema(df))
    table = result["table"].set_index("channel")["distinct user_id"]
    assert table["paid"] == 2  # u1, u3
    assert table["organic"] == 2  # u2, u4


def test_event_stages_counts_unique_visitors():
    df = _event_log()
    plan = {"intent": "event_stages", "event_column": "event", "user_id": "user_id", "metric": {"operation": "count"}}
    result = engine.execute_plan(df, plan, _schema(df))
    table = result["table"].set_index("event")["unique users"]
    assert table["view"] == 3  # u1, u2, u3
    assert table["cart"] == 2  # u1, u4
    assert table["purchase"] == 2  # u1, u2


# ---------------------------------------------------------------------------
# Execution - funnel ordering (unusual ordering)
# ---------------------------------------------------------------------------

def _funnel_df():
    return pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b", "b", "b", "c", "c"],
            "event": ["purchase", "view", "cart", "view", "cart", "purchase", "cart", "view"],
            "ts": pd.to_datetime(
                [
                    "2024-01-01 08:00", "2024-01-01 09:00", "2024-01-01 10:00",
                    "2024-01-01 08:00", "2024-01-01 09:00", "2024-01-01 10:00",
                    "2024-01-01 08:00", "2024-01-01 09:00",
                ]
            ),
        }
    )


def test_funnel_respects_ordered_stages():
    df = _funnel_df()
    plan = {
        "intent": "funnel",
        "user_id": "user_id",
        "event_column": "event",
        "timestamp": "ts",
        "stages": [
            {"label": "view", "event_value": "view"},
            {"label": "cart", "event_value": "cart"},
            {"label": "purchase", "event_value": "purchase"},
        ],
        "metric": {"operation": "count"},
    }
    result = engine.execute_plan(df, plan, _schema(df))
    users = list(result["table"]["users"])
    # user a bought BEFORE viewing -> only view+cart count; user c carts before viewing -> view only
    assert users == [3, 2, 1]


def test_funnel_requires_two_stages():
    df = _funnel_df()
    plan = {"intent": "funnel", "user_id": "user_id", "event_column": "event", "timestamp": "ts",
            "stages": [{"label": "view", "event_value": "view"}]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("two ordered stages" in e for e in errors)


# ---------------------------------------------------------------------------
# Execution - time trend
# ---------------------------------------------------------------------------

def test_time_trend_weekly_rows():
    df = _event_log()
    plan = {"intent": "time_trend", "time": {"column": "order_ts", "period": "W"}, "metric": {"operation": "count"}}
    result = engine.execute_plan(df, plan, _schema(df))
    assert "period" in result["table"].columns
    assert result["table"]["rows"].sum() == 7


def test_time_trend_handles_epoch_milliseconds():
    df = pd.DataFrame(
        {
            "visitor": ["a", "b", "c"],
            "created_at": [1704067200000, 1704672000000, 1705276800000],  # weekly
            "event": ["x", "x", "x"],
        }
    )
    plan = {"intent": "time_trend", "time": {"column": "created_at", "period": "W"}, "metric": {"operation": "count"}}
    result = engine.execute_plan(df, plan, _schema(df))
    assert len(result["table"]) == 3


# ---------------------------------------------------------------------------
# Execution - cohort (dedupe + censoring)
# ---------------------------------------------------------------------------

def test_cohort_dedupes_user_within_period():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "ts": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-08", "2024-01-01", "2024-01-09"]
            ),
        }
    )
    plan = {"intent": "cohort", "user_id": "user_id", "timestamp": "ts", "time": {"period": "W"}}
    result = engine.execute_plan(df, plan, _schema(df))
    table = result["table"]
    # cohort_size is the number of unique users first seen in week 0
    assert int(table.iloc[0]["cohort_size"]) == 2
    # week-0 retention is 100%
    assert float(table.loc[0, "W0"]) == 100.0


def test_cohort_censors_future_periods():
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    plan = {"intent": "cohort", "user_id": "user_id", "timestamp": "ts", "time": {"period": "W"}}
    result = engine.execute_plan(df, plan, _schema(df))
    # only week 0 is observable, no invented future columns
    assert list(result["table"].columns[2:]) == ["W0"]


# ---------------------------------------------------------------------------
# Execution - correlation note
# ---------------------------------------------------------------------------

def test_correlation_is_labelled_association():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 13]})
    plan = {"intent": "correlation", "x": "x", "y": "y"}
    result = engine.execute_plan(df, plan, _schema(df))
    assert result["scalar"] is not None
    assert any("causation" in n for n in result["notes"])


# ---------------------------------------------------------------------------
# Execution - unanswerable
# ---------------------------------------------------------------------------

def test_missing_column_returns_explanatory_note():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "revenue"}, "group_by": ["device"]}
    result = engine.execute_plan(df, plan, _schema(df))
    assert result["table"].empty
    assert any("revenue" in n for n in result["notes"])


def test_unsupported_intent_explained():
    df = _event_log()
    plan = {"intent": "unsupported", "reason": "No satisfaction column exists."}
    result = engine.execute_plan(df, plan, _schema(df))
    assert result["table"].empty
    assert any("satisfaction" in n for n in result["notes"])


def test_sum_on_identifier_column_is_refused():
    df = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 11, 10], "event": ["view", "cart", "view"]})
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "itemid"}, "group_by": ["event"]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert not ok and any("identifier" in e for e in errors), errors


def test_mean_on_identifier_column_is_refused():
    df = pd.DataFrame({"visitorid": [1, 1, 2], "event": ["view", "cart", "view"]})
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "mean", "column": "visitorid"}, "group_by": ["event"]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("identifier" in e for e in errors), errors


def test_sum_currency_text_coerces():
    df = pd.DataFrame({"channel": ["a", "a", "b"], "order_value_sgd": ["10,5", "4,5", "20"]})
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "sum", "column": "order_value_sgd"}, "group_by": ["channel"]}
    result = engine.execute_plan(df, plan, _schema(df))
    table = result["table"].set_index("channel")["sum(order_value_sgd)"]
    assert table["a"] == 15.0
    assert table["b"] == 20.0


# ---------------------------------------------------------------------------
# Execution - funnel must be strictly sequential (regression)
# ---------------------------------------------------------------------------

def _strict_funnel_df():
    # A: view then purchase (NO cart) -> must NOT count at purchase.
    # B: cart then purchase (NO view)  -> must NOT count at all.
    # C: view, cart, purchase          -> counts at every stage.
    return pd.DataFrame(
        {
            "user_id": ["A", "A", "B", "B", "C", "C", "C"],
            "event": ["view", "purchase", "cart", "purchase", "view", "cart", "purchase"],
            "ts": pd.to_datetime(
                [
                    "2024-01-01 08:00", "2024-01-01 12:00",
                    "2024-01-01 08:00", "2024-01-01 12:00",
                    "2024-01-01 08:00", "2024-01-01 10:00", "2024-01-01 12:00",
                ]
            ),
        }
    )


def _funnel_plan():
    return {
        "intent": "funnel",
        "user_id": "user_id",
        "event_column": "event",
        "timestamp": "ts",
        "stages": [
            {"label": "view", "event_value": "view"},
            {"label": "cart", "event_value": "cart"},
            {"label": "purchase", "event_value": "purchase"},
        ],
        "metric": {"operation": "count"},
    }


def test_funnel_never_skips_a_stage():
    df = _strict_funnel_df()
    result = engine.execute_plan(df, _funnel_plan(), _schema(df))
    assert list(result["table"]["users"]) == [2, 1, 1]


def test_funnel_drops_visitor_who_misses_a_stage():
    df = _strict_funnel_df()
    result = engine.execute_plan(df, _funnel_plan(), _schema(df))
    table = result["table"].set_index("stage")
    # B never viewed -> excluded everywhere; A never carted -> excluded at purchase.
    assert table.loc["view", "users"] == 2  # A, C
    assert table.loc["cart", "users"] == 1  # C only
    assert table.loc["purchase", "users"] == 1  # C only


def test_funnel_step_conversion_is_monotonic():
    df = _strict_funnel_df()
    result = engine.execute_plan(df, _funnel_plan(), _schema(df))
    counts = list(result["table"]["users"])
    assert all(b <= a for a, b in zip(counts, counts[1:]))


def test_funnel_matches_standard_report():
    import standard_reports

    df = _strict_funnel_df()
    schema = _schema(df)
    result = engine.execute_plan(df, _funnel_plan(), schema)
    engine_counts = list(result["table"]["users"])

    standard = standard_reports.sequential_funnel(df, standard_reports.detect_columns(df, schema), schema)
    standard_counts = list(standard["table"]["visitors"])
    assert engine_counts == standard_counts == [2, 1, 1]


# ---------------------------------------------------------------------------
# Validation - filters fail closed (regression)
# ---------------------------------------------------------------------------

def test_unsupported_filter_operator_is_rejected():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "channel", "op": "equals", "value": "paid"}]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("Unsupported filter operator" in e for e in errors), errors


def test_non_numeric_comparison_value_is_rejected():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "amount", "op": ">", "value": "not-a-number"}]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("numeric value" in e for e in errors), errors


def test_comparison_on_text_column_is_rejected():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "channel", "op": ">", "value": 3}]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("not numeric" in e for e in errors), errors


def test_in_filter_requires_a_list():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "channel", "op": "in", "value": "paid"}]}
    ok, errors, _ = engine.validate_plan(plan, _schema(df))
    assert not ok and any("list of values" in e for e in errors), errors


def test_invalid_filter_never_executes_partially():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "channel", "op": "equals", "value": "paid"}]}
    result = engine.execute_plan(df, plan, _schema(df))
    assert result["table"].empty
    assert any("cannot be answered" in n for n in result["notes"]), result["notes"]


def test_valid_numeric_filter_still_works():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "amount", "op": ">", "value": 10}]}
    result = engine.execute_plan(df, plan, _schema(df))
    total = int(result["table"]["rows"].sum()) if "rows" in result["table"].columns else int(result["table"].iloc[:, 1].sum())
    assert total == 2  # the two purchase rows (25.0 and 40.0)


# ---------------------------------------------------------------------------
# Privacy regressions - filter literals must never appear in error output
# ---------------------------------------------------------------------------

SENTINEL = "alice@example.com"


def test_filter_validation_error_never_echoes_literal_value():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "amount", "op": ">", "value": SENTINEL}]}
    ok, errors, missing = engine.validate_plan(plan, _schema(df))
    assert not ok
    assert SENTINEL not in " ".join(errors)
    assert SENTINEL not in " ".join(missing)


def test_filter_apply_error_never_echoes_literal_value():
    df = _event_log()
    try:
        engine._apply_filters(df, [{"column": "amount", "op": ">", "value": SENTINEL}])
        raise AssertionError("expected _apply_filters to raise")
    except ValueError as exc:
        assert SENTINEL not in str(exc)


def test_execute_plan_notes_never_echo_filter_literal():
    df = _event_log()
    plan = {"intent": "groupby_aggregate", "metric": {"operation": "count"},
            "group_by": ["device"], "filters": [{"column": "amount", "op": ">", "value": SENTINEL}]}
    result = engine.execute_plan(df, plan, _schema(df))
    assert result["table"].empty
    assert SENTINEL not in " ".join(result["notes"])


# ---------------------------------------------------------------------------
# Privacy regressions - correlation keeps pairs local, meta is aggregate-only
# ---------------------------------------------------------------------------

def test_correlation_keeps_raw_pairs_in_table():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 11.0]})
    plan = {"intent": "correlation", "x": "a", "y": "b"}
    result = engine.execute_plan(df, plan, _schema(df))
    assert list(result["table"].columns) == ["a", "b"]
    assert len(result["table"]) == 5


def test_correlation_meta_is_aggregate_only():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 11.0]})
    plan = {"intent": "correlation", "x": "a", "y": "b"}
    result = engine.execute_plan(df, plan, _schema(df))
    summary = result["meta"]["summary"]
    assert summary["method"] == "pearson"
    assert summary["valid_pairs"] == 5
    assert "count" in summary["a"] and "mean" in summary["b"]
    # aggregate summary never contains per-row observation pairs
    assert not any(isinstance(v, list) for v in summary.values())
    for col in ("a", "b"):
        assert set(summary[col]) == {"count", "mean", "std"}



# ---------------------------------------------------------------------------
# Execution - engine cohort agrees with the standard report
# ---------------------------------------------------------------------------

def test_engine_and_standard_cohort_agree():
    import standard_reports

    df = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-08", "2024-01-22", "2024-01-24"]),
        }
    )
    schema = _schema(df)
    plan = {"intent": "cohort", "user_id": "user_id", "timestamp": "ts", "time": {"period": "W"}}
    engine_table = engine.execute_plan(df, plan, schema)["table"]
    standard_table = standard_reports.weekly_cohort(df, standard_reports.detect_columns(df, schema), schema)["table"]
    assert list(engine_table.columns) == list(standard_table.columns)
    assert engine_table.equals(standard_table)


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
