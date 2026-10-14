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
    assert float(table.loc[0, 0]) == 100.0


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
    assert list(result["table"].columns[2:]) == [0]


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
