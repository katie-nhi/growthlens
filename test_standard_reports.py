"""Unit tests for standard_reports using synthetic, deterministic data.

Run directly:  python tests/test_standard_reports.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import analytics_schema  # noqa: E402
import standard_reports as reports  # noqa: E402


def _funnel_df():
    rows = [
        # visitor, event, timestamp, item, transaction id, amount
        ("v1", "view", "2024-01-01 08:00", 10, None, None),
        ("v1", "view", "2024-01-01 09:00", 11, None, None),
        ("v1", "addtocart", "2024-01-01 10:00", 10, None, None),
        ("v1", "transaction", "2024-01-01 12:00", 10, "t1", 25.0),
        ("v2", "view", "2024-01-01 08:30", 12, None, None),
        ("v2", "transaction", "2024-01-01 09:00", 12, "t2", 40.0),  # no cart -> not a funnel transaction
        ("v3", "addtocart", "2024-01-01 07:00", 13, None, None),  # cart BEFORE view (out of order)
        ("v3", "view", "2024-01-01 08:00", 13, None, None),
        ("v4", "transaction", "2024-01-01 06:00", 14, "t4", 10.0),  # transaction BEFORE view (out of order)
        ("v4", "view", "2024-01-01 07:00", 14, None, None),
        ("v4", "addtocart", "2024-01-01 08:00", 14, None, None),
        ("v5", "view", "2024-01-01 09:00", 15, None, None),
    ]
    df = pd.DataFrame(rows, columns=["visitorid", "event", "timestamp", "itemid", "transactionid", "order_value_sgd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _cohort_df():
    rows = [
        # visitor, timestamp (Mondays 2024-01-01, 01-08, 01-15)
        ("A", "2024-01-01 09:00"),
        ("A", "2024-01-01 10:00"),  # repeated in same week -> one visitor-week
        ("A", "2024-01-08 09:00"),  # retained W1
        ("B", "2024-01-01 11:00"),
        ("C", "2024-01-08 09:00"),
        ("D", "2024-01-08 10:00"),
        ("D", "2024-01-15 10:00"),  # retained W1 within cohort week1
        ("E", "2024-01-15 12:00"),  # cohort week2 -> future weeks N/A
    ]
    df = pd.DataFrame(rows, columns=["visitorid", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["event"] = "view"
    return df


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_detect_columns_retailrocket_like():
    df = _funnel_df()
    cols = reports.detect_columns(df, analytics_schema.infer_schema(df))
    assert cols["visitor"] == "visitorid"
    assert cols["item"] == "itemid"
    assert cols["event"] == "event"
    assert cols["timestamp"] == "timestamp"
    assert cols["amount"] == "order_value_sgd"


def test_stage_values_mapping():
    df = _funnel_df()
    mapping = reports.stage_values(df, "event")
    assert mapping["view"] == "view"
    assert mapping["cart"] == "addtocart"
    assert mapping["transaction"] == "transaction"


# ---------------------------------------------------------------------------
# Executive overview
# ---------------------------------------------------------------------------

def test_executive_overview_metrics():
    df = _funnel_df()
    result = reports.run_all(df)["overview"]
    metrics = dict(result["metrics"])
    assert metrics["Total events (rows)"] == "12"
    assert metrics["Unique visitors"] == "5"
    table = result["table"].set_index("event")
    assert int(table.loc["view", "rows"]) == 6
    assert int(table.loc["view", "unique_visitors"]) == 5


# ---------------------------------------------------------------------------
# Funnel (ordered, out-of-order events)
# ---------------------------------------------------------------------------

def test_funnel_counts_are_ordered():
    df = _funnel_df()
    funnel = reports.run_all(df)["funnel"]
    table = funnel["table"].set_index("stage")["visitors"]
    assert int(table["View"]) == 5
    assert int(table["Cart"]) == 2  # v1 and v4 only
    assert int(table["Transaction"]) == 1  # v1 only (v2 had no cart, v4 bought before viewing)


def test_funnel_step_conversion():
    df = _funnel_df()
    funnel = reports.run_all(df)["funnel"]
    table = funnel["table"].set_index("stage")
    assert float(table.loc["Cart", "step conversion %"]) == 40.0
    assert float(table.loc["Transaction", "step conversion %"]) == 50.0


# ---------------------------------------------------------------------------
# Weekly cohort retention
# ---------------------------------------------------------------------------

def test_cohort_monday_start_and_dedupe():
    df = _cohort_df()
    cohort = reports.run_all(df)["cohort"]
    table = cohort["table"]
    # cohort starts are Mondays
    for value in table["cohort_start"]:
        assert pd.Timestamp(value).weekday() == 0, value
    # visitor A appeared twice in week0 but is one visitor-week record
    assert int(cohort["metrics"][2][1]) == len(set([("A", "2024-01-01"), ("A", "2024-01-08"), ("B", "2024-01-01"),
                                                   ("C", "2024-01-08"), ("D", "2024-01-08"), ("D", "2024-01-15"),
                                                   ("E", "2024-01-15")]))


def test_cohort_retention_values():
    df = _cohort_df()
    table = reports.run_all(df)["cohort"]["table"].set_index("cohort_start")
    week0 = table.loc["2024-01-01"]
    assert int(week0["cohort_size"]) == 2
    assert float(week0["W0"]) == 100.0
    assert float(week0["W1"]) == 50.0
    week1 = table.loc["2024-01-08"]
    assert int(week1["cohort_size"]) == 2
    assert float(week1["W0"]) == 100.0
    assert float(week1["W1"]) == 50.0


def test_cohort_censors_incomplete_follow_up_with_na():
    df = _cohort_df()
    table = reports.run_all(df)["cohort"]["table"].set_index("cohort_start")
    week2 = table.loc["2024-01-15"]
    assert int(week2["cohort_size"]) == 1
    assert float(week2["W0"]) == 100.0
    assert pd.isna(week2["W1"]), "future week must be N/A, not 0"


def test_monday_event_labels_as_same_monday():
    # A single event on Monday 2025-05-05 (a real Monday) must be labelled
    # 2025-05-05, never the preceding Sunday 2025-05-04.
    df = pd.DataFrame({"visitorid": [1], "event": ["view"], "timestamp": pd.to_datetime(["2025-05-05 09:00"])})
    table = reports.run_all(df)["cohort"]["table"]
    assert list(table["cohort_start"]) == ["2025-05-05"]


def test_monday_event_epoch_ms_labels_as_monday():
    df = pd.DataFrame({"visitorid": [1], "event": ["view"], "timestamp": [1746435600000]})
    table = reports.run_all(df)["cohort"]["table"]
    assert list(table["cohort_start"]) == ["2025-05-05"]


def test_cohort_utc_week_for_tzaware_timestamps():
    # 2025-05-04 20:00 -07:00 == 2025-05-05 03:00 UTC -> Monday 2025-05-05 week.
    ts = pd.to_datetime(["2025-05-04 20:00:00-07:00"])
    df = pd.DataFrame({"visitorid": [1], "event": ["view"], "timestamp": ts})
    table = reports.run_all(df)["cohort"]["table"]
    assert list(table["cohort_start"]) == ["2025-05-05"]


def test_zero_retention_is_not_na():
    # Cohort A (week 2024-01-01) can observe W1 but nobody returns -> a valid
    # 0.0%. Cohort C (week 2024-01-22) cannot observe W1 yet -> N/A.
    rows = [
        ("A1", "2024-01-01"),
        ("A2", "2024-01-01"),
        ("A2", "2024-01-15"),
        ("B1", "2024-01-08"),
        ("C1", "2024-01-22"),
    ]
    df = pd.DataFrame(rows, columns=["visitorid", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["event"] = "view"
    table = reports.run_all(df)["cohort"]["table"].set_index("cohort_start")
    assert float(table.loc["2024-01-01", "W1"]) == 0.0
    assert not pd.isna(table.loc["2024-01-01", "W1"]), "observable 0% must not be N/A"
    assert pd.isna(table.loc["2024-01-22", "W1"]), "future week must be N/A"


def test_unique_visitor_week_records_counted():
    df = _cohort_df()
    cohort = reports.run_all(df)["cohort"]
    # A appeared twice in 2024-01-01 -> a single visitor-week record.
    expected = {("A", "2024-01-01"), ("A", "2024-01-08"), ("B", "2024-01-01"),
                ("C", "2024-01-08"), ("D", "2024-01-08"), ("D", "2024-01-15"),
                ("E", "2024-01-15")}
    assert int(cohort["metrics"][2][1]) == len(expected) == 7


def test_cohort_heatmap_labels_match_table():
    df = _cohort_df()
    table = reports.run_all(df)["cohort"]["table"]
    labels = reports.cohort_heatmap_labels(table)
    assert len(labels) == len(table) == 3
    assert labels[0] == "01 Jan 2024"
    assert labels == ["01 Jan 2024", "08 Jan 2024", "15 Jan 2024"]
    # Explicitly non-ISO strings so Plotly cannot auto-parse them as dates.
    assert all("-" not in label for label in labels)


def test_final_partial_week_zero_is_na():
    # Dataset ends mid-week on 2024-01-15, so the 2024-01-15 week is only partly
    # observed: a 0% there is ambiguous and must be N/A, while an earlier fully
    # observed week keeps a genuine 0%.
    rows = [
        ("A1", "2024-01-01 09:00"),
        ("A2", "2024-01-01 09:00"),
        ("B1", "2024-01-08 09:00"),
        ("C1", "2024-01-15 10:00"),
    ]
    df = pd.DataFrame(rows, columns=["visitorid", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["event"] = "view"
    table = reports.run_all(df)["cohort"]["table"].set_index("cohort_start")
    assert float(table.loc["2024-01-01", "W1"]) == 0.0
    assert not pd.isna(table.loc["2024-01-01", "W1"])
    assert pd.isna(table.loc["2024-01-01", "W2"]), "partial final week must be N/A, not 0"
    assert pd.isna(table.loc["2024-01-08", "W1"]), "partial final week must be N/A, not 0"


def test_heatmap_y_axis_is_categorical():
    try:
        import data_analysis
    except Exception:  # noqa: BLE001 - streamlit/plotly not installed
        return
    df = _cohort_df()
    report = reports.run_all(df)["cohort"]
    fig = data_analysis._fig_from_spec(report["chart"], report["table"])
    assert fig.layout.yaxis.type == "category"
    assert list(fig.data[0].y) == reports.cohort_heatmap_labels(report["table"])


def test_synthetic_demo_funnel_and_segmentation_regression():
    # Protects the confirmed synthetic-demo results (180/25/12 and 26/132/22)
    # whenever the sample CSV is available. Skipped otherwise.
    path = os.environ.get(
        "GROWTHLENS_DEMO_CSV",
        "/Users/trandangtunhi/Downloads/GrowthLens_Synthetic_Events_Demo - Sheet1.csv",
    )
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    result = reports.run_all(df)
    reached = result["funnel"]["_reached"]
    assert (reached["view"], reached["cart"], reached["transaction"]) == (180, 25, 12), reached
    seg = result["segmentation"]["table"].set_index("segment")["visitors"]
    assert (int(seg["1 view"]), int(seg["2-4 views"]), int(seg["5+ views"])) == (26, 132, 22), seg
    assert int(result["cohort"]["metrics"][2][1]) == 265


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_segmentation_buckets_and_outcomes():
    df = _funnel_df()
    seg = reports.run_all(df)["segmentation"]
    table = seg["table"].set_index("segment")
    assert int(table.loc["1 view", "visitors"]) == 4
    assert int(table.loc["2-4 views", "visitors"]) == 1
    assert int(table.loc["5+ views", "visitors"]) == 0
    assert int(table.loc["1 view", "reached_cart"]) == 1  # v4
    assert int(table.loc["2-4 views", "reached_transaction"]) == 1  # v1


# ---------------------------------------------------------------------------
# Epoch millisecond timestamps
# ---------------------------------------------------------------------------

def test_reports_handle_epoch_milliseconds():
    df = _funnel_df()
    df["timestamp"] = (df["timestamp"].astype("int64") // 10**6)
    funnel = reports.run_all(df)["funnel"]
    assert funnel["available"] is True
    table = funnel["table"].set_index("stage")["visitors"]
    assert int(table["Transaction"]) == 1


def test_funnel_unavailable_without_event_column():
    df = _funnel_df().rename(columns={"event": "label"})
    df["label"] = ["a"] * len(df)
    result = reports.run_all(df)["funnel"]
    assert result["available"] is False
    assert result["reason"]


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
