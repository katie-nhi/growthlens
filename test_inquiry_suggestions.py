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
