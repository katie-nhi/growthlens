"""Unit tests for analytics_schema (synthetic data only).

Run directly:  python tests/test_analytics_schema.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

import analytics_schema as schema_mod  # noqa: E402


def _event_log():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u4"],
            "event": ["view", "cart", "purchase", "view", "purchase", "view", "cart"],
            "device": ["ios", "ios", "ios", "android", "android", "ios", "android"],
            "channel": ["paid", "paid", "paid", "organic", "organic", "paid", "organic"],
            "amount": [0.0, 0.0, 25.0, 0.0, 40.0, 0.0, 0.0],
            "order_ts": pd.to_datetime(
                [
                    "2024-01-01", "2024-01-02", "2024-01-03",
                    "2024-01-01", "2024-01-05",
                    "2024-01-08", "2024-01-09",
                ]
            ),
        }
    )


def test_identifier_role():
    schema = schema_mod.infer_schema(_event_log())
    col = schema_mod.column_by_name(schema, "user_id")
    assert col["role"] == "identifier"
    assert col["is_identifierish"] is True


def test_timestamp_role():
    schema = schema_mod.infer_schema(_event_log())
    col = schema_mod.column_by_name(schema, "order_ts")
    assert col["role"] == "timestamp"
    assert col["dtype"] == "datetime"


def test_event_role():
    schema = schema_mod.infer_schema(_event_log())
    assert schema_mod.column_by_name(schema, "event")["role"] == "event"
    assert "event" in schema["events"]


def test_dimension_and_currency_roles():
    schema = schema_mod.infer_schema(_event_log())
    assert schema_mod.column_by_name(schema, "device")["role"] == "dimension"
    assert schema_mod.column_by_name(schema, "amount")["role"] == "currency"
    assert "device" in schema["dimensions"]
    assert "channel" in schema["dimensions"]
    assert "amount" in schema["measures"]


def test_grain_guess_event_log():
    schema = schema_mod.infer_schema(_event_log())
    assert "event" in schema["grain_guess"]


def test_epoch_milliseconds_detected_as_timestamp():
    df = pd.DataFrame(
        {
            "visitor": ["a", "b"],
            "created_at": [1704067200000, 1704153600000],  # 2024-01-01, 2024-01-02 in ms
        }
    )
    schema = schema_mod.infer_schema(df)
    col = schema_mod.column_by_name(schema, "created_at")
    assert col["role"] == "timestamp"
    assert col["epoch_unit"] == "ms"
    assert "created_at" in schema["timestamps"]


def test_epoch_seconds_detected_as_timestamp():
    df = pd.DataFrame({"visitor": ["a", "b"], "event_ts": [1704067200, 1704153600]})
    col = schema_mod.column_by_name(schema_mod.infer_schema(df), "event_ts")
    assert col["epoch_unit"] == "s"


def test_public_payload_has_no_raw_rows_or_identifier_values():
    df = _event_log()
    df = df.assign(email=["a@x.com", "b@x.com", "c@x.com", "d@x.com", "e@x.com", "f@x.com", "g@x.com"])
    schema = schema_mod.infer_schema(df)
    payload = schema_mod.public_payload(df, schema)
    assert "row_count" in payload
    assert len(payload["columns"]) == len(df.columns)
    # no column entry should carry an emails/values list for the identifier
    email_entry = [c for c in payload["columns"] if c["name"] == "email"][0]
    assert "top_values" not in email_entry
    # nothing that looks like raw rows leaked
    assert all("top_values" not in c for c in payload["columns"] if c["name"] in ("user_id", "email"))


def test_public_payload_shares_low_cardinality_dimensions():
    payload = schema_mod.public_payload(_event_log(), schema_mod.infer_schema(_event_log()))
    device_entry = [c for c in payload["columns"] if c["name"] == "device"][0]
    assert "top_values" in device_entry
    assert set(device_entry["top_values"]) <= {"ios", "android"}


def test_utc_week_start_is_monday():
    series = pd.Series(pd.to_datetime(["2025-05-05 09:00", "2025-05-11 23:59", "2025-05-12 00:00"]))
    weeks = schema_mod.utc_week_start(series)
    assert list(weeks.dt.strftime("%Y-%m-%d")) == ["2025-05-05", "2025-05-05", "2025-05-12"]


def test_utc_week_start_converts_timezone_to_utc():
    # 2025-05-04 20:00 -07:00 is 2025-05-05 03:00 UTC -> belongs to the Monday
    # 2025-05-05 UTC week (a naive/local reading would wrongly use 2025-04-28).
    aware = pd.Series(pd.to_datetime(["2025-05-04 20:00:00-07:00"]))
    weeks = schema_mod.utc_week_start(aware)
    assert weeks.dt.strftime("%Y-%m-%d").iloc[0] == "2025-05-05"


def test_utc_week_start_epoch_ms_is_utc():
    # 2025-05-05 09:00 UTC in epoch milliseconds.
    series = pd.Series([1746435600000])
    weeks = schema_mod.utc_week_start(series, "ms")
    assert weeks.dt.strftime("%Y-%m-%d").iloc[0] == "2025-05-05"


def test_concatenated_ids_are_identifiers():
    df = pd.DataFrame(
        {
            "visitorid": [1, 1, 2],
            "itemid": [10, 11, 10],
            "transactionid": [None, None, 500],
            "event": ["view", "cart", "transaction"],
        }
    )
    schema = schema_mod.infer_schema(df)
    for name in ("visitorid", "itemid", "transactionid"):
        col = schema_mod.column_by_name(schema, name)
        assert col["role"] == "identifier", name + " should be an identifier"
        assert col["is_identifierish"] is True
        assert name in schema["identifiers"]
    assert "itemid" not in schema["measures"]


def test_european_currency_text_is_numeric():
    df = pd.DataFrame({"visitorid": [1, 2, 3], "event": ["view", "view", "transaction"],
                       "order_value_sgd": ["", "", "105,5"]})
    col = schema_mod.column_by_name(schema_mod.infer_schema(df), "order_value_sgd")
    assert col["role"] == "currency"


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
