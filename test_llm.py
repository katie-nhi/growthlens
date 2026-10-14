"""Unit tests for the optional LLM client (no real network calls).

Run directly:  python tests/test_llm.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402


def _with_env(fn):
    """Run fn with a clean llm._secret monkeypatch, then restore."""
    original = llm._secret
    try:
        return fn()
    finally:
        llm._secret = original


def test_not_configured_without_key():
    def run():
        llm._secret = lambda name: None
        assert llm.is_llm_configured() is False
        try:
            llm.interpret_question("q", {})
            assert False, "should have raised"
        except llm.LLMError as exc:
            assert "AI connection required" in str(exc)
    _with_env(run)


def test_interpret_question_parses_plan():
    def run():
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        plan = {"intent": "distinct_count", "user_id": "user_id", "metric": {"operation": "nunique", "column": "user_id"}}
        original_call = llm._call_model
        try:
            llm._call_model = lambda messages, temperature=0.0: json.dumps(plan)
            assert llm.is_llm_configured() is True
            parsed = llm.interpret_question("how many users per channel", {"columns": []})
            assert parsed["intent"] == "distinct_count"
        finally:
            llm._call_model = original_call
    _with_env(run)


def test_extract_json_handles_code_fences():
    text = "```json\n{\"intent\": \"unsupported\", \"reason\": \"no data\"}\n```"
    parsed = llm._extract_json(text)
    assert parsed["intent"] == "unsupported"


def test_extract_json_raises_on_garbage():
    try:
        llm._extract_json("I cannot help with that.")
        assert False, "should have raised"
    except llm.LLMError:
        pass


def test_explain_findings_uses_model():
    def run():
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        original_call = llm._call_model
        try:
            llm._call_model = lambda messages, temperature=0.0: "Channel A converts best."
            text = llm.explain_findings("q", {"intent": "groupby_aggregate"}, {"intent": "groupby_aggregate"})
            assert "converts" in text
        finally:
            llm._call_model = original_call
    _with_env(run)


def test_generate_recommendations_still_not_implemented():
    try:
        llm.generate_recommendations_with_llm()
        assert False, "should have raised"
    except NotImplementedError:
        pass


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
