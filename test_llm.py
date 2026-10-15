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


def test_owner_can_disable_ai_even_with_key():
    def run():
        secrets = {
            "GROWTHLENS_LLM_API_KEY": "test-key",
            "GROWTHLENS_LLM_DISABLED": "true",
        }
        llm._secret = lambda name: secrets.get(name)
        assert llm.is_llm_configured() is False
        try:
            llm.interpret_question("q", {})
            assert False, "should have raised when disabled"
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


# ---------------------------------------------------------------------------
# P0-5: abuse limits + provider error handling
# ---------------------------------------------------------------------------

def test_per_session_limit_enforced():
    secrets = {
        "GROWTHLENS_LLM_API_KEY": "test-key",
        "GROWTHLENS_LLM_MAX_PER_SESSION": "1",
        "GROWTHLENS_LLM_COOLDOWN_SECONDS": "0",
    }
    session: dict = {}
    original_secret, original_call, original_state = llm._secret, llm._call_model, llm._session_state
    try:
        llm._secret = lambda name: secrets.get(name)
        llm._call_model = lambda messages, temperature=0.0: json.dumps({"intent": "unsupported"})
        llm._session_state = lambda: session
        llm.reset_global_counter()
        llm.interpret_question("q1", {})  # first is allowed
        try:
            llm.interpret_question("q2", {})
            assert False, "second run should have been rate-limited"
        except llm.LLMRateLimitError as exc:
            assert "limit" in str(exc).lower()
    finally:
        llm._secret, llm._call_model, llm._session_state = original_secret, original_call, original_state
        llm.reset_global_counter()


def test_cooldown_enforced():
    secrets = {
        "GROWTHLENS_LLM_API_KEY": "test-key",
        "GROWTHLENS_LLM_MAX_PER_SESSION": "50",
        "GROWTHLENS_LLM_COOLDOWN_SECONDS": "30",
    }
    session = {"_llm_runs": 1, "_llm_last_run": 1_000_000.0}
    original_secret, original_state = llm._secret, llm._session_state
    try:
        llm._secret = lambda name: secrets.get(name)
        llm._session_state = lambda: session
        llm.reset_global_counter()
        try:
            llm.check_inquiry_allowed(now=1_000_005.0)
            assert False, "cooldown should have blocked this run"
        except llm.LLMRateLimitError as exc:
            assert "wait" in str(exc).lower()
    finally:
        llm._secret, llm._session_state = original_secret, original_state


def test_global_cap_enforced():
    secrets = {"GROWTHLENS_LLM_MAX_GLOBAL": "2"}
    original_secret, original_state = llm._secret, llm._session_state
    try:
        llm._secret = lambda name: secrets.get(name)
        llm._session_state = lambda: {}
        llm._GLOBAL_STATE["runs"] = 2
        try:
            llm.check_inquiry_allowed(now=1_000_000.0)
            assert False, "global cap should have blocked this run"
        except llm.LLMRateLimitError as exc:
            assert "budget" in str(exc).lower()
    finally:
        llm._secret, llm._session_state = original_secret, original_state
        llm.reset_global_counter()


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_call_model_429_raises_rate_limit():
    import requests

    original_secret, original_post = llm._secret, requests.post
    try:
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        requests.post = lambda *a, **k: _FakeResponse(429)
        try:
            llm._call_model([{"role": "user", "content": "hi"}])
            assert False, "429 should raise LLMRateLimitError"
        except llm.LLMRateLimitError as exc:
            assert "429" in str(exc)
    finally:
        llm._secret, requests.post = original_secret, original_post


def test_call_model_timeout_is_friendly():
    import requests

    original_secret, original_post = llm._secret, requests.post
    try:
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None

        def _timeout(*a, **k):
            raise requests.exceptions.Timeout("too slow")

        requests.post = _timeout
        try:
            llm._call_model([{"role": "user", "content": "hi"}])
            assert False, "timeout should raise LLMError"
        except llm.LLMError as exc:
            assert "timed out" in str(exc).lower()
    finally:
        llm._secret, requests.post = original_secret, original_post


def test_call_model_5xx_is_friendly():
    import requests

    original_secret, original_post = llm._secret, requests.post
    try:
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        requests.post = lambda *a, **k: _FakeResponse(503)
        try:
            llm._call_model([{"role": "user", "content": "hi"}])
            assert False, "5xx should raise LLMError"
        except llm.LLMError as exc:
            assert "temporary error" in str(exc).lower()
    finally:
        llm._secret, requests.post = original_secret, original_post


def test_successful_call_returns_content():
    import requests

    original_secret, original_post = llm._secret, requests.post
    try:
        llm._secret = lambda name: "test-key" if name == "GROWTHLENS_LLM_API_KEY" else None
        requests.post = lambda *a, **k: _FakeResponse(200)
        assert llm._call_model([{"role": "user", "content": "hi"}]) == "ok"
    finally:
        llm._secret, requests.post = original_secret, original_post


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
