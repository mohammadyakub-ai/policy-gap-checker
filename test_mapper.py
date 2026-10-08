"""Offline tests: no network, no GROQ_API_KEY or GROQ_MODEL required."""
import json
import os

import openai
import pytest
from pydantic import ValidationError

import mapper

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["ACCESS", "ENCRYPT", "LOGGING", "INCIDENT", "BACKUP", "VENDOR", "CHANGE", "TRAINING", "NONE"]


class _FakeResponse:
    """Minimal stand-in for an httpx response (openai exceptions need it)."""
    status_code = 404
    headers = {"x-request-id": "req_test"}
    request = "fake-request"


def _not_found():
    return openai.NotFoundError("model_not_found", response=_FakeResponse(), body=None)


def _client_raising(exc):
    """Build a fake client whose create() always raises `exc`."""

    class FakeCompletions:
        calls = 0

        def create(self, **kwargs):
            FakeCompletions.calls += 1
            raise exc

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    return FakeClient, FakeCompletions


def test_validator_rejects_invalid_control():
    with pytest.raises(ValidationError):
        mapper.Mapping(control="NOT_A_REAL_CONTROL", reason="because")


def test_validator_rejects_unknown_control_foo():
    with pytest.raises(ValidationError):
        mapper.Mapping(control="FOO", reason="because")


def test_validator_accepts_error_sentinel():
    assert mapper.Mapping(control="ERROR", reason="failed").control == "ERROR"


@pytest.mark.parametrize("key", KEYS)
def test_validator_accepts_each_valid_control(key):
    m = mapper.Mapping(control=key, reason="fits")
    assert m.control == key
    assert m.reason == "fits"


def test_system_prompt_never_offers_error():
    assert "ERROR" not in mapper.SYSTEM


def test_load_paragraphs_from_sample_policy():
    path = os.path.join(HERE, "sample_policy.txt")
    paras = mapper.load_paragraphs(path)
    assert paras, "expected at least one paragraph"
    assert all(len(p) > 60 for p in paras)


def test_load_paragraphs_filters_short_chunks(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("too short\n\n" + "x" * 60 + "\n\n" + "y" * 61 + "\n", encoding="utf-8")
    assert mapper.load_paragraphs(str(path)) == ["y" * 61]


def test_classify_falls_back_after_retries(monkeypatch):
    """A client that always raises a soft error must yield ERROR, not NONE."""
    FakeClient, _ = _client_raising(RuntimeError("simulated API error"))
    monkeypatch.setattr(mapper, "_client", lambda: FakeClient())
    m = mapper.classify("Employees must use MFA on every remote login to the company VPN.", retries=2)
    assert m.control == "ERROR"
    assert m.reason.startswith("failed after retries")


def test_classify_reraises_not_found_without_retry(monkeypatch):
    """A 404 model_not_found must surface immediately, with a single attempt."""
    FakeClient, completions = _client_raising(_not_found())
    monkeypatch.setattr(mapper, "_client", lambda: FakeClient())
    with pytest.raises(openai.NotFoundError):
        mapper.classify("Employees must use MFA on every remote login to the company VPN.")
    assert completions.calls == 1


def test_eval_accuracy_excludes_errors(monkeypatch, capsys):
    """Errors leave the denominator: 4 right, 4 wrong, 10 errors -> 4/8."""
    with open(os.path.join(HERE, "labeled.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    answers = []
    for i, item in enumerate(data):
        if i < 4:
            answers.append(item["control"])  # correct
        elif i < 8:
            answers.append("NONE" if item["control"] != "NONE" else "ACCESS")  # wrong
        else:
            answers.append("ERROR")
    it = iter(answers)
    monkeypatch.setattr(
        mapper, "classify",
        lambda text, retries=2: mapper.Mapping(control=next(it), reason="stub"))
    mapper.run_eval()
    out = capsys.readouterr().out
    assert "Evaluating: labeled.json" in out
    assert "Accuracy: 4/8 = 50% (excluding 10 errors)" in out
    assert "Errors: 10 of 18" in out
    assert "WARNING: results incomplete, do not report this accuracy." in out
    assert out.count("expected ") == 4  # only real wrong predictions are listed


def test_parse_args_eval_default_path():
    """`--eval` with no path targets labeled.json next to mapper.py."""
    mode, path = mapper.parse_args(["--eval"])
    assert mode == "eval"
    assert path == mapper.DEFAULT_EVAL
    assert path.endswith("labeled.json")


def test_parse_args_eval_custom_path():
    assert mapper.parse_args(["--eval", "labeled_hard.json"]) == ("eval", "labeled_hard.json")


def test_parse_args_report_list_models_and_usage():
    assert mapper.parse_args(["sample_policy.txt"]) == ("report", "sample_policy.txt")
    assert mapper.parse_args(["--list-models"]) == ("list-models", None)
    assert mapper.parse_args([]) == (None, None)
    assert mapper.parse_args(["--eval", "a.json", "b.json"]) == (None, None)  # too many paths


def test_main_routes_eval_paths(monkeypatch):
    """main() hands the parsed path to run_eval without touching the API."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    seen = []
    monkeypatch.setattr(mapper, "run_eval", lambda path=None: seen.append(path))
    monkeypatch.setattr(mapper.sys, "argv", ["mapper.py", "--eval"])
    mapper.main()
    monkeypatch.setattr(mapper.sys, "argv", ["mapper.py", "--eval", "custom_set.json"])
    mapper.main()
    assert seen[0] == mapper.DEFAULT_EVAL
    assert seen[1] == "custom_set.json"


def test_run_eval_missing_file_exits_with_clear_error(capsys):
    with pytest.raises(SystemExit) as exc:
        mapper.run_eval("does_not_exist.json")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "eval file not found" in out
    assert "does_not_exist.json" in out
    assert "Traceback" not in out


def test_run_eval_ignores_note_field(tmp_path, monkeypatch, capsys):
    """Extra JSON fields such as "note" must not break evaluation."""
    path = tmp_path / "with_notes.json"
    path.write_text(json.dumps([
        {"text": "first item", "control": "ACCESS", "note": "mentions two controls"},
        {"text": "second item", "control": "NONE", "note": "definitions paragraph"},
    ]), encoding="utf-8")
    monkeypatch.setattr(mapper, "classify",
                        lambda text, retries=2: mapper.Mapping(control="ACCESS", reason="stub"))
    mapper.run_eval(str(path))
    out = capsys.readouterr().out
    assert "Evaluating: with_notes.json" in out
    assert "Accuracy: 1/2 = 50% (excluding 0 errors)" in out
    assert "Errors: 0 of 2" in out
