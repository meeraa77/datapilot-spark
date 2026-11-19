import json
import os

import pytest

from agents.insight_agent import (allowed_numbers, build_facts, call_groq,
                                  pick_provider, run, unsupported_numbers)

PROFILE = {"row_count": 1000, "column_count": 5, "duplicate_count": 3,
           "columns": [{"name": "fare", "null_count": 0},
                       {"name": "tip", "null_count": 50}]}
EDA = {"row_count": 1000, "target": "fare",
       "top_correlations": [
           {"col_a": "dist", "col_b": "fare", "correlation": 0.88, "leakage": False},
           {"col_a": "total", "col_b": "fare", "correlation": 0.99, "leakage": True}],
       "outliers": {"dist": {"count": 80, "method": "IQR", "lower": 0, "upper": 9}},
       "top_categories": {"vendor": [{"value": "A", "count": 600}]}}
MODEL = {"problem_type": "regression", "target": "fare", "best_model": "Forest",
         "models": [{"name": "Linear", "metrics": {"rmse": 7.8282, "r2": 0.8028, "mae": 4.4847},
                     "train_seconds": 3.0},
                    {"name": "Forest", "metrics": {"rmse": 7.7772, "r2": 0.8053, "mae": 4.1376},
                     "train_seconds": 10.7}],
         "feature_importance": [{"feature": "dist", "importance": 0.6096},
                                {"feature": "Rate_oh_1", "importance": 0.1587}],
         "warnings": ["Sampled about 1000000 of 5714008 rows for training."]}


def text_of(facts):
    return " ".join(sum(facts.values(), []))


def write_inputs(tmp_path, monkeypatch, model=MODEL):
    monkeypatch.chdir(tmp_path)
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER",
                "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    os.makedirs("outputs/taxi")
    for name, data in [("profile", PROFILE), ("eda_results", EDA), ("model_results", model)]:
        if data is not None:
            with open(f"outputs/taxi/{name}.json", "w") as f:
                json.dump(data, f)


def test_facts_use_numbers_from_the_json():
    t = text_of(build_facts(PROFILE, EDA, MODEL))
    assert "correlation 0.88" in t and "7.7772" in t and "0.6096" in t
    assert "5.0%" in t            # 50 missing tips out of 1000 rows


def test_leakage_column_is_flagged_not_presented_as_a_finding():
    facts = build_facts(PROFILE, EDA, MODEL)
    t = text_of(facts)
    assert "not a real finding" in t
    assert "total is linked" not in t


def test_classification_no_better_than_majority_is_called_out():
    model = {"problem_type": "classification", "target": "late", "best_model": "LR",
             "majority_class_accuracy": 0.8214,
             "models": [{"name": "LR", "metrics": {"accuracy": 0.8213, "f1": 0.74, "auc": 0.6456},
                         "train_seconds": 8.4}], "warnings": []}
    t = text_of(build_facts(PROFILE, EDA, model))
    assert "no better than always predicting" in t
    assert "AUC is only 0.6456" in t


def test_missing_model_results_gives_a_limitation():
    facts = build_facts(PROFILE, EDA, None)
    assert any("No model results" in s for s in facts["limitations"])


def test_checker_accepts_rounded_percent_and_comma_numbers():
    allowed = allowed_numbers(build_facts(PROFILE, EDA, MODEL))
    ok = ["The forest has RMSE 7.78 and about 8.0% outliers.",
          "Sampled 1,000,000 of 5,714,008 rows.", "Importance of Rate_oh_1 is 0.16."]
    assert unsupported_numbers(ok, allowed) == []


def test_checker_rejects_invented_numbers():
    allowed = allowed_numbers(build_facts(PROFILE, EDA, MODEL))
    assert unsupported_numbers(["Fares rose 37.5% last year."], allowed) == ["37.5"]


GOOD = {"findings": ["Fare is closely tied to dist (correlation 0.88)."],
        "limitations": ["Only one train/test split was used."],
        "next_steps": ["Review outliers in dist."]}


def test_llm_answer_is_used_when_numbers_check_out(tmp_path, monkeypatch):
    write_inputs(tmp_path, monkeypatch)
    r = run("taxi", llm=lambda s, u: json.dumps(GOOD))
    assert r["mode"] == "llm" and r["findings"] == GOOD["findings"]


def test_invented_number_falls_back_to_template(tmp_path, monkeypatch):
    write_inputs(tmp_path, monkeypatch)
    bad = dict(GOOD, findings=["Fares are up 99.9% year on year."])
    calls = []

    def llm(s, u):
        calls.append(1)
        return json.dumps(bad)
    r = run("taxi", llm=llm)
    assert r["mode"] == "template" and "99.9" in r["fallback_reason"]
    assert len(calls) == 2                      # one retry, then give up


def test_llm_error_falls_back_to_template(tmp_path, monkeypatch):
    write_inputs(tmp_path, monkeypatch)

    def boom(s, u):
        raise TimeoutError("timed out")
    r = run("taxi", llm=boom)
    assert r["mode"] == "template" and "timed out" in r["fallback_reason"]


def test_no_api_key_uses_template_and_saves_file(tmp_path, monkeypatch):
    write_inputs(tmp_path, monkeypatch)
    r = run("taxi")
    assert r["mode"] == "template" and r["findings"]
    with open("outputs/taxi/insights.json") as f:
        assert json.load(f)["dataset"] == "taxi"


def test_missing_input_gives_readable_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        run("taxi")
    assert "not found" in str(e.value)


def test_placeholder_key_is_ignored(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "your-groq-api-key-here")
    assert pick_provider() is None


def test_provider_is_picked_from_the_key(monkeypatch):
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realkey")
    assert pick_provider() == "groq"


def test_generic_key_needs_explicit_provider(monkeypatch):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "somekey")
    assert pick_provider() is None
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert pick_provider() == "groq"


def test_groq_call_sends_bearer_key_and_reads_reply(monkeypatch):
    import io
    import urllib.request
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realkey")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return io.BytesIO(json.dumps(
            {"choices": [{"message": {"content": "{\"ok\": 1}"}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert call_groq("sys", "user") == '{"ok": 1}'
    assert seen["auth"] == "Bearer gsk_realkey"
    assert seen["body"]["messages"][0]["role"] == "system"