import json
import os

from agents.supervisor import STAGES, run_pipeline


def make_runner(calls, fail=None, skip_output=None, ml_warning=None):
    """A fake stage runner: records calls and writes each stage's output file."""
    by_module = {s["module"]: s for s in STAGES}

    def runner(module, dataset):
        calls.append(module)
        if module == fail:
            return 1, "", "Traceback...\nSystemExit: target column not found"
        out = by_module[module]["output"].format(n=dataset)
        if module != skip_output:
            if module == "spark_jobs.clean":
                os.makedirs(out, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                data = {}
                if module == "agents.ml_agent" and ml_warning:
                    data = {"warnings": [ml_warning]}
                with open(out, "w") as f:
                    json.dump(data, f)
        return 0, "", ""
    return runner


def go(tmp_path, monkeypatch, **kw):
    monkeypatch.chdir(tmp_path)
    calls = []
    runner = make_runner(calls, **{k: v for k, v in kw.items() if k != "exists"})
    report = run_pipeline("taxi", runner=runner, known=["taxi"],
                          exists=kw.get("exists", lambda m: False))
    return report, calls


def test_stages_run_in_order(tmp_path, monkeypatch):
    report, calls = go(tmp_path, monkeypatch)
    assert calls == ["agents.data_agent", "spark_jobs.clean",
                     "agents.eda_agent", "agents.ml_agent"]
    assert report["status"] == "ok"


def test_optional_insight_stage_is_skipped_when_missing(tmp_path, monkeypatch):
    report, _ = go(tmp_path, monkeypatch)
    assert report["stages"][-1]["status"] == "skipped"


def test_failed_stage_stops_pipeline_with_readable_message(tmp_path, monkeypatch):
    report, calls = go(tmp_path, monkeypatch, fail="agents.eda_agent")
    assert "agents.ml_agent" not in calls
    assert report["status"] == "failed"
    assert "target column not found" in report["message"]
    assert "Traceback" not in report["message"]
    assert [s["status"] for s in report["stages"][-2:]] == ["not_run", "not_run"]


def test_missing_output_file_is_a_failure(tmp_path, monkeypatch):
    report, _ = go(tmp_path, monkeypatch, skip_output="agents.data_agent")
    assert report["status"] == "failed"
    assert "did not write" in report["message"]


def test_ml_warning_is_reported(tmp_path, monkeypatch):
    report, _ = go(tmp_path, monkeypatch, ml_warning="ML stage stopped: too few rows")
    assert report["status"] == "warning"
    assert "too few rows" in report["message"]


def test_unknown_dataset_gives_message_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = run_pipeline("nope", known=["taxi", "flights"])
    assert report["status"] == "failed"
    assert "Unknown dataset" in report["message"]


def test_report_is_saved(tmp_path, monkeypatch):
    go(tmp_path, monkeypatch)
    with open("outputs/taxi/pipeline_run.json") as f:
        assert json.load(f)["dataset"] == "taxi"