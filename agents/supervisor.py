"""Supervisor Agent: runs every pipeline stage in order for one dataset.

Run:  python -m agents.supervisor <dataset>

Stages (each runs as its own process, the same as running it by hand):
  1. python -m agents.data_agent <dataset>   -> outputs/<dataset>/profile.json
  2. python -m spark_jobs.clean <dataset>    -> data/processed/<dataset>/
  3. python -m agents.eda_agent <dataset>    -> outputs/<dataset>/eda_results.json
  4. python -m agents.ml_agent <dataset>     -> outputs/<dataset>/model_results.json
  5. python -m agents.insight_agent <dataset> -> outputs/<dataset>/insights.json
     (skipped automatically until agents/insight_agent.py exists)

A failed stage stops the pipeline with a readable message, never a stack trace.
The result of every run is saved to outputs/<dataset>/pipeline_run.json.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

STAGES = [
    {"key": "profile", "title": "Data Agent (profile)", "module": "agents.data_agent",
     "output": "outputs/{n}/profile.json", "optional": False},
    {"key": "clean", "title": "Spark cleaning", "module": "spark_jobs.clean",
     "output": "data/processed/{n}", "optional": False},
    {"key": "eda", "title": "EDA Agent", "module": "agents.eda_agent",
     "output": "outputs/{n}/eda_results.json", "optional": False},
    {"key": "ml", "title": "ML Agent", "module": "agents.ml_agent",
     "output": "outputs/{n}/model_results.json", "optional": False},
    {"key": "insights", "title": "Insight Agent", "module": "agents.insight_agent",
     "output": "outputs/{n}/insights.json", "optional": True},
]


# ---------------------------------------------------------------- helpers
def known_datasets():
    from spark_jobs.datasets import DATASETS   # imported late: needs pyspark
    return list(DATASETS)


def module_exists(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def run_stage(module, dataset):
    """Run one stage in its own process. Returns (returncode, stdout, stderr)."""
    proc = subprocess.run([sys.executable, "-m", module, dataset],
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def short_error(stdout, stderr):
    """One readable line explaining a failure (no Spark log noise, no traceback)."""
    for text in (stderr, stdout):
        lines = [ln.strip() for ln in (text or "").splitlines()
                 if ln.strip() and " WARN " not in ln and "SUCCESS:" not in ln]
        if lines:
            return lines[-1][:300]
    return "the stage exited with an error and printed no message"


def output_ready(path, started):
    """A stage output counts if it exists (files must also be newer than the stage start)."""
    if not os.path.exists(path):
        return False
    if os.path.isdir(path):
        return True
    return os.path.getmtime(path) >= started - 1


def ml_warning(dataset):
    """The ML Agent never crashes; it records problems in model_results.json."""
    try:
        with open(f"outputs/{dataset}/model_results.json") as f:
            warnings = json.load(f).get("warnings", [])
    except (OSError, ValueError):
        return None
    stopped = [w for w in warnings if "ML stage stopped" in w]
    return stopped[0] if stopped else None


# ---------------------------------------------------------------- pipeline
def run_pipeline(dataset, on_progress=None, stages=STAGES, runner=run_stage,
                 exists=module_exists, known=None):
    """Run all stages for `dataset` (a name from spark_jobs/datasets.py).

    on_progress(stage_title, status, report) is called as each stage starts
    ('running') and ends, so a dashboard can show progress.
    Returns a report dict: {"dataset", "status", "message", "seconds", "stages": [...]}.
    status is ok, warning or failed.
    """
    def notify(title, status):
        if on_progress:
            on_progress(title, status, report)

    names = known if known is not None else known_datasets()
    report = {"dataset": dataset, "status": "ok", "message": "", "seconds": 0.0,
              "stages": []}
    if dataset not in names:
        report["status"] = "failed"
        report["message"] = f"Unknown dataset '{dataset}'. Choose from: {names}"
        return report

    t0 = time.perf_counter()
    failed = False
    for st in stages:
        entry = {"key": st["key"], "title": st["title"], "status": "not_run",
                 "seconds": 0.0, "message": ""}
        report["stages"].append(entry)
        if failed:
            continue                      # an earlier stage failed: do not run later ones
        if st["optional"] and not exists(st["module"]):
            entry["status"], entry["message"] = "skipped", "not built yet"
            notify(st["title"], "skipped")
            continue

        notify(st["title"], "running")
        started = time.time()
        t = time.perf_counter()
        code, out, err = runner(st["module"], dataset)
        entry["seconds"] = round(time.perf_counter() - t, 1)
        path = st["output"].format(n=dataset)

        if code != 0:
            entry["status"] = "failed"
            entry["message"] = f"{st['title']} failed: {short_error(out, err)}"
        elif not output_ready(path, started):
            entry["status"] = "failed"
            entry["message"] = f"{st['title']} finished but did not write {path}"
        else:
            entry["status"] = "ok"
            if st["key"] == "ml":
                warn = ml_warning(dataset)
                if warn:
                    entry["status"], entry["message"] = "warning", warn

        if entry["status"] == "failed":
            failed = True
            report["status"], report["message"] = "failed", entry["message"]
        elif entry["status"] == "warning" and report["status"] == "ok":
            report["status"], report["message"] = "warning", entry["message"]
        notify(st["title"], entry["status"])

    report["seconds"] = round(time.perf_counter() - t0, 1)
    if report["status"] == "ok":
        report["message"] = "All stages finished."

    os.makedirs(f"outputs/{dataset}", exist_ok=True)
    with open(f"outputs/{dataset}/pipeline_run.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def print_progress(title, status, report):
    if status == "running":
        print(f"... {title}", flush=True)
    else:
        print(f"[{status.upper()}] {title}", flush=True)


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "taxi"
    result = run_pipeline(dataset_name, on_progress=print_progress)
    print(f"\n{result['status'].upper()} in {result['seconds']} s: {result['message']}")
    sys.exit(0 if result["status"] != "failed" else 1)