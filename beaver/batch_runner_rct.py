"""
Batch runner to be used for RCT experiments

Orchestrates running multiple experiments across multiple models:
  1. Runs each specified experiment on its corresponding model
  2. Organises logs as  {output_dir}/{model_name}/{experiment_name}/logs_*/

Batch YAML format:
  experiments: list of paths to experiment YAMLs (relative to the batch YAML)

Usage:
    beaver rct-batch --batch configs/batches/example_batch.yaml
    beaver rct-batch --batch configs/batches/example_batch.yaml --dry-run
"""

import concurrent.futures
import copy
import importlib.util
import inspect
import io
import json
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from beaver import run
from beaver.logging import (
    get_log_data,
    summarize_log_data,
    create_plots,
    create_time_plots,
    get_profile_data,
    summarize_profile_data,
)

# ── helpers ───────────────────────────────────────────────────────────────


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with *override* values merged on top of *base*."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _find_logs_dir(base_dir: Path):
    """Find the most recent logs_* subdirectory inside base_dir, if any."""
    logs_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")],
        reverse=True,
    )
    return logs_dirs[0] if logs_dirs else None


def is_experiment_completed(exp_base_dir: Path) -> bool:
    """Check whether a previous run of this experiment completed successfully.

    An experiment is considered complete if any ``logs_*/summary.json``
    exists under *exp_base_dir*.
    """
    if not exp_base_dir.is_dir():
        return False
    logs_dir = _find_logs_dir(exp_base_dir)
    return logs_dir is not None and (logs_dir / "summary.json").is_file()


def generate_summary(exp_base_dir: Path):
    """Run plot_logs summary on the most recent logs dir that has data."""
    if not exp_base_dir.is_dir():
        return
    logs_dir = _find_logs_dir(exp_base_dir)
    if logs_dir is None:
        return
    if (logs_dir / "summary.json").is_file():
        return  # already summarised
    try:
        all_data = get_log_data(logs_dir)
        all_profile_data = get_profile_data(logs_dir)
        if all_data:
            summarize_log_data(all_data, logs_dir)
            create_plots(all_data, logs_dir)
            create_time_plots(all_data, all_profile_data, logs_dir)
            summarize_profile_data(logs_dir)
            _log(f"  Generated summary: {logs_dir / 'summary.json'}")
    except Exception as e:
        _log(f"  Warning: failed to generate summary: {e}", "WARN")

# ── config loading ────────────────────────────────────────────────────────

# Structural keys in experiment YAML — never passed to load_prompts or run()
_YAML_META_KEYS = frozenset({
    "experiment_file",
    "load_prompts_fn",
    "constraint_fn",
    "check_call_fn",
    "instance_context_fn",
    "cache",
    "cache_dataset_name",
    "grammar",
    "semantic_symbol",
})

# Keys that go to beaver.run() as algo/config params
_BEAVER_RUN_KEYS = frozenset({
    "verifier",
    "gen_length",
    "temperature",
    "top_p",
    "top_k",
    "max_iterations",
    "epsilon",
    "max_workers",
    "num_logprobs",
    "max_frontier_size",
    "max_frontier_prob",
    "frontier_scoring_strategy",
    "use_grammar",
    "use_chat_template",
    "num_shots",
    "verbose",
    "log_dir",
    "glove_embed",
    "gpu_uuid",
    "model_type",
    "model_args",
    "vocab_size",
})


def load_experiment_config(exp_yaml_path: Path) -> dict[str, Any]:
    """Load experiment config from a file path."""
    if not exp_yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {exp_yaml_path}")
    cfg = load_yaml(exp_yaml_path)
    cfg["_yaml_path"] = str(exp_yaml_path)
    cfg["_yaml_dir"] = str(exp_yaml_path.parent)
    cfg.setdefault("_name", exp_yaml_path.stem)
    return cfg

# ── batch config ──────────────────────────────────────────────────────────


def load_batch_config(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    config.setdefault("output_dir", "./batch_results")
    config.setdefault("models", [])
    config.setdefault("experiments", [])
    config.setdefault("execution", {})
    return config

# ── experiment execution (in-process) ────────────────────────────────────

_import_lock = threading.Lock()


def _import_module_from_path(module_path: Path):
    """Dynamically import a Python file as a module.

    Uses the file's stem as the module name and adds its parent directory to
    sys.path so that spawned worker processes (which inherit sys.path) can
    re-import the module when unpickling constraint functions.

    Thread-safe: if two parallel experiments load the same file, only one
    import happens and both get the same module object.  This is required for
    pickle to work correctly — if two threads each create a fresh module for
    the same file, the second ``sys.modules[name] = mod`` clobbers the first,
    making the first thread's function objects un-picklable.
    """
    module_name = module_path.stem  # e.g. "enron" from "enron.py"
    parent_dir = str(module_path.parent.resolve())

    with _import_lock:
        if module_name in sys.modules:
            return sys.modules[module_name]

        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # Required for pickle to find it in workers
        spec.loader.exec_module(mod)

    return mod

def _get_load_prompts_kwargs(load_prompts_fn, exp_cfg: dict) -> dict:
    """Return kwargs for load_prompts_fn by introspecting its signature."""
    sig = inspect.signature(load_prompts_fn)
    skip = _YAML_META_KEYS | _BEAVER_RUN_KEYS
    result = {}
    for param_name in sig.parameters:
        if param_name in exp_cfg and param_name not in skip:
            val = exp_cfg[param_name]
            if val is not None:
                result[param_name] = val
    return result


EXPERIMENT_TIMEOUT: int = 60 * 60  # 60 minutes

def run_experiment(
    exp_cfg: dict,
    model_path: str,
    exp_base_dir: Path,
    model_type: str,
    model_config_path: str,
    dry_run: bool = False,
    verbose: bool = False,
    timeout: int = EXPERIMENT_TIMEOUT,
) -> bool:
    """Run a single experiment in-process.

    *exp_base_dir* is the experiment-level directory, e.g.
    ``batch_results/model/enron``. Logs go to ``exp_base_dir/logs_*/``
    (created by ``beaver.run()`` via ``new_log_dir()``).
    """
    exp_name = exp_cfg["_name"]
    exp_dir = Path(exp_cfg["_yaml_dir"])

    _log(f"Running experiment: {exp_name}")

    if dry_run:
        _log(f"  [DRY RUN] Would run: {exp_name} for model {model_path}")
        return True
    
    # ── Import experiment module ───────────────────────────────────────────
    exp_file = exp_dir / exp_cfg["experiment_file"]
    if not exp_file.exists():
        _log(f"  Error: experiment file not found: {exp_file}", "ERROR")
        return False
    
    try:
        mod = _import_module_from_path(exp_file)
    except Exception as e:
        _log(f"  Error importing {exp_file}: {e}", "ERROR")
        return False
    
    load_prompts_fn = getattr(mod, exp_cfg["load_prompts_fn"])
    constraint_fn = getattr(mod, exp_cfg["constraint_fn"])
    check_call_fn_name = exp_cfg.get("check_call_fn")
    check_call_fn = getattr(mod, check_call_fn_name) if check_call_fn_name else None
    instance_context_fn_name = exp_cfg.get("instance_context_fn")
    instance_context_fn = getattr(mod, instance_context_fn_name) if instance_context_fn_name else None

    # ── Build load_prompts kwargs ──────────────────────────────────────────
    load_kwargs = _get_load_prompts_kwargs(load_prompts_fn, exp_cfg)
    if verbose:
        _log(f"  load_prompts kwargs: {load_kwargs}")
    
    # ── Build beaver.run() kwargs ──────────────────────────────────────────
    run_kwargs = {
        k: exp_cfg[k]
        for k in _BEAVER_RUN_KEYS
        if k in exp_cfg and exp_cfg[k] is not None
    }

    # Override log_dir to point to exp_base_dir
    run_kwargs["log_dir"] = str(exp_base_dir)

    # Override model_type to that passed into the batch runner
    run_kwargs["model_type"] = model_type

    # Override model_args to be model_config_path
    run_kwargs["model_args"] = model_config_path

    exp_base_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment config for debugging
    with open(exp_base_dir / "experiment_config.json", "w") as f:
        json.dump(
            {k: v for k, v in exp_cfg.items() if not k.startswith("_")}, f, indent=2
        )

    # ── Run in a thread with timeout ───────────────────────────────────────
    result: dict = {"ok": False, "error": None}

    def _run():
        try:

            prompts = load_prompts_fn(**load_kwargs)
            run(
                prompts=prompts,
                constraint_fn=constraint_fn,
                check_call_fn=check_call_fn,
                cache=exp_cfg.get("cache", False),
                cache_dataset_name=exp_cfg.get("cache_dataset_name"),
                instance_context_fn=instance_context_fn,
                grammar=exp_cfg.get("grammar"),
                semantic_symbol=exp_cfg.get("semantic_symbol"),
                model=model_path,
                **run_kwargs,
            )
            result["ok"] = True
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        _log(
            f"  Experiment {exp_name} timed out after {timeout}s",
            "ERROR",
        )
        return False

    if result["ok"]:
        _log(f"  Experiment {exp_name} completed successfully")
        return True

    _log(f"  Experiment {exp_name} failed: {result['error']}", "ERROR")
    return False

# ── logging ───────────────────────────────────────────────────────────────


_log_fh: io.TextIOWrapper | None = None
_log_lock = threading.Lock()


def _log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    with _log_lock:
        print(line)
        if _log_fh is not None:
            _log_fh.write(line + "\n")
            _log_fh.flush()

# ── main loop ─────────────────────────────────────────────────────────────

def run_batch(
    batch_cfg: dict,
    batch_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    global _log_fh

    # define directories
    output_dir = Path(batch_cfg["output_dir"])
    batch_dir = batch_path.parent

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_log_path = output_dir / f"batch_runner_{datetime.now():%Y%m%d_%H%M%S}.log"
        _log_fh = open(batch_log_path, "w")
    
    # pull info from config
    exec_cfg = batch_cfg["execution"]

    stop_on_failure = exec_cfg.get("stop_on_failure", False)
    cooldown = exec_cfg.get("cooldown_between_experiments", 10)
    skip_completed = exec_cfg.get("skip_completed", False)

    model_paths: list[str] = batch_cfg["models"]
    model_config_paths: list[str] = batch_cfg["model_configs"]
    exp_paths: list[str] = batch_cfg["experiments"]
    model_type: str = batch_cfg["model_type"]

    # log config info
    _log("=" * 60)
    _log("Starting batch run")
    _log(f"  Output dir    : {output_dir}")
    _log(f"  Models        : {model_paths}")
    _log(f"  Model Configs : {model_config_paths}")
    _log(f"  Model Type    : {model_type}")
    _log(f"  Experiments   : {exp_paths}")
    _log("=" * 60)

    # Pre-load all experiment configs (fail fast if any are missing)
    exp_cfgs: list[dict] = []
    for exp_path in exp_paths:
        abs_path = (batch_dir / exp_path).resolve()
        exp_cfgs.append(load_experiment_config(abs_path))
    
    results: list[dict] = []
    t_batch = time.time()
    total_experiments = len(model_paths)
    completed_experiments = 0

    #FIXME: a lazy approach here, expects index of model and experiment name to match so just iterate
    # but a more dynamic approach could be used instead
    for i in range(len(exp_cfgs)):
        _log("-" * 40)

        # retrieve model / experiment details
        model_path = Path(model_paths[i])
        model_config_path = model_config_paths[i]
        exp_cfg = exp_cfgs[i]
        #FIXME: there has to be a more elegant way of doing this .-.
        model_name = str(Path(model_path.parent).stem + "/" + model_path.stem + "/" + model_type + "/logits")

        _log(
            f"Model [{i+1}/{len(model_paths)}]: {model_name}  (config: {model_config_path})"
        )

        model_log_dir = output_dir / model_name
        print(f"[DEBUG] output_dir: {output_dir}, model_name: {model_name}, model_log_dir: {model_log_dir}")
        exp_name = exp_cfg["_name"]
        exp_log_dir = model_log_dir

        # skip those that have already been done if requested
        if skip_completed:
            done = is_experiment_completed(model_log_dir)
            if done:
                _log(f"  The experiment is already completed for {model_name} — skipping model")
                completed_experiments += 1
                results.append(
                    {
                        "model": model_name,
                        "experiment": model_path.parent,
                        "success": True,
                        "log_dir": str(model_log_dir),
                        "skipped": True,
                    }
                )
                continue

        _log(
            f"  Experiment [{i+1}/{len(exp_cfgs)}] "
            f"(overall {completed_experiments}/{total_experiments}): "
            f"{exp_name}"
        )
        
        # run the experiment with the model path and experiment config
        ok = run_experiment(
            exp_cfg,
            str(model_path),
            exp_log_dir,
            model_type,
            model_config_path,
            dry_run=dry_run,
            verbose=verbose,
        )

        results.append(
            {
                "model": model_name,
                "experiment": exp_name,
                "success": ok,
                "log_dir": str(exp_log_dir),
            }
        )

        if not ok and stop_on_failure:
            _log("Stopping batch due to failure", "ERROR")
            return False

        if cooldown > 0:
            time.sleep(cooldown)
    
    # ── summary ───────────────────────────────────────────────────────────
    duration = time.time() - t_batch
    ok_count = sum(1 for r in results if r["success"])
    total = len(results)

    _log("=" * 60)
    _log("Batch complete")
    _log(f"  Time: {duration:.1f}s")
    skipped_count = sum(1 for r in results if r.get("skipped"))
    for r in results:
        if r.get("skipped"):
            tag = "SKIP"
        elif r["success"]:
            tag = "OK"
        else:
            tag = "FAIL"
        _log(f"    [{tag}] {r['model']} / {r['experiment']}")
    _log(f"  {ok_count}/{total} succeeded ({skipped_count} skipped)")
    _log("=" * 60)

    if not dry_run:
        summary_path = output_dir / f"batch_summary_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(
                {
                    "duration_seconds": round(duration, 1),
                    "success_count": ok_count,
                    "total_count": total,
                    "results": results,
                },
                f,
            )
        _log(f"Summary: {summary_path}")

    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None

    return ok_count == total