"""TPM-OPT command-line interface.

Sub-commands:
  tpm-opt run       — runs an optimization sequence 
"""

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path
import os
import torch
import yaml

# Keys in experiment YAML that are meta/structural (never passed to load_prompts or run)
_YAML_META_KEYS = frozenset(
    {
        "experiment_file",
        "load_prompts_fn",
        "constraint_fn",
        "check_call_fn",
        "instance_context_fn",
        "cache",
        "cache_dataset_name",
        "grammar",
        "semantic_symbol",
    }
)

# Keys that are beaver.run() algorithm params
_BEAVER_RUN_KEYS = frozenset(
    {
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
    }
)

_TPM_OPT_RUN_KEYS = frozenset(
    {
        "code_path",
        "dataset",
        "num_instances",
        "num_children",
        "max_steps"
    }
)

# ────── Utility Functions ────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}

def _import_module_from_path(module_path: Path):
    """Dynamically import a Python file as a module.

    Uses the file's stem as the module name and adds its parent directory to
    sys.path so that spawned worker processes (which inherit sys.path) can
    re-import the module when unpickling constraint functions.
    """
    import sys

    parent_dir = str(module_path.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # Required for pickle to find it in workers
    spec.loader.exec_module(mod)
    return mod

def _get_load_prompts_kwargs(load_prompts_fn, merged_cfg: dict, extra: dict) -> dict:
    """Return kwargs to pass to load_prompts_fn by introspecting its signature."""
    sig = inspect.signature(load_prompts_fn)
    skip = _YAML_META_KEYS | _BEAVER_RUN_KEYS | _TPM_OPT_RUN_KEYS
    result = {}
    for param_name in sig.parameters:
        if param_name in extra:
            result[param_name] = extra[param_name]
        elif param_name in merged_cfg and param_name not in skip:
            result[param_name] = merged_cfg[param_name]
    return result


def _get_run_kwargs_beaver(merged_cfg: dict) -> dict:
    """Extract beaver.run() kwargs from merged config."""
    return {
        k: v for k, v in merged_cfg.items() if k in _BEAVER_RUN_KEYS and v is not None
    }

def _get_run_kwargs_tpm_opt(merged_cfg: dict) -> dict:
    """Extract beaver.run() kwargs from merged config."""
    return {
        k: v for k, v in merged_cfg.items() if k in _TPM_OPT_RUN_KEYS and v is not None
    }

# ──── Main Loop ───────────────────────────────────────────────

def _run_cmd(argv):
    parser = argparse.ArgumentParser(
        prog="tpm-opt run",
        description="Run a single optimization sequence.",
    )

    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment YAML (e.g. experiments/gsm_symbolic/experiment.yaml)")

    # Optimizer arguments
    parser.add_argument("--code_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_instances", type=int, default=100)
    parser.add_argument("--num_children", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=5)

    # BEAVER arguments
    parser.add_argument("--glove_embed", type=int, default=None)
    parser.add_argument("--gpu_uuid", type=str, default=None)
    parser.add_argument("--vocab_size", type=str, default=None)
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    # Data slicing (forwarded to load_prompts_fn)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--debug_ids", type=str, default=None)

    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--num_logprobs", type=int, default=None)
    parser.add_argument("--max_frontier_size", type=int, default=None)
    parser.add_argument("--max_frontier_prob", type=float, default=None)
    parser.add_argument(
        "--frontier_scoring_strategy",
        type=str,
        default=None,
        choices=["highest-prob", "length-bias", "random-select", "sample-select"],
    )
    parser.add_argument(
        "--use_grammar",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
    )
    parser.add_argument(
        "--use_chat_template",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
    )
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args(argv)

    # ── Load experiment YAML ───────────────────────────────────────────────
    exp_yaml_path = Path(args.experiment).resolve()
    if not exp_yaml_path.exists():
        print(f"Error: experiment YAML not found: {exp_yaml_path}", file=sys.stderr)
        sys.exit(1)

    exp_dir = exp_yaml_path.parent
    cfg = _load_yaml(exp_yaml_path)

    # ── Import experiment module ───────────────────────────────────────────
    exp_file = exp_dir / cfg["experiment_file"]
    if not exp_file.exists():
        print(f"Error: experiment file not found: {exp_file}", file=sys.stderr)
        sys.exit(1)

    mod = _import_module_from_path(exp_file)

    load_prompts_fn = getattr(mod, cfg["load_prompts_fn"])
    constraint_fn = getattr(mod, cfg["constraint_fn"])
    check_call_fn = (
        getattr(mod, cfg.get("check_call_fn") or "", None)
        if cfg.get("check_call_fn")
        else None
    )
    instance_context_fn = (
        getattr(mod, cfg.get("instance_context_fn") or "", None)
        if cfg.get("instance_context_fn")
        else None
    )

    # ── Merge YAML params with explicit CLI flags (CLI wins) ───────────────
    merged_cfg = dict(cfg)
    #FIXME: worried this might crash bc we are not including all args from beaver cli in this one
    cli_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k in ( _BEAVER_RUN_KEYS | _TPM_OPT_RUN_KEYS )
    }
    merged_cfg.update(cli_overrides)

    # ── Build load_prompts kwargs ──────────────────────────────────────────
    load_kwargs = _get_load_prompts_kwargs(load_prompts_fn, merged_cfg, {})

    # ── Load prompts ───────────────────────────────────────────────────────
    prompts = load_prompts_fn(**load_kwargs)

    # ── Build beaver.run() kwargs ──────────────────────────────────────────
    beaver_run_kwargs = _get_run_kwargs_beaver(merged_cfg)

    # ── Build tpm_opt.run() kwargs ─────────────────────────────────────────
    tpm_opt_run_kwargs = _get_run_kwargs_tpm_opt(merged_cfg)

    # ── Write additional beaver.run() kwargs ───────────────────────────────
    beaver_run_kwargs["model_type"] = "code"
    beaver_run_kwargs["verifier"] = "frontier"
    beaver_run_kwargs["constraint_fn"] = constraint_fn
    beaver_run_kwargs["check_call_fn"] = check_call_fn
    beaver_run_kwargs["cache"] = cfg.get("cache", False)
    beaver_run_kwargs["cache_dataset_name"] = cfg.get("cache_dataset_name")
    beaver_run_kwargs["instance_context_fn"] = instance_context_fn
    beaver_run_kwargs["grammar"] = cfg.get("grammar")
    beaver_run_kwargs["semantic_symbol"] = cfg.get("semantic_symbol")
    
    # ── Call beaver.run() ──────────────────────────────────────────────────
    # import beaver

    # # set GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid or ""
    # if torch.cuda.is_available():
    #     print(f"[DEBUG] Using GPU {torch.cuda.get_device_name()} with properties {torch.cuda.get_device_properties()}")
    # else:
    #     print(f"[INFO] CUDA is unavailable, using CPU")

    # beaver.run(
    #     prompts=prompts,
    #     constraint_fn=constraint_fn,
    #     check_call_fn=check_call_fn,
    #     cache=cfg.get("cache", False),
    #     cache_dataset_name=cfg.get("cache_dataset_name"),
    #     instance_context_fn=instance_context_fn,
    #     grammar=cfg.get("grammar"),
    #     semantic_symbol=cfg.get("semantic_symbol"),
    #     model=args.model,
    #     **run_kwargs,
    # )

    print(f"[DEBUG] beaver_run_kwargs: {beaver_run_kwargs}")
    print(f"[DEBUG] tpm_opt_run_kwargs: {tpm_opt_run_kwargs}")

    return


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="tpm-opt",
        description="TPM-OPT: An optimization architecture that utilizes BEAVER for evaluation",
    )
    parser.add_argument(
        "command",
        choices=["run"],
        help="Sub-command to execute.",
    )

    if not argv:
        parser.print_help()
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "run":
        _run_cmd(rest)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

    