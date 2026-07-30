"""One experiment = one config + one results folder. Also the sweep runner.

Two roles, deliberately in one module:

*Imported* by the stage scripts, it resolves the active config and every output
path from the environment, so `results/<experiment_id>/` replaces the hardcoded
`results/metrics` and `results/retrieval` the stages used to write to.

*Run* as a script, it expands a sweep of config values into experiments, and
drives the stages for each one:

    python scripts/experiment.py --sweep-json '{"mutual_knn_k": [4, 10, 20]}'
    python scripts/experiment.py --id knn10 --set mutual_knn_k=10
    python scripts/experiment.py --id 00_prepare --stages 01

Everything an experiment produces lands under results/<id>/: config.yaml (the
merged config actually consumed), manifest.json (git SHA, overrides, resolved
dirs, per-stage status/duration), logs/ (one file per stage plus run.log), and
the retrieval/ and metrics/ outputs of stages 2-5.

Why the environment rather than a --config flag: the stages, and
src/graph/semantic_graph.py, read the config and build output paths at IMPORT
time, so no flag could reach all of them anyway. Env also inherits through
subprocess for free.

Environment (all optional):
  GRAPHS_PROJECT_EXPERIMENT_ID  Folder name under results/. Default "default".
  GRAPHS_PROJECT_CONFIG         YAML to load. Default config/base.yaml.
  GRAPHS_PROJECT_DATA_ROOT      Where the caches live. Default <repo>/data.
                                Point it at mounted Drive to survive a Colab
                                restart.
  GRAPHS_PROJECT_PROCESSED_DIR  Exact processed dir, bypassing fingerprinting.
  GRAPHS_PROJECT_GRAPHS_DIR     Exact graph dir, bypassing fingerprinting.

The caches under data/ are keyed by a fingerprint of the config keys that decide
their contents, not by experiment id: two experiments differing only in a
stage-3 knob share one set of graphs, while a change to mutual_knn_k or
metadata_graph.fields gets its own folder.

Caveat: editing src/graph/*.py does not change the fingerprint, so a stale graph
dir can be reused - the manifest records the git SHA and dirty flag, and
--force-graphs rebuilds. The SHA is deliberately not in the fingerprint; that
would throw away all reuse after every commit.
"""
from __future__ import annotations

import argparse
import codecs
import copy
import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # so `import src...` works in every stage

import yaml  # noqa: E402  (after the sys.path fix, deliberately)

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

# Config keys that decide what lands in data/. batch_size is excluded on purpose:
# it changes speed, not content. pcst.* and dense_baseline.* are excluded because
# they only affect stage 3 onwards, whose outputs are per-experiment anyway.
#
# metadata_graph belongs in the PREPROC set because scripts/01 embeds the
# metadata records (metadata_graph.fields decides what goes into them).
# mutual_knn_k is a top-level scalar, shared by all three graph builders.
PREPROC_KEYS = ("embedding", "ner", "metadata_graph")
GRAPH_KEYS = PREPROC_KEYS + ("mutual_knn_k", "entity_graph")
_VOLATILE_KEYS = ("batch_size",)

# Stage 3 is the monolith, not the nine split scripts: it is the only one that
# applies pcst.combined_cost_e, and it loads the graphs, computes the similarity
# matrix and calibrates seed_k once for all ten conditions instead of once per
# process. A single condition is still reachable as `--stages 03_entity_pcst`.
STAGES: dict[str, list[str]] = {
    "01": ["01_prepare_data.py"],
    "02": ["02_build_graphs.py"],
    "03": ["03_run_retrieval.py"],
    "04": ["04_evaluate.py"],
    "05": ["05_analyze_results.py"],
}
DEFAULT_STAGES = "01,02,03,04"  # 05 is opt-in: it rebuilds graphs nine times


class Paths(NamedTuple):
    experiment_id: str
    config_path: Path
    processed: Path
    graphs: Path
    results: Path
    retrieval: Path
    metrics: Path
    logs: Path


# --------------------------------------------------------------------------- #
# Resolution helpers - used both by the stages (via the module constants below)
# and by the runner, which re-resolves explicitly after parsing its arguments.
# --------------------------------------------------------------------------- #

def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fingerprint(cfg: dict, keys: tuple[str, ...]) -> str:
    """Short, stable digest of the config values that change files on disk.

    Each key is either a section (a dict, whose volatile keys are dropped) or a
    top-level scalar such as mutual_knn_k, which is taken as it stands.
    """
    payload = {}
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in _VOLATILE_KEYS}
        payload[key] = value
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def validate_experiment_id(experiment_id: str) -> str:
    if not _ID_RE.match(experiment_id):
        raise ValueError(
            f"invalid experiment id {experiment_id!r}: start with a letter or digit and "
            "use only letters, digits, dot, dash or underscore - it becomes a folder name"
        )
    return experiment_id


def resolve(experiment_id: str, config_path: str | Path, cfg: dict) -> Paths:
    """Where this experiment reads and writes. Pure: no mkdir, no logging."""
    validate_experiment_id(experiment_id)

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    data_root = Path(os.environ.get("GRAPHS_PROJECT_DATA_ROOT") or (ROOT / "data"))
    processed = Path(os.environ.get("GRAPHS_PROJECT_PROCESSED_DIR")
                     or data_root / "processed" / f"p-{fingerprint(cfg, PREPROC_KEYS)}")
    graphs = Path(os.environ.get("GRAPHS_PROJECT_GRAPHS_DIR")
                  or data_root / "graphs" / f"g-{fingerprint(cfg, GRAPH_KEYS)}")
    results = ROOT / "results" / experiment_id

    return Paths(experiment_id, config_path, processed, graphs,
                 results, results / "retrieval", results / "metrics", results / "logs")


def ensure_dirs() -> None:
    """Create this experiment's output dirs. Called by each stage's main().

    Not done at import time so that importing this module - which the runner
    does to itself - has no side effects on the filesystem.
    """
    for d in (PROCESSED, GRAPHS, RETRIEVAL, METRICS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> None:
    # UTF-8 first: src/data/embedder.py logs "Computing embeddings -> path" with
    # a real arrow, which raises UnicodeEncodeError on a cp1252 Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "filelock", "sentence_transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Sweep expansion - importable, so a notebook can preview the matrix before
# spending an hour of GPU time on it.
# --------------------------------------------------------------------------- #

def _slug(dotted: str, value) -> str:
    """`mutual_knn_k` + 10 -> `mutual_knn_k10`; `metadata_graph.fields` +
    ["title", "author"] -> `fields-title-author`."""
    if value is None:
        text = "null"
    elif isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, (list, tuple)):
        text = "-" + "-".join(str(v) for v in value)
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{dotted.split('.')[-1]}{text}")


def _arm_id(overrides: dict, prefix: str = "") -> str:
    name = "__".join(_slug(k, v) for k, v in overrides.items()) or "baseline"
    if prefix:
        name = f"{prefix}__{name}"
    if len(name) > 64 or not _ID_RE.match(name):
        # Keep it a legal folder name without losing which arm it is.
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{re.sub(r'[^A-Za-z0-9._-]+', '-', name)[:55]}-{digest}"
    return name


def expand_sweep(sweep: dict[str, list], mode: str = "grid",
                 include_baseline: bool = False, prefix: str = "") -> list[tuple[str, dict]]:
    """Turn {dotted key: [values]} into [(experiment_id, overrides), ...].

    mode "grid" takes the cartesian product: every combination of every value.
    mode "axis" varies one key at a time, leaving the others at their config
    default - N+M arms instead of N*M, which is what you want for a first look
    at several knobs.

    Ids are derived from the overrides (`mutual_knn_k10__threshold0.65`), so the
    results folder says what it contains. Duplicate arms are dropped, keeping
    their first occurrence.
    """
    if mode not in ("grid", "axis"):
        raise ValueError(f"unknown sweep mode {mode!r}: use 'grid' or 'axis'")
    for key, values in sweep.items():
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"sweep value for {key!r} must be a list, got {values!r}")

    combos: list[dict] = [{}] if include_baseline else []
    if mode == "grid":
        keys = list(sweep)
        for values in itertools.product(*(sweep[k] for k in keys)):
            combos.append(dict(zip(keys, values)))
    else:
        for key, values in sweep.items():
            combos += [{key: v} for v in values]

    arms: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for overrides in combos:
        arm_id = _arm_id(overrides, prefix)
        if arm_id not in seen:
            seen.add(arm_id)
            arms.append((arm_id, overrides))
    return arms


# --------------------------------------------------------------------------- #
# Module-level resolution: what the stage scripts import.
# --------------------------------------------------------------------------- #

EXPERIMENT_ID = os.environ.get("GRAPHS_PROJECT_EXPERIMENT_ID") or "default"
CONFIG_PATH = Path(os.environ.get("GRAPHS_PROJECT_CONFIG") or (ROOT / "config" / "base.yaml"))
cfg = load_config(CONFIG_PATH)

PATHS = resolve(EXPERIMENT_ID, CONFIG_PATH, cfg)
CONFIG_PATH = PATHS.config_path
PROCESSED, GRAPHS = PATHS.processed, PATHS.graphs
RESULTS, RETRIEVAL, METRICS, LOGS = PATHS.results, PATHS.retrieval, PATHS.metrics, PATHS.logs

setup_logging()
if __name__ != "__main__":
    # Provenance at the top of every stage log: if a stale GRAPHS_PROJECT_* is
    # exported in your shell, these two lines are how you find out. The runner
    # skips them - it logs the dirs it resolved for each experiment instead.
    _log = logging.getLogger("experiment")
    _log.info("experiment=%s config=%s", EXPERIMENT_ID, CONFIG_PATH)
    _log.info("processed=%s graphs=%s results=%s", PROCESSED, GRAPHS, RESULTS)


# --------------------------------------------------------------------------- #
# The runner.
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run one experiment, or a sweep of them.",
        epilog="""examples:
  python scripts/experiment.py --sweep-json '{"mutual_knn_k": [4, 10, 20]}'
  python scripts/experiment.py --sweep-json '{"pcst.topk": [3, 6, 9]}' --include-baseline
  python scripts/experiment.py --id knn10 --set mutual_knn_k=10
  python scripts/experiment.py --id 00_prepare --stages 01""",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--sweep-json", default=None, metavar="JSON",
                   help='{"dotted.key": [v1, v2], ...} - every value becomes an experiment')
    p.add_argument("--sweep-mode", default="grid", choices=("grid", "axis"),
                   help="grid: every combination (default). axis: one key at a time")
    p.add_argument("--include-baseline", action="store_true",
                   help="also run an all-defaults arm called 'baseline'")
    p.add_argument("--experiments-json", default=None, metavar="JSON",
                   help='[{"id": ..., "overrides": {...}}, ...] for hand-picked combinations')
    p.add_argument("--id", default=None, help="single experiment; becomes results/<id>/")
    p.add_argument("--set", dest="sets", action="append", default=[], metavar="DOTTED=VALUE",
                   help="override for --id, repeatable, e.g. --set entity_graph.min_shared_entities=1")
    p.add_argument("--id-prefix", default="", help="prefix for auto-generated experiment ids")

    p.add_argument("--base-config", default=str(ROOT / "config" / "base.yaml"),
                   help="YAML to start from (default config/base.yaml)")
    p.add_argument("--stages", default=DEFAULT_STAGES,
                   help=f"comma separated: 01,02,03,04,05 or 'all' (default {DEFAULT_STAGES}); "
                        "a script stem such as 03_entity_pcst also works")
    p.add_argument("--force", action="store_true",
                   help="delete results/<id>/ first, so every stage reruns")
    p.add_argument("--force-graphs", action="store_true",
                   help="delete this config's graph dir first, so stage 2 rebuilds")
    p.add_argument("--prune-graphs", action="store_true",
                   help="delete the graph dir when every stage succeeded (long sweeps)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="abandon the sweep on the first failing experiment")
    p.add_argument("--allow-new-keys", action="store_true",
                   help="allow overrides to create keys absent from the base config")
    p.add_argument("--dry-run", action="store_true",
                   help="print the experiments, their ids and resolved dirs; write nothing")
    return p.parse_args(argv)


def collect_arms(args) -> list[tuple[str, dict]]:
    """Every requested experiment as (id, overrides), from whichever flag was used."""
    given = [bool(args.sweep_json), bool(args.experiments_json), bool(args.id)]
    if sum(given) != 1:
        raise SystemExit("pass exactly one of --sweep-json, --experiments-json or --id")

    if args.sweep_json:
        sweep = json.loads(args.sweep_json)
        if not isinstance(sweep, dict):
            raise SystemExit('--sweep-json must be an object: {"dotted.key": [v1, v2]}')
        return expand_sweep(sweep, args.sweep_mode, args.include_baseline, args.id_prefix)

    if args.experiments_json:
        loaded = json.loads(args.experiments_json)
        if not isinstance(loaded, list):
            raise SystemExit('--experiments-json must be a list of {"id":…, "overrides":{…}}')
        arms = []
        for entry in loaded:
            overrides = entry.get("overrides", {})
            arms.append((entry.get("id") or _arm_id(overrides, args.id_prefix), overrides))
        return arms

    overrides = {}
    for item in args.sets:
        if "=" not in item:
            raise SystemExit(f"--set needs DOTTED=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)  # "10"->10, "null"->None, "a"->"a"
    return [(args.id, overrides)]


def apply_overrides(cfg: dict, overrides: dict, allow_new_keys: bool) -> dict:
    merged = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        parts = str(dotted).split(".")
        node = merged
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                if not allow_new_keys:
                    # A typo like semantic.mutual_knn_k would otherwise cost a
                    # full GPU run and change nothing.
                    raise SystemExit(f"{dotted!r}: no section {part!r} in the base config "
                                     "(pass --allow-new-keys if that is intentional)")
                node[part] = {}
            node = node[part]
        if parts[-1] not in node and not allow_new_keys:
            raise SystemExit(f"{dotted!r}: no key {parts[-1]!r} in the base config "
                             "(pass --allow-new-keys if that is intentional)")
        node[parts[-1]] = value
    return merged


def expand_stages(spec: str) -> list[str]:
    if spec.strip() == "all":
        spec = ",".join(STAGES)
    scripts: list[str] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        # PowerShell turns an unquoted 01,02 into 1,2, so accept both spellings.
        token = token.zfill(2) if token.isdigit() else token
        if token in STAGES:
            scripts += STAGES[token]
        elif (SCRIPTS / f"{token}.py").exists():
            scripts.append(f"{token}.py")
        else:
            raise SystemExit(f"unknown stage {token!r}; try one of {list(STAGES)} "
                             "or a script stem such as 03_entity_pcst")
    return scripts


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"sha": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_stage(script: str, env: dict, logs: Path) -> tuple[int, float]:
    """Run one stage, streaming its output to this process AND to two log files."""
    started = time.perf_counter()
    header = f"\n===== {script}  {utc_now()} =====\n"
    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    with open(logs / f"{Path(script).stem}.log", "a", encoding="utf-8") as stage_log, \
         open(logs / "run.log", "a", encoding="utf-8") as run_log:
        sinks = (stage_log, run_log)
        for f in sinks:
            f.write(header)

        # bufsize=0 plus a raw read(): bytes arrive as they are produced, so tqdm
        # bars animate live in the notebook instead of landing in one lump.
        proc = subprocess.Popen([sys.executable, "-u", str(SCRIPTS / script)],
                                cwd=str(ROOT), env=env, bufsize=0,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pending = ""
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            # Normalise Windows line endings first: a child's "\r\n" must not be
            # mistaken for one of tqdm's bare "\r" redraws below, which would
            # strip every line in the log down to nothing.
            text = decoder.decode(chunk).replace("\r\n", "\n")
            sys.stdout.write(text)
            sys.stdout.flush()
            pending += text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                # tqdm redraws with \r; keep only its last frame per line, or the
                # log becomes one 200 KB line of half-drawn progress bars.
                for f in sinks:
                    f.write(line.rsplit("\r", 1)[-1] + "\n")
            if "\r" in pending:
                pending = pending.rsplit("\r", 1)[-1]
            for f in sinks:
                f.flush()

        pending += decoder.decode(b"", True)
        if pending.strip():
            for f in sinks:
                f.write(pending.rsplit("\r", 1)[-1] + "\n")
        rc = proc.wait()

    return rc, time.perf_counter() - started


def run_experiment(experiment_id: str, overrides: dict, merged_cfg: dict,
                   scripts: list[str], args, log) -> str:
    """Run every stage of one experiment. Returns its manifest status."""
    results = ROOT / "results" / experiment_id
    if args.force and results.exists():
        assert results.parent == ROOT / "results"  # never rmtree outside results/
        log.info("--force: removing %s", results)
        shutil.rmtree(results)

    paths = resolve(experiment_id, results / "config.yaml", merged_cfg)
    paths.logs.mkdir(parents=True, exist_ok=True)

    # The config the stages read IS the config saved with the results.
    paths.config_path.write_text(
        f"# Written by scripts/experiment.py for experiment {experiment_id!r}.\n"
        f"# base: {Path(args.base_config).as_posix()}\n"
        f"# overrides: {json.dumps(overrides, sort_keys=True)}\n"
        f"# created: {utc_now()}\n"
        + yaml.safe_dump(merged_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    env = os.environ.copy()
    env["GRAPHS_PROJECT_EXPERIMENT_ID"] = experiment_id
    env["GRAPHS_PROJECT_CONFIG"] = str(paths.config_path)
    env["PYTHONUNBUFFERED"] = "1"      # so the tee sees output as it happens
    env["PYTHONIOENCODING"] = "utf-8"  # so the pipe carries UTF-8 on Windows too

    if args.force_graphs and paths.graphs.exists():
        log.info("--force-graphs: removing %s", paths.graphs)
        shutil.rmtree(paths.graphs)

    log.info("config    %s", paths.config_path)
    log.info("processed %s (%s)", paths.processed,
             "HIT" if (paths.processed / "embeddings.pt").exists() else "MISS")
    log.info("graphs    %s (%s)", paths.graphs,
             "HIT" if (paths.graphs / "combined_graph.pt").exists() else "MISS")

    manifest = {
        "experiment_id": experiment_id,
        "created_at": utc_now(),
        "base_config": Path(args.base_config).as_posix(),
        "overrides": overrides,
        "stages_requested": scripts,
        "git": git_info(),
        "python": sys.version.split()[0],
        "paths": {k: str(v) for k, v in paths._asdict().items()},
        "status": "running",
        "stage_runs": [],
    }
    manifest_path = results / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save_manifest()
    failed = None
    for script in scripts:
        log.info("--- %s ---", script)
        rc, secs = run_stage(script, env, paths.logs)
        manifest["stage_runs"].append({
            "script": script, "returncode": rc, "duration_s": round(secs, 1),
            "finished_at": utc_now(), "log": f"logs/{Path(script).stem}.log",
            "status": "ok" if rc == 0 else "failed",
        })
        save_manifest()
        log.info("%s -> rc=%d in %.1fs", script, rc, secs)
        if rc != 0:
            # Evaluating a half-written stage 3 is worse than not evaluating.
            failed = script
            break

    manifest["status"] = "ok" if failed is None else f"failed:{failed}"
    save_manifest()

    if failed is None and args.prune_graphs and paths.graphs.exists():
        log.info("--prune-graphs: removing %s", paths.graphs)
        shutil.rmtree(paths.graphs)

    if failed:
        log.error("experiment %s FAILED at %s; see %s",
                  experiment_id, failed, paths.logs / f"{Path(failed).stem}.log")
    else:
        log.info("experiment %s complete -> %s", experiment_id, results)
    return manifest["status"]


def main(argv=None) -> int:
    args = parse_args(argv)
    log = logging.getLogger("experiment")

    base_cfg = load_config(args.base_config)
    arms = collect_arms(args)
    scripts = expand_stages(args.stages)

    # Validate every arm before running any of them: a typo should cost a second,
    # not an hour of GPU time in the middle of a sweep.
    plan = []
    for experiment_id, overrides in arms:
        try:
            validate_experiment_id(experiment_id)
        except ValueError as exc:
            raise SystemExit(str(exc))
        merged = apply_overrides(base_cfg, overrides, args.allow_new_keys)
        plan.append((experiment_id, overrides, merged))

    log.info("%d experiment(s), stages %s, free disk %.1f GB", len(plan),
             ",".join(Path(s).stem for s in scripts),
             shutil.disk_usage(str(ROOT)).free / 1e9)

    if args.dry_run:
        for experiment_id, overrides, merged in plan:
            paths = resolve(experiment_id, ROOT / "results" / experiment_id / "config.yaml", merged)
            print(f"\n=== {experiment_id} ===")
            print(f"  overrides {json.dumps(overrides, sort_keys=True)}")
            print(f"  results   {paths.results}")
            print(f"  processed {paths.processed}")
            print(f"  graphs    {paths.graphs}")
        print(f"\n{len(plan)} experiment(s), stages: "
              f"{', '.join(Path(s).stem for s in scripts)}  (dry run, nothing written)")
        return 0

    statuses: dict[str, str] = {}
    timings: dict[str, float] = {}
    for experiment_id, overrides, merged in plan:
        print(f"\n{'=' * 78}\n=== {experiment_id}   {json.dumps(overrides, sort_keys=True)}"
              f"\n{'=' * 78}", flush=True)
        t0 = time.perf_counter()
        statuses[experiment_id] = run_experiment(
            experiment_id, overrides, merged, scripts, args, log)
        timings[experiment_id] = time.perf_counter() - t0
        if statuses[experiment_id] != "ok" and args.stop_on_error:
            log.error("--stop-on-error: abandoning the remaining experiments")
            break

    print(f"\n{'=' * 78}\n=== summary\n{'=' * 78}")
    for experiment_id, status in statuses.items():
        print(f"{experiment_id:<32} {status:<24} {timings[experiment_id] / 60:>6.1f} min")
    skipped = [e for e, _, _ in plan if e not in statuses]
    if skipped:
        print("not run:", ", ".join(skipped))

    return 0 if all(s == "ok" for s in statuses.values()) and not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
