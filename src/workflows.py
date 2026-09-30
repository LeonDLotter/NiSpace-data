"""
Run a named workflow: executes a sequence of prep scripts in order.

Usage:
    python src/workflows.py <workflow_name> [script_to_exclude ...] [--from <script>]

    --from <script>   start at this step (name with or without .py, or a unique prefix, e.g. prep3_12)

All output (incl. that of the scripts) is also written to logs/<YYYYmmdd>_<HHMMSS>_<workflow>.log
(never overwritten; logs/ is gitignored). The log header records the nispace-data and nispace commits.

Reference dataset scripts (prep3_XX_ref_<name>.py) are discovered automatically: a new script is
added to "all" and "new_parcellation" (and to "update_all_maprefs" if its ref.yaml has maps:), and
gets its own "update_<name>" workflow. Plotting (prep5_plots.py) runs after build_datalib.py, which
compiles the YAML sources that prep5_plots fetches parcellations from.

Available workflows:
    all                prep0_0..2 → prep1_0..2 → prep3_*_ref_* → prep4_example → build → plots (overwrite all)
    new_parcellation   prep1_0..2 → prep3_*_ref_* → prep4_example → build → plots (overwrite all)
    new_template       prep0_0 → prep0_1
    new_transforms     prep0_2
    new_example        prep4_example
    update_all_maprefs map-based prep3_*_ref_* (re-generate all map files + parcellated tables) → build → reference plots
    update_<name>      prep3_XX_ref_<name> → build → plot of that dataset (e.g. update_mrna, update_celltypes-pak2024)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

SRC = Path(__file__).parent
WD = SRC.parent
LOG_DIR = WD / "logs"

# Scripts that are intentionally not part of any workflow
_EXCLUDED = {"generate_hashes.py", "utils.py", "workflows.py", "test_dataset_fetching.py"}

BUILD = ("build_datalib.py",)
PLOTS_ALL = ("prep5_plots.py", "--overwrite")
PLOTS_REFS = ("prep5_plots.py", "--refs", "--overwrite")


def _ref_scripts():
    """{dataset script name: script file} for all prep3_XX_ref_<name>.py, in numeric order."""
    scripts = {}
    for p in sorted(SRC.glob("prep3_*_ref_*.py"), key=lambda p: int(p.name.split("_")[1])):
        scripts[re.match(r"prep3_\d+_ref_(.+)\.py$", p.name).group(1)] = p.name
    return scripts


def _datasets(script_name):
    """Reference datasets written by a ref script: the exact name, else all names with that prefix
    (e.g. enigma -> enigmaarea, enigmathick)."""
    names = [d.name for d in sorted((WD / "reference").iterdir()) if (d / "ref.yaml").exists()]
    return [script_name] if script_name in names else [n for n in names if n.startswith(script_name)]


def _is_mapref(script_name):
    return any(
        "maps" in yaml.safe_load((WD / "reference" / d / "ref.yaml").read_text())
        for d in _datasets(script_name)
    )


def _plots_for(script_name):
    return [("prep5_plots.py", "--refs", "--overwrite", "--name", d) for d in _datasets(script_name)]


REF_SCRIPTS = _ref_scripts()

WORKFLOWS = {
    "all": [
        "prep0_0_affines.py",
        "prep0_1_template.py",
        "prep0_2_transforms.py",
        "prep1_0_parc.py",
        "prep1_1_parc_distmat.py",
        "prep1_2_parc_spinmat.py",
        *REF_SCRIPTS.values(),
        "prep4_example.py",
        BUILD,
        PLOTS_ALL,
    ],
    "new_parcellation": [
        "prep1_0_parc.py",
        "prep1_1_parc_distmat.py",
        "prep1_2_parc_spinmat.py",
        *REF_SCRIPTS.values(),
        "prep4_example.py",
        BUILD,
        PLOTS_ALL,
    ],
    "new_template": [
        "prep0_0_affines.py",
        "prep0_1_template.py",
    ],
    "new_transforms": [
        "prep0_2_transforms.py",
    ],
    "new_example": [
        "prep4_example.py",
    ],
    "update_all_maprefs": [
        *[s for n, s in REF_SCRIPTS.items() if _is_mapref(n)],
        BUILD,
        PLOTS_REFS,
    ],
    **{
        f"update_{n}": [s, BUILD, *_plots_for(n)]
        for n, s in REF_SCRIPTS.items()
    },
}


def _script(entry):
    return entry if isinstance(entry, str) else entry[0]


def _check_coverage() -> None:
    on_disk = {p.name for p in SRC.glob("prep*.py")}
    covered = {_script(e) for e in WORKFLOWS["all"]}
    missing = on_disk - covered - _EXCLUDED
    if missing:
        print(
            f"WARNING: the following prep scripts are not covered by the 'all' workflow:\n"
            + "\n".join(f"  {s}" for s in sorted(missing))
        )


_check_coverage()


def _git_commit(repo) -> str:
    """Short commit hash of a repo, with '+dirty' if there are uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "unknown"
    return commit + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def _open_log(name: str):
    """Open a new log file; never overwrites an existing one."""
    LOG_DIR.mkdir(exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    path, n = LOG_DIR / f"{stem}.log", 1
    while path.exists():
        path, n = LOG_DIR / f"{stem}_{n}.log", n + 1
    return path, open(path, "w")


def _run_logged(cmd, log) -> int:
    """Run a command, streaming its output (stdout + stderr) to the terminal and the log.

    The terminal gets the raw stream (progress bars work); the log only gets the final state of
    each carriage-return-updated line, so tqdm bars don't bloat it.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    pending = b""
    while chunk := proc.stdout.read1(65536):
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            log.write(line.split(b"\r")[-1].decode(errors="replace") + "\n")
        log.flush()
    if pending:
        log.write(pending.split(b"\r")[-1].decode(errors="replace") + "\n")
    return proc.wait()


def _resolve_from(entries, start: str) -> int:
    """Index of the first entry whose script matches `start` (exact, without .py, or unique prefix)."""
    scripts = [_script(e) for e in entries]
    for cand in (start, f"{start}.py"):
        if cand in scripts:
            return scripts.index(cand)
    matches = sorted({s for s in scripts if s.startswith(start)})
    if len(matches) == 1:
        return scripts.index(matches[0])
    print(f"--from '{start}' matches {matches or 'no step'} in this workflow. Steps: {', '.join(scripts)}")
    sys.exit(1)


def run_workflow(name: str, exclude: list[str] = [], start: str = None) -> None:
    entries = WORKFLOWS.get(name)
    if entries is None:
        print(f"Unknown workflow '{name}'. Available: {', '.join(WORKFLOWS)}")
        sys.exit(1)

    if start:
        entries = entries[_resolve_from(entries, start):]
    entries = [e for e in entries if _script(e) not in exclude]

    log_path, log = _open_log(name)

    def out(msg=""):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    nispace_path = json.loads((WD / "config.json").read_text()).get("nispace_toolbox_path") \
        if (WD / "config.json").exists() else None
    out(f"=== workflow: {name} ({len(entries)} step(s)) ===")
    out(f"started:      {datetime.now():%Y-%m-%d %H:%M:%S}")
    out(f"command:      {' '.join(sys.argv)}")
    out(f"nispace-data: {_git_commit(WD)}")
    out(f"nispace:      {_git_commit(nispace_path) if nispace_path else 'unknown (no config.json)'}")
    out(f"python:       {sys.executable}")
    if start:
        out(f"from:         {_script(entries[0]) if entries else start}")
    if exclude:
        out(f"excluding:    {', '.join(exclude)}")
    out(f"log:          {log_path}")
    out()

    t_wf = time.time()
    for i, entry in enumerate(entries, 1):
        script, *args = (entry,) if isinstance(entry, str) else entry
        out(f"[{i}/{len(entries)}] {' '.join([script, *args])}  ({datetime.now():%H:%M:%S})")
        t = time.time()
        returncode = _run_logged([sys.executable, str(SRC / script), *args], log)
        if returncode != 0:
            out(f"\nFailed at {script} (exit {returncode}) after {_fmt(time.time() - t)}. Stopping.")
            log.close()
            sys.exit(returncode)
        out(f"[{i}/{len(entries)}] done in {_fmt(time.time() - t)}")
        out()

    out(f"=== workflow '{name}' complete ({_fmt(time.time() - t_wf)}) ===")
    log.close()


def _fmt(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m{rem % 60:02d}s" if h else f"{rem // 60}m{rem % 60:02d}s"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow", help="workflow name")
    parser.add_argument("exclude", nargs="*", help="scripts to skip")
    parser.add_argument("--from", dest="start", metavar="SCRIPT", help="start at this step")
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    a = parser.parse_intermixed_args()
    run_workflow(a.workflow, exclude=a.exclude, start=a.start)
