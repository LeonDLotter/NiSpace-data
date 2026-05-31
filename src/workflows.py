"""
Run a named workflow: executes a sequence of prep scripts in order.

Usage:
    python src/workflows.py <workflow_name>

Available workflows:
    all                prep0_0..1 → prep1_0..2 → prep2_0..4 → prep3_0..5 → prep4_example
    new_parcellation   prep1_0 → prep1_1 → prep1_2 → prep3_0..5 → prep4_example
    new_template       prep0_0 → prep0_1
    new_example        prep4_example
    update_all_maprefs prep2_0..4 → prep3_0_tabref_parcmaps
    update_parcmaps    prep3_0_tabref_parcmaps
    update_mrna        prep3_1_tabref_mrna
    update_enigma      prep3_4_tabref_enigma
    update_neurosynth  prep3_3_tabref_neurosynth
    update_magicc      prep3_2_tabref_magicc
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent

# Scripts that are intentionally excluded from the "all" workflow
_EXCLUDED = {"generate_hashes.py", "utils.py", "workflows.py", "test_dataset_fetching.py"}

def _check_coverage() -> None:
    on_disk = {p.name for p in SRC.glob("prep*.py")}
    covered = set(WORKFLOWS["all"])
    missing = on_disk - covered - _EXCLUDED
    if missing:
        print(
            f"WARNING: the following prep scripts are not covered by the 'all' workflow:\n"
            + "\n".join(f"  {s}" for s in sorted(missing))
        )

WORKFLOWS = {
    "all": [
        "prep0_0_affines.py",
        "prep0_1_template.py",
        "prep1_0_parc.py",
        "prep1_1_parc_distmat.py",
        "prep1_2_parc_spinmat.py",
        "prep2_0_mapref_petsurfmaps.py",
        "prep2_1_mapref_cortexfeatures.py",
        "prep2_2_mapref_bigbrain.py",
        "prep2_3_mapref_tpm.py",
        "prep2_4_mapref_rsn.py",
        "prep2_5_mapref_rsn17.py",
        "prep3_0_tabref_parcmaps.py",
        "prep3_1_tabref_mrna.py",
        "prep3_2_tabref_magicc.py",
        "prep3_3_tabref_neurosynth.py",
        "prep3_4_tabref_enigma.py",
        "prep3_5_tabref_grf.py",
        "prep4_example.py",
    ],
    "new_parcellation": [
        "prep1_0_parc.py",
        "prep1_1_parc_distmat.py",
        "prep1_2_parc_spinmat.py",
        "prep3_0_tabref_parcmaps.py",
        "prep3_1_tabref_mrna.py",
        "prep3_2_tabref_magicc.py",
        "prep3_3_tabref_neurosynth.py",
        "prep3_4_tabref_enigma.py",
        "prep3_5_tabref_grf.py",
        "prep4_example.py",
    ],
    "new_template": [
        "prep0_0_affines.py",
        "prep0_1_template.py",
    ],
    "new_example": [
        "prep4_example.py",
    ],
    "update_all_maprefs": [
        "prep2_0_mapref_petsurfmaps.py",
        "prep2_1_mapref_cortexfeatures.py",
        "prep2_2_mapref_bigbrain.py",
        "prep2_3_mapref_tpm.py",
        "prep2_4_mapref_rsn.py",
        "prep2_5_mapref_rsn17.py",
        "prep3_0_tabref_parcmaps.py",
    ],
    "update_parcmaps": [
        "prep3_0_tabref_parcmaps.py",
    ],
    "update_mrna": [
        "prep3_1_tabref_mrna.py",
    ],
    "update_magicc": [
        "prep3_2_tabref_magicc.py",
    ],
    "update_neurosynth": [
        "prep3_3_tabref_neurosynth.py",
    ],
    "update_enigma": [
        "prep3_4_tabref_enigma.py",
    ],
    "update_grf": [
        "prep3_5_tabref_grf.py",
    ],
}


_check_coverage()


def run_workflow(name: str, exclude: list[str] = []) -> None:
    scripts = WORKFLOWS.get(name)
    if scripts is None:
        print(f"Unknown workflow '{name}'. Available: {', '.join(WORKFLOWS)}")
        sys.exit(1)

    scripts = [s for s in scripts if s not in exclude]
    if exclude:
        print(f"Excluding: {', '.join(exclude)}\n")

    print(f"=== workflow: {name} ({len(scripts)} script(s)) ===\n")
    for i, script in enumerate(scripts, 1):
        path = SRC / script
        print(f"[{i}/{len(scripts)}] {script}")
        result = subprocess.run([sys.executable, str(path)], check=False)
        if result.returncode != 0:
            print(f"\nFailed at {script} (exit {result.returncode}). Stopping.")
            sys.exit(result.returncode)
        print()

    print(f"=== workflow '{name}' complete ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_workflow(sys.argv[1], exclude=sys.argv[2:])
