"""
Run a named workflow: executes a sequence of prep scripts in order.

Usage:
    python src/workflows.py <workflow_name>

Available workflows:
    all                prep0_0..1 → prep1_0..2 → prep3_01..11 → prep4_example
    new_parcellation   prep1_0 → prep1_1 → prep1_2 → prep3_01..11 → prep4_example
    new_template       prep0_0 → prep0_1
    new_example        prep4_example
    update_all_maprefs prep3_01..06 (re-generate all map files + parcellated tables)
    update_mrna        prep3_07_ref_mrna
    update_magicc      prep3_08_ref_magicc
    update_neurosynth  prep3_09_ref_neurosynth
    update_enigma      prep3_10_ref_enigma
    update_grf         prep3_11_ref_grf
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
        "prep3_01_ref_pet.py",
        "prep3_02_ref_cortexfeatures.py",
        "prep3_03_ref_bigbrain.py",
        "prep3_04_ref_tpm.py",
        "prep3_05_ref_rsn.py",
        "prep3_06_ref_rsn17.py",
        "prep3_07_ref_mrna.py",
        "prep3_08_ref_magicc.py",
        "prep3_09_ref_neurosynth.py",
        "prep3_10_ref_enigma.py",
        "prep3_11_ref_grf.py",
        "prep4_example.py",
    ],
    "new_parcellation": [
        "prep1_0_parc.py",
        "prep1_1_parc_distmat.py",
        "prep1_2_parc_spinmat.py",
        "prep3_01_ref_pet.py",
        "prep3_02_ref_cortexfeatures.py",
        "prep3_03_ref_bigbrain.py",
        "prep3_04_ref_tpm.py",
        "prep3_05_ref_rsn.py",
        "prep3_06_ref_rsn17.py",
        "prep3_07_ref_mrna.py",
        "prep3_08_ref_magicc.py",
        "prep3_09_ref_neurosynth.py",
        "prep3_10_ref_enigma.py",
        "prep3_11_ref_grf.py",
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
        "prep3_01_ref_pet.py",
        "prep3_02_ref_cortexfeatures.py",
        "prep3_03_ref_bigbrain.py",
        "prep3_04_ref_tpm.py",
        "prep3_05_ref_rsn.py",
        "prep3_06_ref_rsn17.py",
    ],
    "update_mrna": [
        "prep3_07_ref_mrna.py",
    ],
    "update_magicc": [
        "prep3_08_ref_magicc.py",
    ],
    "update_neurosynth": [
        "prep3_09_ref_neurosynth.py",
    ],
    "update_enigma": [
        "prep3_10_ref_enigma.py",
    ],
    "update_grf": [
        "prep3_11_ref_grf.py",
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
