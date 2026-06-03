# %% Init

import yaml
from pathlib import Path

from nispace.utils.utils_datasets import download

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# GIN commit hash for llotter/mni_freesurfer — update after cluster push
gin_commit = "dead48f557087a694e93b633e863c641c21cbafb"

# All 11 pairs as (ref, flo).
# ref = target hub space, flo = source space.
# backward field resamples flo into ref space (primary direction).
PAIRS = [
    ("MNI152NLin2009cAsym", "MNI152NLin6Asym"),
    ("MNI152NLin2009cAsym", "MNI152Lin"),
    ("MNI152NLin6Asym",     "MNI152Lin"),
    ("MNI152NLin2009cAsym", "MNI305"),
    ("MNI152NLin6Asym",     "MNI305"),
    ("MNI152NLin2009cAsym", "MNIColin27"),
    ("MNI152NLin6Asym",     "MNIColin27"),
    ("MNI152NLin2009cAsym", "MNI152NLin2009cSym"),
    ("MNI152NLin6Asym",     "MNI152NLin6Sym"),
    ("MNI152NLin2009cAsym", "MNI152NLin6Sym"),
    ("MNI152NLin6Asym",     "MNI152NLin2009cSym"),
]

# %% Download EasyReg fields from GIN

GIN_FIELD_NAMES = {"backward": "bak_field", "forward": "fwd_field"}

for ref, flo in PAIRS:
    ref_dir = wd / "transform" / ref
    ref_dir.mkdir(parents=True, exist_ok=True)

    for direction, gin_name in GIN_FIELD_NAMES.items():
        fname = f"from-{flo}_to-{ref}_dir-{direction}_field.nii.gz"
        dst = ref_dir / fname
        if dst.exists():
            print(f"  exists: {dst.relative_to(wd)}")
            continue
        url = (
            f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}"
            f"/easyreg/from-{flo}_to-{ref}/{gin_name}.nii.gz"
        )
        print(f"  downloading: {dst.relative_to(wd)}")
        download(url, dst)

print("All fields downloaded.")

# %% Write transform.yaml

transform_dir = wd / "transform"
transform_dir.mkdir(parents=True, exist_ok=True)

# Nested dict: ref (target) → flo (source) → {backward, forward}
# JSON access pattern: transform_json[mni_to][mni_from]["backward"]
transform_cfg = {}
for ref, flo in PAIRS:
    if ref not in transform_cfg:
        transform_cfg[ref] = {}
    transform_cfg[ref][flo] = {
        "backward": {
            "host": "github-nispace",
            "remote": f"transform/{ref}/from-{flo}_to-{ref}_dir-backward_field.nii.gz",
        },
        "forward": {
            "host": "github-nispace",
            "remote": f"transform/{ref}/from-{flo}_to-{ref}_dir-forward_field.nii.gz",
        },
    }

yaml_path = transform_dir / "transform.yaml"
yaml_path.write_text(
    yaml.dump(transform_cfg, default_flow_style=False, sort_keys=True, allow_unicode=True)
)
print(f"Wrote {yaml_path.relative_to(wd)}")
