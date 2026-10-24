# %% Init

import yaml
from pathlib import Path

import numpy as np
import nibabel as nib

from nispace.utils.utils_datasets import download

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# GIN commit hash for llotter/mni_freesurfer — update after cluster push
gin_commit = "ee79841bda3984689d726c2592bed7f68b1bbae6"

# All 15 pairs as (ref, flo).
# ref = target hub space, flo = source space.
# forward field resamples flo into ref space (pull convention, correct for nitransforms).
# backward field is a push map (flo grid → ref coords), used by mri_easywarp.
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
    # Non-templateflow spaces (templates hosted on GIN, see prep0_0_affines.py)
    ("MNI152NLin2009cAsym", "CAT12IXI555"),
    ("MNI152NLin6Asym",     "CAT12IXI555"),
    ("MNI152NLin2009cAsym", "CVSavg35inMNI152"),
    ("MNI152NLin6Asym",     "CVSavg35inMNI152"),
]

# %% Download EasyReg fields from GIN

GIN_FIELD_NAMES = {"backward": "bak_field", "forward": "fwd_field"}

# GIN fields are float32 (~2.4 GB total, some files > GitHub's 100 MB limit).
# Stored here as int16 with a fixed 0.01 mm step (scl_slope): ~1/3 of the size, max error 0.005 mm.
# Range covered: ±327 mm (MNI field coords are within ±151 mm). nibabel applies the scaling on load.
FIELD_STEP = 0.01


def quantize_field(src, dst, step=FIELD_STEP):
    img = nib.load(src)
    data = img.get_fdata(dtype=np.float32)
    q = np.round(data / step)
    assert np.abs(q).max() < np.iinfo(np.int16).max, f"field range exceeds int16 at step {step}: {src}"
    out = nib.Nifti1Image(q.astype(np.int16), img.affine, img.header)
    out.set_data_dtype(np.int16)
    out.header.set_slope_inter(step, 0)
    out.to_filename(dst)
    err = np.abs(nib.load(dst).get_fdata(dtype=np.float32) - data).max()
    assert err <= step / 2 + 1e-4, f"quantization error {err} mm too large: {dst}"

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
        quantize_field(download(url), dst)

print("All fields downloaded.")

# %% Write transform.yaml

transform_dir = wd / "transform"
transform_dir.mkdir(parents=True, exist_ok=True)

# Nested dict: ref (target) → flo (source) → {backward, forward}
# JSON access pattern: transform_json[mni_to][mni_from]["forward"]  ← pull resampling
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
