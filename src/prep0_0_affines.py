# %% Init

import json
import yaml
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image
from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from neuromaps.images import load_gifti

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from utils import tflow_get

nispace_toolbox_path = Path.home() / "projects" / "nispace"

FSLR_DENSITIES = ["4k", "8k", "32k", "164k"]
FSAVERAGE_DENSITIES = ["3k", "10k", "41k", "164k"]

# MNI sources: provide a URL string or a loaded SpatialImage for each native resolution.
# Resolutions listed here are downloaded/loaded directly; RESAMPLE_TARGETS are derived from 1mm.
# To add a new space, add a new key with at minimum a "1mm" entry.
MNI_SOURCES = {
    "MNI152NLin6Asym": {
        "1mm": tflow_get("MNI152NLin6Asym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
        "2mm": tflow_get("MNI152NLin6Asym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
    },
    "MNI152NLin2009cAsym": {
        "1mm": tflow_get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
        "2mm": tflow_get("MNI152NLin2009cAsym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
    },
    "MNIColin27": {
        "1mm": tflow_get("MNIColin27", desc="brain", suffix="mask", extension="nii.gz"),
    },
    "MNI305": {
        "1mm": tflow_get("MNI305", desc="brain", suffix="mask", extension="nii.gz"),
    },
    "MNI152Lin": {
        "1mm": tflow_get("MNI152Lin", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
        "2mm": tflow_get("MNI152Lin", resolution="02", desc=None, suffix="T1w", extension="nii.gz"),
    },
    "MNI152NLin2009cSym": {
        # templateflow only has res-1 (not res-01) for this space
        "1mm": tflow_get("MNI152NLin2009cSym", resolution="1", desc=None, suffix="T1w", extension="nii.gz"),
    },
    "MNI152NLin6Sym": {
        "1mm": tflow_get("MNI152NLin6Sym", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
    },
}
ALL_RESOLUTIONS = ["1mm", "2mm", "3mm", "4mm"]  # always produced for every space


def load_source(source):
    """Load a NIfTI image from a Path or return a SpatialImage directly."""
    if isinstance(source, nib.spatialimages.SpatialImage):
        return source
    return nib.load(source)


def geom_entry(img):
    """Extract affine (nested list) and shape (list of 3 ints) from a NIfTI image."""
    return {
        "affine": img.affine.tolist(),
        "shape":  list(img.shape[:3]),
    }


# %% MNI volumes

print("\n=== MNI volumes ===")
affines = {}

for space, res_sources in MNI_SOURCES.items():
    print(f"\n{space}:")
    affines[space] = {}
    loaded = {}  # res -> image

    # Load native resolutions from tflow cache
    for res, source in res_sources.items():
        print(f"  Loading {Path(source).name} ...")
        loaded[res] = load_source(source)

    # Best native source = smallest mm value (1mm preferred over 2mm etc.)
    best_native = min(loaded.keys(), key=lambda r: int(r.replace("mm", "")))
    src_img = loaded[best_native]

    # Produce all target resolutions
    for res in ALL_RESOLUTIONS:
        if res in loaded:
            img = loaded[res]
            tag = "native"
        else:
            target_mm = int(res.replace("mm", ""))
            src_shape = np.array(src_img.shape[:3])
            # Floor-safe shape: floor((N-1)/zoom)+1 ensures the last output
            # voxel maps back within the 1mm source FOV (resample_img without
            # target_shape can round up by 1, causing IndexError in nitransforms)
            tgt_shape = tuple(((src_shape - 1) // target_mm + 1).tolist())
            # Explicit origin: min world coord of source (handles ±voxel-size affines)
            vox_last = (src_img.affine @ np.array([*(src_shape - 1), 1]))[:3]
            origin = np.minimum(src_img.affine[:3, 3], vox_last)
            tgt_affine = np.diag([float(target_mm)] * 3 + [1.0])
            tgt_affine[:3, 3] = origin
            img = image.resample_img(
                src_img,
                target_affine=tgt_affine,
                target_shape=tgt_shape,
                interpolation="nearest",
                copy_header=True,
                force_resample=True,
            )
            tag = f"resampled from {best_native}"
        affines[space][res] = geom_entry(img)
        print(f"  {res} ({tag}): shape={img.shape[:3]}, origin={img.affine[:3, 3].round(2)}")

# %% Surface spaces (vertex counts only)

print("\n=== Surface spaces ===")

for space, fetch_fn, densities in [
    ("fsLR",      fetch_fslr,      FSLR_DENSITIES),
    ("fsaverage", fetch_fsaverage, FSAVERAGE_DENSITIES),
]:
    print(f"\n{space}:")
    affines[space] = {}
    for density in densities:
        atlas = fetch_fn(density=density)
        # Use sulc (.shape.gii) — agg_data() returns a 1D per-vertex array
        n_L = len(load_gifti(str(atlas["sulc"].L)).agg_data())
        n_R = len(load_gifti(str(atlas["sulc"].R)).agg_data())
        affines[space][density] = {"L": n_L, "R": n_R}
        print(f"  {density}: L={n_L}, R={n_R}")

# %% Write affines.yaml
# build_datalib.py runs automatically via pre-commit hook on nispace-data commit

out_yaml = wd / "template" / "affines.yaml"
out_yaml.write_text(yaml.dump(affines, default_flow_style=False, sort_keys=False, allow_unicode=True))
print(f"\nWritten → {out_yaml}")
