# %% Init

import json
import shutil
import yaml
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image, datasets
import templateflow.api as tflow

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.utils.utils import apply_transform
from nispace.utils.utils_datasets import download
from neuromaps.datasets import fetch_fslr, fetch_fsaverage
from utils import tflow_get

nispace_data_path = wd
nispace_toolbox_path = Path.home() / "projects" / "nispace"

# Load canonical geometry reference built by prep0_0_affines.py
with open(nispace_data_path / "template" / "affines.yaml") as _f:
    affines_db = yaml.safe_load(_f)

FSLR_DENSITIES = ["4k", "8k", "32k", "164k"]
FSAVERAGE_DENSITIES = ["3k", "10k", "41k", "164k"]

# Liberal gray matter masks sourced from Schaefer parcellation (templateflow)
mask_liberal_MNI6_1mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="01", suffix="dseg", extension="nii.gz")
)
mask_liberal_MNI6_2mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="02", suffix="dseg", extension="nii.gz")
)
mask_liberal_MNI9_1mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin2009cAsym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="01", suffix="dseg", extension="nii.gz")
)
mask_liberal_MNI9_2mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin2009cAsym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="02", suffix="dseg", extension="nii.gz")
)

# Conservative gray matter masks sourced from freesurfer segmentation of MNI templates (gin)
gin_commit = "5839f92fdf207714fae18088da7a114b21a015c8"
mask_tight_MNI6_1mm = image.math_img(
    "img > 0", 
    img=download(f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/parcellations/MNI152NLin6Asym/seg-aparc_space-MNI152NLin6Asym.nii.gz")
)
mask_tight_MNI9_1mm = image.math_img(
    "img > 0", 
    img=download(f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/parcellations/MNI152NLin2009cAsym/seg-aparc_space-MNI152NLin2009cAsym.nii.gz")
)

# Sources for MNI templates.
# Values are either a URL string (load/download directly) or a (space, res, desc) tuple
# referencing an already-processed entry (resample if same space, apply_transform if different).
MNI_TEMPLATE_SOURCES = {
    "MNI152NLin2009cAsym": {
        "1mm": {
            "T1w":          tflow_get("MNI152NLin2009cAsym", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":        tflow_get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="T1w", extension="nii.gz"),
            "mask":         tflow_get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
            "gmprob":       tflow_get("MNI152NLin2009cAsym", resolution="01", label="GM", suffix="probseg", extension="nii.gz"),
            "mask_gm":      mask_liberal_MNI9_1mm,
            "mask_gm_tight":mask_tight_MNI9_1mm,
        },
        "2mm": {
            "T1w":          tflow_get("MNI152NLin2009cAsym", resolution="02", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":        tflow_get("MNI152NLin2009cAsym", resolution="02", desc="brain", suffix="T1w", extension="nii.gz"),
            "mask":         tflow_get("MNI152NLin2009cAsym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
            "gmprob":       tflow_get("MNI152NLin2009cAsym", resolution="02", label="GM", suffix="probseg", extension="nii.gz"),
            "mask_gm":      mask_liberal_MNI9_2mm,
            "mask_gm_tight":("MNI152NLin2009cAsym", "1mm", "mask_gm_tight"),
        },
        "3mm": {
            "T1w":          ("MNI152NLin2009cAsym", "1mm", "T1w"),
            "brain":        ("MNI152NLin2009cAsym", "1mm", "brain"),
            "mask":         ("MNI152NLin2009cAsym", "1mm", "mask"),
            "gmprob":       ("MNI152NLin2009cAsym", "1mm", "gmprob"),
            "mask_gm":      ("MNI152NLin2009cAsym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin2009cAsym", "1mm", "mask_gm_tight"),
        },
    },
    "MNI152NLin6Asym": {
        "1mm": {
            "T1w":          tflow_get("MNI152NLin6Asym", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":        tflow_get("MNI152NLin6Asym", resolution="01", desc="brain", suffix="T1w", extension="nii.gz"),
            "mask":         tflow_get("MNI152NLin6Asym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
            "gmprob":       ("MNI152NLin2009cAsym", "1mm", "gmprob"), # resample from MNI9 since MNI6 doesn't have TPMs in templateflow
            "mask_gm":      mask_liberal_MNI6_1mm,
            "mask_gm_tight":mask_tight_MNI6_1mm,
        },
        "2mm": {
            "T1w":          tflow_get("MNI152NLin6Asym", resolution="02", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":        tflow_get("MNI152NLin6Asym", resolution="02", desc="brain", suffix="T1w", extension="nii.gz"),
            "mask":         tflow_get("MNI152NLin6Asym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
            "gmprob":       ("MNI152NLin6Asym", "1mm", "gmprob"),
            "mask_gm":      mask_liberal_MNI6_2mm,
            "mask_gm_tight":("MNI152NLin6Asym", "1mm", "mask_gm_tight"),
        },
        "3mm": {
            "T1w":          ("MNI152NLin6Asym", "1mm", "T1w"),
            "brain":        ("MNI152NLin6Asym", "1mm", "brain"),
            "mask":         ("MNI152NLin6Asym", "1mm", "mask"),
            "gmprob":       ("MNI152NLin6Asym", "1mm", "gmprob"),
            "mask_gm":      ("MNI152NLin6Asym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin6Asym", "1mm", "mask_gm_tight"),
        },
    },
}


def dst_path_mni(space, desc, res):
    fname = f"tpl-{space}_desc-{desc}_res-{res}.nii.gz"
    return nispace_data_path / "template" / space / "map" / desc / fname


def get_gii_ext(src_path):
    """Infer .surf.gii / .shape.gii / .label.gii from source filename."""
    name = str(src_path)
    for prefix in [".surf.", ".shape.", ".label.", ".func."]:
        if prefix in name:
            return prefix[1:] + "gii"
    return "gii"


def copy_and_register(space, density, atlas):
    """Copy neuromaps surface files to nispace-data and return template.json sub-dict."""
    entries = {}
    for desc, files in atlas.items():
        entries[desc] = {}
        for hemi, src in [("L", Path(str(files.L))), ("R", Path(str(files.R)))]:
            ext = get_gii_ext(src)
            dst_name = f"tpl-{space}_desc-{desc}_res-{density}_hemi-{hemi}.{ext}"
            dst = nispace_data_path / "template" / space / "map" / desc / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  {dst_name}")
            shutil.copy(src, dst)
            entries[desc][hemi] = {
                "host": "github-nispace",
                "remote": f"template/{space}/map/{desc}/{dst_name}",
            }
    return entries


# %% Build complete template.json

template_json = {}

# %% MNI templates — unified loop: download URLs, resample (same space) or transform (cross-space)

print("\n=== MNI templates ===")
processed = {}   # (space, res, desc) -> Path

for space, space_data in MNI_TEMPLATE_SOURCES.items():
    template_json[space] = {}
    for res, descs in space_data.items():
        print(f"\n{space} {res}:")
        template_json[space][res] = {}

        # Reference geometry from affines.json — no T1w-first ordering needed
        ref_aff   = np.array(affines_db[space][res]["affine"])
        ref_shape = tuple(affines_db[space][res]["shape"])
        interp    = lambda desc: "nearest" if "mask" in desc else "continuous"

        for desc, source in descs.items():
            dst = dst_path_mni(space, desc, res)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(source, (str, Path, nib.spatialimages.SpatialImage)):
                if isinstance(source, Path):
                    print(f"  Copying {dst.name} ...")
                    shutil.copy(source, dst)
                    img = image.load_img(dst)
                elif isinstance(source, str):
                    print(f"  Downloading {dst.name} ...")
                    download(source, dst)
                    img = image.load_img(dst)
                else:
                    print(f"  Saving {dst.name} ...")
                    source.to_filename(dst)
                    img = source

                if img.shape[:3] != ref_shape or not np.allclose(img.affine, ref_aff):
                    print(
                        f"  WARNING: geometry mismatch for '{desc}' — "
                        f"shape {img.shape[:3]} (expected {ref_shape}), "
                        f"voxsize {np.abs(np.diag(img.affine)[:3]).round(4)} "
                        f"(expected {np.abs(np.diag(ref_aff)[:3]).round(4)}) — resampling ..."
                    )
                    image.resample_img(
                        img,
                        target_affine=ref_aff,
                        target_shape=ref_shape,
                        interpolation=interp(desc),
                        copy_header=True,
                        force_resample=True,
                    ).to_filename(dst)

            elif isinstance(source, tuple):
                src_space, src_res, src_desc = source
                src_path = processed[(src_space, src_res, src_desc)]

                if src_space == space:
                    # Same space — resample to canonical reference geometry from affines.json
                    print(f"  Resampling {src_path.name} -> {dst.name} ...")
                    image.resample_img(
                        image.load_img(src_path),
                        target_affine=ref_aff,
                        target_shape=ref_shape,
                        interpolation=interp(desc),
                        copy_header=True,
                        force_resample=True,
                    ).to_filename(dst)

                else:
                    # Different space — apply MNI-to-MNI transform, then check geometry
                    print(f"  Transforming {src_path.name} ({src_space} -> {space}) ...")
                    apply_transform(
                        img=src_path,
                        mni_from=src_space,
                        mni_to=space,
                        res=res,
                        order=0 if "mask" in desc else 3,
                    ).to_filename(dst)
                    img = image.load_img(dst)
                    if img.shape[:3] != ref_shape or not np.allclose(img.affine, ref_aff):
                        print(
                            f"  WARNING: geometry mismatch after transform for '{desc}' — "
                            f"shape {img.shape[:3]} (expected {ref_shape}), "
                            f"voxsize {np.abs(np.diag(img.affine)[:3]).round(4)} "
                            f"(expected {np.abs(np.diag(ref_aff)[:3]).round(4)}) — resampling ..."
                        )
                        image.resample_img(
                            img,
                            target_affine=ref_aff,
                            target_shape=ref_shape,
                            interpolation=interp(desc),
                            copy_header=True,
                            force_resample=True,
                        ).to_filename(dst)

            processed[(space, res, desc)] = dst
            template_json[space][res][desc] = {
                "host": "github-nispace",
                "remote": f"template/{space}/map/{desc}/{dst.name}",
            }

# %% fsLR — all densities from neuromaps, stored in nispace-data

print("\n=== fsLR ===")
template_json["fsLR"] = {}
for density in FSLR_DENSITIES:
    print(f"\nfsLR {density}:")
    atlas = fetch_fslr(density=density)
    template_json["fsLR"][density] = copy_and_register("fsLR", density, atlas)

# %% fsaverage — all densities from neuromaps, stored in nispace-data

print("\n=== fsaverage ===")
template_json["fsaverage"] = {}
for density in FSAVERAGE_DENSITIES:
    print(f"\nfsaverage {density}:")
    atlas = fetch_fsaverage(density=density)
    template_json["fsaverage"][density] = copy_and_register("fsaverage", density, atlas)

# %% Write template.yaml per space
# build_datalib.py runs automatically via pre-commit hook on nispace-data commit

for space, resolutions in template_json.items():
    yaml_path = nispace_data_path / "template" / space / "template.yaml"
    cfg = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {"name": space}
    cfg["resolutions"] = resolutions
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
    print(f"Written → {yaml_path}")
