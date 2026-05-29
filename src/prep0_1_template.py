# %% Init

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image, datasets

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.utils.utils import apply_transform
from neuromaps.datasets import fetch_fslr, fetch_fsaverage

nispace_data_path = wd
nispace_toolbox_path = Path.home() / "projects" / "nispace"

# Load canonical geometry reference built by prep0_0_affines.py
with open(nispace_toolbox_path / "nispace" / "datalib" / "affines.json") as _f:
    affines_db = json.load(_f)

FSLR_DENSITIES = ["4k", "8k", "32k", "164k"]
FSAVERAGE_DENSITIES = ["3k", "10k", "41k", "164k"]

# Masks from Yeo et al. 2011 (nicely separated into tight and liberal)
# These are in MN152NLin6Asym. "thick" equals the Schaefer parcellation masks
datasets.fetch_atlas_yeo_2011()
mask_liberal_MNI6 = image.index_img(image.math_img(
    "img > 0", 
    img=datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thick").maps
), 0)
mask_tight_MNI6 = image.index_img(image.math_img(
    "img > 0", 
    img=datasets.fetch_atlas_yeo_2011(n_networks=7, thickness="thin").maps
), 0)


# Sources for MNI templates.
# Values are either a URL string (load/download directly) or a (space, res, desc) tuple
# referencing an already-processed entry (resample if same space, apply_transform if different).
MNI_TEMPLATE_SOURCES = {
    "MNI152NLin6Asym": {
        "1mm": {
            "T1w":          "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz",
            "brain":        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz",
            "mask":         "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz",
            "mask_gm":      mask_liberal_MNI6,
            "mask_gm_tight":mask_tight_MNI6,
        },
        "2mm": {
            "T1w":          "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz",
            "brain":        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz",
            "mask":         "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz",
            "mask_gm":      ("MNI152NLin6Asym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin6Asym", "1mm", "mask_gm_tight"),
        },
        "3mm": {
            "T1w":          ("MNI152NLin6Asym", "1mm", "T1w"),
            "brain":        ("MNI152NLin6Asym", "1mm", "brain"),
            "mask":         ("MNI152NLin6Asym", "1mm", "mask"),
            "mask_gm":      ("MNI152NLin6Asym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin6Asym", "1mm", "mask_gm_tight"),
        },
    },
    "MNI152NLin2009cAsym": {
        "1mm": {
            "T1w":          "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
            "brain":        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz",
            "mask":         "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz",
            "mask_gm":      ("MNI152NLin6Asym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin6Asym", "1mm", "mask_gm_tight"),
        },
        "2mm": {
            "T1w":          "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz",
            "brain":        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz",
            "mask":         "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
            "mask_gm":      ("MNI152NLin2009cAsym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin2009cAsym", "1mm", "mask_gm_tight"),
        },
        "3mm": {
            "T1w":          ("MNI152NLin2009cAsym", "1mm", "T1w"),
            "brain":        ("MNI152NLin2009cAsym", "1mm", "brain"),
            "mask":         ("MNI152NLin2009cAsym", "1mm", "mask"),
            "mask_gm":      ("MNI152NLin2009cAsym", "1mm", "mask_gm"),
            "mask_gm_tight":("MNI152NLin2009cAsym", "1mm", "mask_gm_tight"),
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

            if isinstance(source, (str, nib.spatialimages.SpatialImage)):
                if isinstance(source, str):
                    print(f"  Downloading {dst.name} ...")
                    urllib.request.urlretrieve(source, dst)
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

# %% Write complete template.json to nispace toolbox

out = nispace_toolbox_path / "nispace" / "datalib" / "template.json"
with open(out, "w") as f:
    json.dump(template_json, f, indent=4)
print(f"\nWritten → {out}")
