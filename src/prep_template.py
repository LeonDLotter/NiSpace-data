# %% Init

import sys
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

from neuromaps.datasets import fetch_fslr, fetch_fsaverage

nispace_data_path = wd
nispace_toolbox_path = Path.home() / "projects" / "nispace"

FSLR_DENSITIES = ["4k", "8k", "32k", "164k"]
FSAVERAGE_DENSITIES = ["3k", "10k", "41k", "164k"]

# TemplateFlow source URLs for MNI templates (1mm and 2mm only; 3mm is resampled below)
MNI_TEMPLATEFLOW_URLS = {
    "MNI152NLin2009cAsym": {
        "1mm": {
            "T1w":    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz",
            "brain":  "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz",
            "mask":   "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_desc-brain_mask.nii.gz",
            "gmprob": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_label-GM_probseg.nii.gz",
        },
        "2mm": {
            "T1w":    "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz",
            "brain":  "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz",
            "mask":   "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
            "gmprob": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_label-GM_probseg.nii.gz",
        },
    },
    "MNI152NLin6Asym": {
        "1mm": {
            "T1w":   "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_T1w.nii.gz",
            "brain": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz",
            "mask":  "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-01_desc-brain_mask.nii.gz",
        },
        "2mm": {
            "T1w":   "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz",
            "brain": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz",
            "mask":  "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz",
        },
    },
}


def dst_path_mni(space, desc, res):
    fname = f"tpl-{space}_desc-{desc}_res-{res}.nii.gz"
    return nispace_data_path / "template" / space / "map" / desc / fname


def download_and_register_mni(space, descs_urls, res):
    """Download MNI templates from TemplateFlow and return template.json sub-dict."""
    entries = {}
    for desc, url in descs_urls.items():
        dst = dst_path_mni(space, desc, res)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"  Downloading {dst.name} ...")
            urllib.request.urlretrieve(url, dst)
        else:
            print(f"  Exists: {dst.name}")
        entries[desc] = {
            "host": "github-nispace",
            "remote": f"template/{space}/map/{desc}/{dst.name}",
        }
    return entries


def resample_and_register_mni(space, descs, src_res="1mm", target_res="3mm"):
    """Resample MNI templates from src_res to target_res and return template.json sub-dict."""
    entries = {}
    target_mm = int(target_res.replace("mm", ""))
    for desc in descs:
        src = dst_path_mni(space, desc, src_res)
        dst = dst_path_mni(space, desc, target_res)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print(f"  Resampling {src.name} -> {dst.name} ...")
            resampled = image.resample_img(
                image.load_img(src),
                target_affine=np.diag([target_mm] * 3),
                interpolation="nearest" if desc == "mask" else "linear",
            )
            resampled.to_filename(dst)
        else:
            print(f"  Exists: {dst.name}")
        entries[desc] = {
            "host": "github-nispace",
            "remote": f"template/{space}/map/{desc}/{dst.name}",
        }
    return entries


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

# %% MNI152NLin2009cAsym — all resolutions downloaded/generated and stored in nispace-data

print("\n=== MNI152NLin2009cAsym ===")
template_json["MNI152NLin2009cAsym"] = {}
for res, descs_urls in MNI_TEMPLATEFLOW_URLS["MNI152NLin2009cAsym"].items():
    print(f"\nMNI152NLin2009cAsym {res}:")
    template_json["MNI152NLin2009cAsym"][res] = download_and_register_mni(
        "MNI152NLin2009cAsym", descs_urls, res
    )

print("\nMNI152NLin2009cAsym 3mm (resampled from 1mm):")
template_json["MNI152NLin2009cAsym"]["3mm"] = resample_and_register_mni(
    "MNI152NLin2009cAsym",
    list(MNI_TEMPLATEFLOW_URLS["MNI152NLin2009cAsym"]["1mm"].keys()),
    src_res="1mm", target_res="3mm",
)

# %% MNI152NLin6Asym — all resolutions downloaded/generated and stored in nispace-data

print("\n=== MNI152NLin6Asym ===")
template_json["MNI152NLin6Asym"] = {}
for res, descs_urls in MNI_TEMPLATEFLOW_URLS["MNI152NLin6Asym"].items():
    print(f"\nMNI152NLin6Asym {res}:")
    template_json["MNI152NLin6Asym"][res] = download_and_register_mni(
        "MNI152NLin6Asym", descs_urls, res
    )

print("\nMNI152NLin6Asym 3mm (resampled from 1mm):")
template_json["MNI152NLin6Asym"]["3mm"] = resample_and_register_mni(
    "MNI152NLin6Asym",
    list(MNI_TEMPLATEFLOW_URLS["MNI152NLin6Asym"]["1mm"].keys()),
    src_res="1mm", target_res="3mm",
)

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
