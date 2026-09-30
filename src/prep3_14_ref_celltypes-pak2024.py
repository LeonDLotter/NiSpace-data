# %% Init

from pathlib import Path
import pandas as pd
import nibabel as nib
from nilearn import image
from neuromaps import transforms

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.transforms import mni_to_mni
from nispace.utils.utils_datasets import download

nispace_source_data_path = wd
dataset = "celltypes-pak2024"


# %% Map info
# source: Pak et al., 2024, eLife — https://doi.org/10.7554/eLife.89368
# maps: https://github.com/neuropm-lab/cellmaps (CC BY 4.0, added on our request 2026-09-28;
# permission: _archive/usage_permission/Pak2024_permission.pdf)
# The maps are on the SPM 1.5mm MNI grid (121x145x121); their GM mask is ~identical to AAL2
# (Dice 0.96) -> treated as MNI152NLin6Asym, same as AAL in prep1_0_parc.py.

GH_BASE_URL = "https://raw.githubusercontent.com/neuropm-lab/cellmaps/9f97684e8cf2b8196e5b13c11d1001236aec2665"

# map_id : github filename
MAP_INFO = {
    "cell-astrocytes_pub-pak2024":       "ast",
    "cell-endothelial_pub-pak2024":      "end",
    "cell-microglia_pub-pak2024":        "mic",
    "cell-neurons_pub-pak2024":          "neu",
    "cell-oligodendrocytes_pub-pak2024": "oli",
    "cell-opc_pub-pak2024":              "opc",
}


# %% Reference images and GM masks

mask_gm_MNI6 = wd / "template" / "MNI152NLin6Asym" / "map" / "gmmask" / "tpl-MNI152NLin6Asym_desc-gmmask_res-2mm.nii.gz"
mask_gm_MNI2009 = wd / "template" / "MNI152NLin2009cAsym" / "map" / "gmmask" / "tpl-MNI152NLin2009cAsym_desc-gmmask_res-2mm.nii.gz"


# %% Process each map

for map_id, gh_name in MAP_INFO.items():
    print(f"\nProcessing: {map_id}")

    map_dir = nispace_source_data_path / "reference" / dataset / "map" / map_id
    map_dir.mkdir(parents=True, exist_ok=True)

    # download original map from GitHub to a temp file (not stored in repo)
    tmp = download(f"{GH_BASE_URL}/{gh_name}.nii")
    orig_img = nib.load(tmp)

    # -----------------------------------------------------------------------
    # space: MNI152NLin6Asym
    # resample to 2mm MNI152NLin6Asym affine (no deformable transform),
    # apply binary GM mask
    # -----------------------------------------------------------------------
    map_MNI6_2mm = image.resample_to_img(
        orig_img, mask_gm_MNI6, interpolation="continuous",
        force_resample=True, copy_header=True,
    )
    map_MNI6_2mm_mask = image.math_img(
        "(img * mask).astype(np.float32)",
        img=map_MNI6_2mm, mask=mask_gm_MNI6,
    )
    map_MNI6_2mm_mask.to_filename(map_dir / f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz")

    # -----------------------------------------------------------------------
    # space: MNI152NLin2009cAsym
    # apply MNI transform to original image, directly resample to 2mm
    # apply binary GM mask
    # -----------------------------------------------------------------------
    map_MNI2009_2mm = mni_to_mni(
        orig_img, mni_from="MNI152NLin6Asym", mni_to="MNI152NLin2009cAsym", order=3, res="2mm",
    )
    map_MNI2009_2mm_mask = image.math_img(
        "(img * mask).astype(np.float32)",
        img=map_MNI2009_2mm, mask=mask_gm_MNI2009,
    )
    map_MNI2009_2mm_mask.to_filename(map_dir / f"{map_id}_space-MNI152NLin2009cAsym_desc-proc.nii.gz")

    # -----------------------------------------------------------------------
    # space: fsLR 32k — from original map (valid bc original map is treated as MNI152NLin6Asym)
    # -----------------------------------------------------------------------
    map_fsLR = transforms.mni152_to_fslr(orig_img, fslr_density="32k", method="linear")
    map_fsLR[0].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-L.surf.gii.gz")
    map_fsLR[1].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-R.surf.gii.gz")

    # -----------------------------------------------------------------------
    # space: fsaverage 41k — from original map (valid bc original map is treated as MNI152NLin6Asym)
    # -----------------------------------------------------------------------
    map_fsavg = transforms.mni152_to_fsaverage(orig_img, fsavg_density="41k", method="linear")
    map_fsavg[0].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-L.surf.gii.gz")
    map_fsavg[1].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-R.surf.gii.gz")

    print(f"  Saved all spaces for {map_id}")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / dataset
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)


# %% Parcellate ------------------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parcellate_mapref

parcellate_mapref(wd, dataset, spaces=["MNI152NLin6Asym"])

# %%
