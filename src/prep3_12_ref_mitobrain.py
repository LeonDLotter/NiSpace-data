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


# %% Map info
# source: Mosharov et al., 2025, Nature — https://doi.org/10.1038/s41586-025-08740-6
# maps: https://neurovault.org/collections/16418/

NV_BASE_URL = "https://neurovault.org/media/images/16418"

# map_id : nv_filename
MAP_INFO = {
    "feature-complexi_pub-mosharov2025":    "CI",
    "feature-complexii_pub-mosharov2025":   "CII",
    "feature-complexiv_pub-mosharov2025":   "CIV",
    "feature-mitodensity_pub-mosharov2025": "MitoD",
    "feature-tissueresp_pub-mosharov2025":  "TRC",
    "feature-mitoresp_pub-mosharov2025":    "MRC",
}


# %% Reference images and GM masks

mask_gm_MNI6 = wd / "template" / "MNI152NLin6Asym" / "map" / "gmmask" / "tpl-MNI152NLin6Asym_desc-gmmask_res-2mm.nii.gz"
mask_gm_MNI2009 = wd / "template" / "MNI152NLin2009cAsym" / "map" / "gmmask" / "tpl-MNI152NLin2009cAsym_desc-gmmask_res-2mm.nii.gz"


# %% Process each map

for map_id, nv_name in MAP_INFO.items():
    print(f"\nProcessing: {map_id}")

    map_dir = nispace_source_data_path / "reference" / "mitobrain" / "map" / map_id
    map_dir.mkdir(parents=True, exist_ok=True)

    # download original map from NeuVault to a temp file (not stored in repo)
    tmp = download(f"{NV_BASE_URL}/{nv_name}.nii.gz")
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
    # space: fsLR 32k — from original map (no prior processing, valid bc original map is MNI152NLin6Asym)
    # -----------------------------------------------------------------------
    map_fsLR = transforms.mni152_to_fslr(orig_img, fslr_density="32k", method="linear")
    map_fsLR[0].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-L.surf.gii.gz")
    map_fsLR[1].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-R.surf.gii.gz")

    # -----------------------------------------------------------------------
    # space: fsaverage 41k — from original map (no prior processing, valid bc original map is MNI152NLin6Asym)
    # -----------------------------------------------------------------------
    map_fsavg = transforms.mni152_to_fsaverage(orig_img, fsavg_density="41k", method="linear")
    map_fsavg[0].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-L.surf.gii.gz")
    map_fsavg[1].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-R.surf.gii.gz")

    print(f"  Saved all spaces for {map_id}")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "mitobrain"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)


# %% Parcellate ------------------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parcellate_mapref

parcellate_mapref(wd, "mitobrain", spaces=["MNI152NLin6Asym"])

# %%
