# %% Init

from pathlib import Path
import numpy as np
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
# Meta-analytic "reporting baseline" maps: voxel-wise density of reported activation foci across a
# coordinate-based meta-analysis database (C-SALE; Magielse et al., 2025). NiSpace uses them as sampling
# priors for NiMARE null maps (null foci drawn with probability proportional to the map value) -> the
# relative voxel values are the model and must NOT be rescaled or transformed nonlinearly.
# Original maps are downloaded to a temp file (not stored in repo); ref.yaml points to the same URLs
# as MNIOriginal (keep in sync).

# source: https://github.com/CNG-LAB/cerebellum_specific_ALE (MIT), commit 943417e (2025-06-06).
# Built by get_kernels_sum() in scripts/utils.py (called from prep_neurosynth()): all Neurosynth v7 foci,
# each convolved with an ALE kernel for n=20, summed. Native grid is FSL's MNI152 2mm (= MNI152NLin6Asym).
CSALE_BASE_URL = ("https://raw.githubusercontent.com/CNG-LAB/cerebellum_specific_ALE/"
                  "943417e389efb12dab5660a7c94b19354917ff44/output/data")

# map_id : source url
MAP_INFO = {
    "db-neurosynth_subset-all_pub-magielse2025": f"{CSALE_BASE_URL}/Neurosynth_dump_kernels_sum.nii.gz",
}


# %% Brain masks

mask_MNI6 = nib.load(wd / "template" / "MNI152NLin6Asym" / "map" / "brainmask" / "tpl-MNI152NLin6Asym_desc-brainmask_res-2mm.nii.gz")
mask_MNI2009 = nib.load(wd / "template" / "MNI152NLin2009cAsym" / "map" / "brainmask" / "tpl-MNI152NLin2009cAsym_desc-brainmask_res-2mm.nii.gz")

# mask without rescaling; stop if the map is not strictly positive within the mask
# (for the TPMs, prep3_04 min-max rescales in that case, which would change the relative weights here)
def mask_positive(img, mask, map_id, space):
    dat3d = img.get_fdata()
    dat3d_mask = mask.get_fdata().astype(bool)
    dat1d = dat3d[dat3d_mask]
    print(f"  {space}: within mask min = {dat1d.min()}, max = {dat1d.max()}")
    if dat1d.min() < 1e-6:
        raise ValueError(f"{map_id} ({space}): smallest value within mask is {dat1d.min()} (< 1e-6)! "
                         "Not rescaling as the relative values must be preserved; check the map.")
    return image.new_img_like(img, (dat3d * dat3d_mask).astype(np.float32), copy_header=True)


# %% Process each map

for map_id, url in MAP_INFO.items():
    print(f"\nProcessing: {map_id}")

    map_dir = nispace_source_data_path / "reference" / "mabaseline" / "map" / map_id
    map_dir.mkdir(parents=True, exist_ok=True)

    # download original map to a temp file (not stored in repo)
    orig_img = nib.load(download(url))

    # -----------------------------------------------------------------------
    # space: MNI152NLin6Asym
    # resample to 2mm MNI152NLin6Asym grid (identity for the C-SALE maps), cast to float32,
    # apply binary brain mask, NO rescaling
    # -----------------------------------------------------------------------
    map_MNI6 = image.resample_to_img(
        orig_img, mask_MNI6, interpolation="continuous",
        force_resample=True, copy_header=True,
    )
    map_MNI6 = image.math_img("img.astype(np.float32)", img=map_MNI6)
    map_MNI6_mask = mask_positive(map_MNI6, mask_MNI6, map_id, "MNI152NLin6Asym")
    map_MNI6_mask.to_filename(map_dir / f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz")

    # -----------------------------------------------------------------------
    # space: MNI152NLin2009cAsym
    # apply MNI transform to the (unmasked) 2mm MNI152NLin6Asym map, apply binary brain mask, NO rescaling
    # -----------------------------------------------------------------------
    map_MNI2009 = mni_to_mni(
        map_MNI6, mni_from="MNI152NLin6Asym", mni_to="MNI152NLin2009cAsym", order=3,
    )
    map_MNI2009_mask = mask_positive(map_MNI2009, mask_MNI2009, map_id, "MNI152NLin2009cAsym")
    map_MNI2009_mask.to_filename(map_dir / f"{map_id}_space-MNI152NLin2009cAsym_desc-proc.nii.gz")

    # -----------------------------------------------------------------------
    # space: fsLR 32k — from the (unmasked) 2mm MNI152NLin6Asym map
    # -----------------------------------------------------------------------
    map_fsLR = transforms.mni152_to_fslr(map_MNI6, fslr_density="32k", method="linear")
    map_fsLR[0].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-L.shape.gii.gz")
    map_fsLR[1].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-R.shape.gii.gz")

    # -----------------------------------------------------------------------
    # space: fsaverage 41k — from the (unmasked) 2mm MNI152NLin6Asym map
    # -----------------------------------------------------------------------
    map_fsavg = transforms.mni152_to_fsaverage(map_MNI6, fsavg_density="41k", method="linear")
    map_fsavg[0].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-L.shape.gii.gz")
    map_fsavg[1].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-R.shape.gii.gz")

    print(f"  Saved all spaces for {map_id}")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "mabaseline"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)


# %% Parcellate ------------------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parcellate_mapref

parcellate_mapref(wd, "mabaseline", spaces=["MNI152NLin6Asym"])

# %%
