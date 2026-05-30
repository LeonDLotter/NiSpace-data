# %% Init

from pathlib import Path
import pandas as pd
from neuromaps import transforms
import nibabel as nib
from nilearn import image

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.datasets import fetch_reference, reference_lib
from nispace.io import load_img
from nispace.utils.utils import apply_transform
from nispace.utils.utils_datasets import download

# nispace data path 
nispace_source_data_path = wd


# %% Prep the GM/WM/CSF TPM atlas
# source: SPM TPM, https://github.com/spm/spm/tree/main/tpm

# link to the data
url = "https://raw.githubusercontent.com/spm/spm/5d76594776bbe62e3a013c0560fb96bbe5bf4ede/tpm/TPM.nii"

# download and load 4D volume
tpm_vol = image.load_img(download(url))

# save tissue types we want
for idx, tissue in [(0, "gm"), (1, "wm"), (2, "csf")]:
    tpm_vol_tissue = image.index_img(tpm_vol, idx)
    fp = (nispace_source_data_path / "reference" / "tpm" / "map" / f"tissue-{tissue}_pub-spm" / 
            f"tissue-{tissue}_pub-spm_space-MNI152.nii.gz")
    fp.parent.mkdir(parents=True, exist_ok=True)
    tpm_vol_tissue.to_filename(fp)  
    
        
# %% Load all atlases via reference_lib
template_MNI152NLin6Asym = wd / "template" / "MNI152NLin6Asym" / "map" / "mask" / "tpl-MNI152NLin6Asym_desc-mask_res-2mm.nii.gz"

for m in reference_lib["tpm"]["map"]:
    print("Processing map:", m)

    # get original map — SPM-derived maps exist locally; arteries/veins are fetched from remote
    if any(t in m for t in ["gm", "wm", "csf"]):
        fp = [nispace_source_data_path / "reference" / "tpm" / "map" / m / f"{m}_space-MNI152.nii.gz"]
    else:
        fp = fetch_reference("tpm", maps=m, space="MNI152", verbose=False, check_file_hash=False)
    print(fp)
    
    # ----------------------------------------------------------------------------------------------
    # space: MNI152NLin6Asym
    # ----------------------------------------------------------------------------------------------
    # maps are already in MNI152NLin6Asym-like spaces, so we start with that
    map_MNI152NLin6Asym = load_img(fp)
    # resample to target space
    map_MNI152NLin6Asym = image.resample_to_img(
        source_img=map_MNI152NLin6Asym, 
        target_img=template_MNI152NLin6Asym, 
        interpolation="continuous",
        force_resample=True,
        copy_header=True
    )
    # fix data type
    map_MNI152NLin6Asym = image.math_img("img.astype(np.float32)", img=map_MNI152NLin6Asym)
    # maps might be scaled from 0 or 1 to 100, we want them to be scaled from 0 to 1
    if map_MNI152NLin6Asym.get_fdata().max() > 50:
        map_MNI152NLin6Asym = image.math_img("img / 100", img=map_MNI152NLin6Asym)
    print(map_MNI152NLin6Asym.get_fdata().min(), map_MNI152NLin6Asym.get_fdata().max())
    fp = nispace_source_data_path / "reference" / "tpm" / "map" / m / f"{m}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    fp.parent.mkdir(parents=True, exist_ok=True)
    map_MNI152NLin6Asym.to_filename(fp)
    print(f"Saved {m} to {fp}...")
    
    # ----------------------------------------------------------------------------------------------
    # space: MNI152NLin2009cAsym
    # ----------------------------------------------------------------------------------------------
    print("Attempting MNI152NLin2009cAsym transform...")
    map_MNI152NLin2009cAsym = apply_transform(
        map_MNI152NLin6Asym, mni_from="MNI152NLin6Asym", mni_to="MNI152NLin2009cAsym", order=3
    )
    map_MNI152NLin2009cAsym.to_filename(fp.parent / fp.name.replace("6Asym", "2009cAsym"))
    
    # ----------------------------------------------------------------------------------------------
    # space: fsLR
    # ----------------------------------------------------------------------------------------------
    # transform from MNI152NLin6Asym with neuromaps
    print("Attempting fsLR transform...")
    map_fsLR = transforms.mni152_to_fslr(map_MNI152NLin6Asym, fslr_density="32k", method="linear")
    map_fsLR[0].to_filename(fp.parent / f"{m}_space-fsLR_desc-proc_hemi-L.shape.gii.gz")
    map_fsLR[1].to_filename(fp.parent / f"{m}_space-fsLR_desc-proc_hemi-R.shape.gii.gz")
    
    # ----------------------------------------------------------------------------------------------
    # space: fsaverage
    # ----------------------------------------------------------------------------------------------
    # transform from MNI152NLin6Asym with neuromaps
    print("Attempting fsaverage transform...")
    map_fsaverage = transforms.mni152_to_fsaverage(map_MNI152NLin6Asym, fsavg_density="41k", method="linear")
    map_fsaverage[0].to_filename(fp.parent / f"{m}_space-fsaverage_desc-proc_hemi-L.shape.gii.gz")
    map_fsaverage[1].to_filename(fp.parent / f"{m}_space-fsaverage_desc-proc_hemi-R.shape.gii.gz")
        


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "tpm"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

# %%
