# %% Init

import os
import sys
from pathlib import Path
import numpy as np
from neuromaps import transforms
import nibabel as nib
from nilearn import image
import subprocess
import tempfile
import zipfile
import templateflow.api as tf

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference, fetch_parcellation, reference_lib, parcellation_lib, fetch_template
from nispace.io import parcellate_data, load_img
from nispace.utils.utils_datasets import get_file, download
from nispace.stats.misc import residuals 

# nispace data path 
nispace_source_data_path = wd


# %% Prep the GM/WM/CSF TPM atlas
# source: https://www.unil.ch/lren/home/menuinst/teaching--utilities/data--utilities.html
# paper: https://doi.org/10.1016/j.neuroimage.2016.01.062

# link to the data
url = "https://www.unil.ch/files/live/sites/lren/files/shared/bogdan/enhanced_TPM.zip"

# download, unzip, load, and save all in temporary directory
with tempfile.TemporaryDirectory() as tmp_dir:
    
    # download
    tmp_dir = Path(tmp_dir)
    fp = tmp_dir / "enhanced_TPM.zip"
    download(url, fp)
    with zipfile.ZipFile(fp, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)
        
    # load 4D volume
    tpm_vol = image.load_img(tmp_dir / "enhanced_TPM.nii")
    
    # save tissue types we want
    for idx, tissue in [(0, "gm"), (1, "wm"), (2, "csf")]:
        tpm_vol_tissue = image.index_img(tpm_vol, idx)
        fp = (nispace_source_data_path / "reference" / "tpm" / "map" / f"tissue-{tissue}_pub-lorio2016" / 
              f"tissue-{tissue}_pub-lorio2016_space-MNI152.nii.gz")
        fp.parent.mkdir(parents=True, exist_ok=True)
        tpm_vol_tissue.to_filename(fp)  
    
        
# %% Load all atlases via reference_lib
template_MNI152NLin6Asym = fetch_template("MNI152NLin6Asym", desc="mask", res="2mm")
template_MNI152NLin2009cAsym = fetch_template("MNI152NLin2009cAsym", desc="mask", res="2mm")

for m in reference_lib["tpm"]["map"]:
    print("Processing map:", m)

    # get original map
    if any(t in m for t in ["gm", "wm", "csf"]):
        os.environ["NISPACE_DATA_DIR"] = str(nispace_source_data_path)
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
        interpolation="continuous"
    )
    # apply mask and set data type
    map_MNI152NLin6Asym = image.math_img("(img * mask).astype(np.float32)", img=map_MNI152NLin6Asym, mask=template_MNI152NLin6Asym)
    # maps might be scaled from 0 or 1 to 100, we want them to be scaled from 0 to 1
    if map_MNI152NLin6Asym.get_fdata().max() > 50:
        map_MNI152NLin6Asym = image.math_img("img / 100", img=map_MNI152NLin6Asym)
    print(map_MNI152NLin6Asym.get_fdata().min(), map_MNI152NLin6Asym.get_fdata().max())
    # save
    fp = nispace_source_data_path / "reference" / "tpm" / "map" / m / f"{m}_space-MNI152NLin6Asym_desc-proc.nii.gz"
    fp.parent.mkdir(parents=True, exist_ok=True)
    map_MNI152NLin6Asym.to_filename(fp)
    print(f"Saved {m} to {fp}...")
    
    # ----------------------------------------------------------------------------------------------
    # space: MNI152NLin2009cAsym
    # ----------------------------------------------------------------------------------------------
    # transform with pre-estimated transform from templateflow and antsApplyTransforms
    print("Attempting MNI152NLin2009cAsym transform...")
    transform = tf.get(template="MNI152NLin2009cAsym", extension="h5")
    assert isinstance(transform, Path), "Transform is not a Path. List?"
    cmd = (
        "/Applications/ants-2.6.0/bin/antsApplyTransforms "
        f"-d 3 -i {fp} -r {template_MNI152NLin2009cAsym} -t {transform} "
        f"-o {fp.parent / fp.name.replace('6Asym', '2009cAsym')}"
    )
    subprocess.run(cmd, shell=True)
    
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
        
    # reset environment variable
    os.environ["NISPACE_DATA_DIR"] = str(Path.home() / "nispace-data")
        
# %%
