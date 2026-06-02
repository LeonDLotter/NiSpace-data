# %% Init

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy.io import loadmat
from neuromaps import transforms
wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.utils.utils_datasets import download

nispace_source_data_path = wd


# %% Download Kong2022 17-network MSHBM parcellations
# source: https://github.com/ThomasYeoLab/Kong2022_ArealMSHBM
# paper: https://doi.org/10.1093/cercor/bhab101

url = ("https://raw.githubusercontent.com/ThomasYeoLab/Kong2022_ArealMSHBM/"
       "9982416f18b4b2fcbc7d7df8be7ccea9db471f04/"
       "Parcellations/17/HCP_1029sub_17Networks_Kong2019_MSHBM.mat")
archive_dir = nispace_source_data_path / "_archive" / "rsn17"
archive_dir.mkdir(parents=True, exist_ok=True)
fp_mat = archive_dir / "HCP_1029sub_17Networks_Kong2019_MSHBM.mat"
download(url, fp_mat)


# %% Load parcellations and define network names

mat = loadmat(fp_mat)
print("Keys:", [k for k in mat.keys() if not k.startswith("_")])

lh_labels = mat["lh_labels_all"]  # (32492, 1029)
rh_labels = mat["rh_labels_all"]  # (32492, 1029)
n_subjects = lh_labels.shape[1]
print(f"n_vertices (LH): {lh_labels.shape[0]}, n_subjects: {n_subjects}")

# Network names derived from color-matching Kong mat colors to paper figure legend
# Kong R, Yang Q, Gordon E, et al. Cerebral Cortex 2023. doi:10.1093/cercor/bhab101
NETWORK_NAMES = {
    1:  "Auditory",
    2:  "DorsAttnA",
    3:  "ControlA",
    4:  "SomatomotorA",
    5:  "SalVenAttnB",
    6:  "DefaultB",
    7:  "DefaultC",
    8:  "VisualC",
    9:  "VisualA",
    10: "DorsAttnB",
    11: "Language",
    12: "ControlB",
    13: "VisualB",
    14: "ControlC",
    15: "DefaultA",
    16: "SalVenAttnA",
    17: "SomatomotorB",
}


# %% Compute probability maps and save

for k, network_name in NETWORK_NAMES.items():
    map_id = f"nw-{network_name}_pub-kong2022"
    print(f"Processing: {map_id}")

    map_dir = nispace_source_data_path / "reference" / "rsn17" / "map" / map_id
    map_dir.mkdir(parents=True, exist_ok=True)

    # Probability map: fraction of subjects with vertex assigned to network k
    prob_L = (lh_labels == k).sum(axis=1).astype(np.float32) / n_subjects  # (32492,)
    prob_R = (rh_labels == k).sum(axis=1).astype(np.float32) / n_subjects  # (32492,)

    # Create GIfTI images
    gifti_L = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(prob_L)])
    gifti_R = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(prob_R)])

    # space: fsLR 32k — source space, save directly
    gifti_L.to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-L.surf.gii.gz")
    gifti_R.to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-R.surf.gii.gz")

    # space: fsaverage 41k — transform using neuromaps
    map_fsavg = transforms.fslr_to_fsaverage(
        (gifti_L, gifti_R), target_density="41k", method="linear"
    )
    map_fsavg[0].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-L.surf.gii.gz")
    map_fsavg[1].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-R.surf.gii.gz")

    print(f"  Saved all spaces for {map_id}")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "rsn17"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)


# %% QC plots

# from nispace.plotting import brainplot
# import matplotlib.pyplot as plt

# qc_dir = archive_dir / "qc"
# qc_dir.mkdir(parents=True, exist_ok=True)
# space = "fsLR"

# for map_dir in sorted((ref_dir / "map").iterdir()):
#     map_id = map_dir.name
#     lh = nib.load(map_dir / f"{map_id}_space-{space}_desc-proc_hemi-L.surf.gii.gz")
#     rh = nib.load(map_dir / f"{map_id}_space-{space}_desc-proc_hemi-R.surf.gii.gz")
#     fig = brainplot((lh, rh), space=space, title=map_id, surf_mesh="veryinflated")
#     plt.show()

# %%
