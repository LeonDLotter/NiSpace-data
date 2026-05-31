# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from neuromaps import images

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.nulls import get_distance_matrix

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import save_csv_gz

# nispace data path 
nispace_source_data_path = wd

# all parcellations 
PARCS = sorted(
    [p.name for p in (nispace_source_data_path / "parcellation").glob("*") if p.is_dir()]
)
print("PARCS:", PARCS)

# %% Generate distance matrices

for parc in PARCS:
    print("Parcellation:", parc)
    spaces = sorted(
        [s.name for s in (nispace_source_data_path / "parcellation" / parc).glob("*") if s.is_dir()]
    )
    print("Available spaces:", spaces)
    
    for space in spaces:
        print(parc, space)
        
        # load
        if "mni" in space.lower():
            parc_loaded = images.load_nifti(
                nispace_source_data_path / "parcellation" / parc / space / 
                f"parc-{parc}_space-{space}.label.nii.gz"
            )
        else:
            parc_loaded = (
                images.load_gifti(nispace_source_data_path / "parcellation" / parc / space / 
                                  f"parc-{parc}_space-{space}_hemi-L.label.gii.gz"),
                images.load_gifti(nispace_source_data_path / "parcellation" / parc / space /    
                                  f"parc-{parc}_space-{space}_hemi-R.label.gii.gz")
            )
            
        # distance matrix 
        dist_mat = get_distance_matrix(
            parc_loaded, 
            parc_space=space,
            parc_resample={
                "MNI152NLin2009cAsym": 2, "MNI152NLin6Asym": 2, "fsLR": "32k", "fsaverage": "41k" 
            }[space],
            centroids=False,
            surf_euclidean=False,
            n_proc=-1,
            dtype=np.float32
        )
        if not isinstance(dist_mat, tuple):
            parc_idc = np.trim_zeros(np.unique(parc_loaded.get_fdata()))
            if not (parc_idc.shape[0] == dist_mat.shape[0] == dist_mat.shape[1]):
                raise ValueError("Shape mismatch between parcellation and distance matrix:"
                                 f"parc_idc.shape={parc_idc.shape}, dist_mat.shape={dist_mat.shape}")
            save_csv_gz(pd.DataFrame(dist_mat),
                        nispace_source_data_path / "parcellation" / parc / space /
                        f"parc-{parc}_space-{space}.dist.csv.gz",
                        header=None, index=None)
        else:
            parc_idc_lh = np.trim_zeros(np.unique(parc_loaded[0].agg_data()))
            parc_idc_rh = np.trim_zeros(np.unique(parc_loaded[1].agg_data()))
            if not (parc_idc_lh.shape[0] == dist_mat[0].shape[0] == dist_mat[0].shape[1]):
                raise ValueError("Shape mismatch between parcellation and distance matrix:"
                                 f"parc_idc_lh.shape={parc_idc_lh.shape}, dist_mat[0].shape={dist_mat[0].shape}")
            if not (parc_idc_rh.shape[0] == dist_mat[1].shape[0] == dist_mat[1].shape[1]):
                raise ValueError("Shape mismatch between parcellation and distance matrix:"
                                 f"parc_idc_rh.shape={parc_idc_rh.shape}, dist_mat[1].shape={dist_mat[1].shape}")
            for mat, hemi in zip(dist_mat, ["L", "R"]):
                save_csv_gz(pd.DataFrame(mat),
                            nispace_source_data_path / "parcellation" / parc / space /
                            f"parc-{parc}_space-{space}_hemi-{hemi}.dist.csv.gz",
                            header=None, index=None)
# %%
