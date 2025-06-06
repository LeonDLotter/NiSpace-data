# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from neuromaps import images

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.nulls import get_distance_matrix

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
        if space == "fsLR":
            continue
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
            parc_resample=2,
            centroids=False,
            surf_euclidean=False,
            n_proc=-1,
            dtype=np.float32
        )
        if not isinstance(dist_mat, tuple):
            pd.DataFrame(dist_mat).to_csv(
                nispace_source_data_path / "parcellation" / parc / space / 
                f"parc-{parc}_space-{space}.dist.csv.gz", 
                header=None, index=None
            )
        else:
            for mat, hemi in zip(dist_mat, ["L", "R"]):
                pd.DataFrame(mat).to_csv(
                    nispace_source_data_path / "parcellation" / parc / space / 
                    f"parc-{parc}_space-{space}_hemi-{hemi}.dist.csv.gz", 
                    header=None, index=None
                )
# %%
