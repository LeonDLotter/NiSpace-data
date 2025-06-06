# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
# ...

# nispace data path 
nispace_source_data_path = wd

# all parcellations to iterate
# ONLY NON-SYMMETRIC PARCELLATIONS FOR THIS
PARCS = ["Schaefer100", "Schaefer200", "Schaefer400"]

#%% TEMP to get l2rmap for Schaefer parcellations

# for parc, parc_old in [("Schaefer100", "Schaefer100MelbourneS1"), 
#                        ("Schaefer200", "Schaefer200MelbourneS2"), 
#                        ("Schaefer400", "Schaefer400MelbourneS3")]:
#     print(parc, parc_old)
    
#     for space in ["MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsLR", "fsaverage"]:
        
#         # load old l2rmap
#         l2rmap_old = pd.read_csv(nispace_source_data_path / "parcellation" / parc_old / space / f"parc-{parc_old}_space-{space}.l2rmap.csv.gz", index_col=0)
#         print("old l2rmap:", l2rmap_old.shape)
#         display(l2rmap_old.head())
#         assert np.allclose(l2rmap_old.sum(), 1.0)
        
#         # number of parcels on one hemisphere
#         schaefer_n = int(parc[-3:]) // 2
#         print(schaefer_n)
        
#         # get l2rmap for cortex only
#         l2rmap = l2rmap_old.iloc[:schaefer_n, :schaefer_n]
        
#         # new labels
#         if "mni" in space.lower():
#             labels = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.label.txt", dtype=str
#             )
#             labels_lh = [l for l in labels if "hemi-L_" in l]
#             labels_rh = [l for l in labels if "hemi-R_" in l]
#         else:
#             labels_lh = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}_hemi-L.label.txt", dtype=str
#             ).tolist()
#             labels_rh = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}_hemi-R.label.txt", dtype=str
#             ).tolist()
        
#         # check
#         assert len(labels_lh) == len(labels_rh) == schaefer_n
        
#         # assign new labels
#         l2rmap.index = labels_lh
#         l2rmap.columns = labels_rh
        
#         # checks
#         display(l2rmap.head())
#         print("new l2rmap:", l2rmap.shape)
#         assert np.allclose(l2rmap.sum(), 1.0)
        
#         # save
#         l2rmap.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")



# %% Generate distance matrices

# fit models
for parc in PARCS:
    print("Parcellation:", parc)
    spaces = sorted(
        [s.name for s in (nispace_source_data_path / "parcellation" / parc).glob("*") if s.is_dir()]
    )
    print("Available spaces:", spaces)
    
    for space in spaces:
        print(parc, space)
        
        # load labels
        if "mni" in space.lower():
            labels = np.loadtxt(
                nispace_source_data_path / "parcellation" / parc / space / 
                f"parc-{parc}_space-{space}.label.txt", dtype=str
            )
            lh_labels = [l for l in labels if "hemi-L_" in l]
            rh_labels = [l for l in labels if "hemi-R_" in l]
        else:
            lh_labels = np.loadtxt(
                nispace_source_data_path / "parcellation" / parc / space / 
                f"parc-{parc}_space-{space}_hemi-L.label.txt", dtype=str
            )
            rh_labels = np.loadtxt(
                nispace_source_data_path / "parcellation" / parc / space / 
                f"parc-{parc}_space-{space}_hemi-R.label.txt", dtype=str
            )
        
        # load random maps
        grf_data = pd.read_csv(nispace_source_data_path / "reference" / "grf" / "tab" / f"dset-grf_parc-{parc}.csv.gz", index_col=0)
        grf_data = grf_data.loc[grf_data.index.str.contains("alpha-0.0")]
        print("Shape of grf_data:", grf_data.shape)
        
        # df to store betas
        betas_df = pd.DataFrame(
            index=pd.Index(lh_labels, name="lh_betas"), 
            columns=pd.Index(rh_labels, name="rh_predicted"),
            dtype=float
        )
        
        # predictors are the LH labels
        X = grf_data.loc[:, lh_labels]
        
        # fit model predicting each rh label from lh labels
        for rh_label in tqdm(rh_labels):
            
            # response
            y = grf_data.loc[:, rh_label]
            
            # fit model
            model = LinearRegression()
            model.fit(X, y)
            
            # weights
            betas = pd.Series(model.coef_, index=lh_labels)
            
            # check
            if np.isclose(betas.sum(), 0):
                raise ValueError(f"Sum of betas after fitting {rh_label} model is 0")
            
            # threshold weights
            betas[betas < 0.1] = 0
            
            # check again
            if np.isclose(betas.sum(), 0):
                raise ValueError(f"Sum of betas after thresholding {rh_label} model is 0")
            
            # adjust to sum to 1
            betas = betas / betas.sum()
            
            # check again
            if not np.isclose(betas.sum(), 1):
                raise ValueError(f"Sum of betas after adjusting {rh_label} model is not 1")
            
            # store
            betas_df.loc[lh_labels, rh_label] = betas
            
        # show
        print("Beta df:")
        print(betas_df.head())
        
        # save
        betas_df.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")
        
            
            
            
            
            
        
        
        
        
# %%
