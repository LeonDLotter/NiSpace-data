# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm
from neuromaps.datasets import fetch_fslr
import seaborn as sn
import matplotlib.pyplot as plt

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.io import load_labels, load_img
from nispace.nulls import _img_density_for_neuromaps
from nispace.datasets import parcellation_lib

# nispace data path 
nispace_source_data_path = wd

# all parcellations to iterate
# ONLY NON-SYMMETRIC PARCELLATIONS FOR THIS
PARCS = ["Schaefer100", "Schaefer200", "Schaefer400"]

# %% Generate LR mapping from fsLR versions of parcellation

for parc in PARCS:
    print("Parcellation:", parc)
    
    # fetch fsLR maps, labels, and contained data
    fp = nispace_source_data_path / "parcellation" / parc / "fsLR" / f"parc-{parc}_space-fsLR_hemi-%s.label.gii.gz"
    (img_lh, img_rh) = load_img(
        (fp.as_posix() % "L", fp.as_posix() % "R"))
    (labels_lh, labels_rh) = load_labels(
        (fp.as_posix().replace(".gii.gz", ".txt") % "L", fp.as_posix().replace(".gii.gz", ".txt") % "R"), concat=False)
    dat_lh, dat_rh = img_lh.agg_data().astype(int), img_rh.agg_data().astype(int)
    idc_lh, idc_rh = np.trim_zeros(np.unique(dat_lh)), np.trim_zeros(np.unique(dat_rh))
    assert len(idc_lh) == len(labels_lh), "Number of left indices does not match number of left labels"
    assert len(idc_rh) == len(labels_rh), "Number of right indices does not match number of right labels"
    
    # fetch fslr medial wall
    den = _img_density_for_neuromaps(img_lh)
    medial_lh, medial_rh = load_img(fetch_fslr(den)["medial"])
    
    # mask
    dat_lh[medial_lh.agg_data() == 0] = 0
    dat_rh[medial_rh.agg_data() == 0] = 0
    
    # iterate left indices
    l2r_map = pd.DataFrame(np.zeros((len(idc_lh), len(idc_rh)), dtype=float), index=idc_lh, columns=idc_rh)
    for lh_idx in tqdm(idc_lh):
        
        # get vertex locations in left hemi
        lh_loc = np.where(dat_lh == lh_idx)[0]
        
        # distributions of indices in right hemi in these locations
        rh_idc_at_loc, rh_counts_at_loc = np.unique(dat_rh[lh_loc], return_counts=True)
        
        # remove 0
        isnull = rh_idc_at_loc == 0
        rh_idc_at_loc, rh_counts_at_loc = rh_idc_at_loc[~isnull], rh_counts_at_loc[~isnull]
        
        # calculate fractions
        frac = rh_counts_at_loc / rh_counts_at_loc.sum()
        assert 0 not in rh_idc_at_loc, "0 is in right indices"
        assert np.isclose(frac.sum(), 1), "Fractions do not sum to 1"
        
        # store
        l2r_map.loc[lh_idx, rh_idc_at_loc] = frac
    
    # assign labels
    l2r_map.index = labels_lh
    l2r_map.columns = labels_rh
    
    # save
    # TODO: this is stupd, we will save the same for each space...
    for space in parcellation_lib[parc]:
        l2r_map.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")
    
    # plot
    sn.heatmap(
        l2r_map.replace(0, np.nan),
        cmap="viridis",
        vmin=0,
        vmax=1
    )
    plt.title(parc)
    plt.show()
     

# %% Generate LR mapping from regression on random maps. 
# DEPRECATED APPROACH BUT WORKED PRETTY WELL, TOO

# # fit models
# for parc in PARCS[:1]:
#     print("Parcellation:", parc)
#     spaces = sorted(
#         [s.name for s in (nispace_source_data_path / "parcellation" / parc).glob("*") if s.is_dir()]
#     )
#     print("Available spaces:", spaces)
    
#     for space in spaces:
#         print(parc, space)
        
#         # load labels
#         if "mni" in space.lower():
#             labels = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / 
#                 f"parc-{parc}_space-{space}.label.txt", dtype=str
#             )
#             lh_labels = [l for l in labels if "hemi-L_" in l]
#             rh_labels = [l for l in labels if "hemi-R_" in l]
#         else:
#             lh_labels = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / 
#                 f"parc-{parc}_space-{space}_hemi-L.label.txt", dtype=str
#             )
#             rh_labels = np.loadtxt(
#                 nispace_source_data_path / "parcellation" / parc / space / 
#                 f"parc-{parc}_space-{space}_hemi-R.label.txt", dtype=str
#             )
        
#         # load random maps
#         grf_data = pd.read_csv(nispace_source_data_path / "reference" / "grf" / "tab" / f"dset-grf_parc-{parc}.csv.gz", index_col=0)
#         grf_data = grf_data.loc[grf_data.index.str.contains("alpha-0.0")]
#         print("Shape of grf_data:", grf_data.shape)
        
#         # df to store betas
#         betas_df = pd.DataFrame(
#             index=pd.Index(lh_labels, name="lh_betas"), 
#             columns=pd.Index(rh_labels, name="rh_predicted"),
#             dtype=float
#         )
        
#         # predictors are the LH labels
#         X = grf_data.loc[:, lh_labels]
        
#         # fit model predicting each rh label from lh labels
#         for rh_label in tqdm(rh_labels):
            
#             # response
#             y = grf_data.loc[:, rh_label]
            
#             # fit model
#             model = LinearRegression()
#             model.fit(X, y)
            
#             # weights
#             betas = pd.Series(model.coef_, index=lh_labels)
            
#             # check
#             if np.isclose(betas.sum(), 0):
#                 raise ValueError(f"Sum of betas after fitting {rh_label} model is 0")
            
#             # threshold weights
#             betas[betas < 0.1] = 0
            
#             # check again
#             if np.isclose(betas.sum(), 0):
#                 raise ValueError(f"Sum of betas after thresholding {rh_label} model is 0")
            
#             # adjust to sum to 1
#             betas = betas / betas.sum()
            
#             # check again
#             if not np.isclose(betas.sum(), 1):
#                 raise ValueError(f"Sum of betas after adjusting {rh_label} model is not 1")
            
#             # store
#             betas_df.loc[lh_labels, rh_label] = betas
            
#         # show
#         print("Beta df:")
#         print(betas_df.head())
        
#         # save
#         betas_df.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")
        
        
        
# %%
