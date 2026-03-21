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


# %% NEW APPROACH: Cross-hemisphere parcel correlations from GRF random maps
# Produces a n_lh x n_rh correlation matrix capturing structural correspondence
# between LH and RH parcels. Used ONLY for the off-diagonal (LH-RH, RH-LH)
# blocks of a full bilateral weight matrix W for null map generation.
# The diagonal (within-hemisphere) blocks must use actual surface distances.

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
        n_lh, n_rh = len(lh_labels), len(rh_labels)

        # load GRF random maps (alpha-0.0: no smoothing, bilateral structure
        # comes from the full-brain GRF generation)
        grf_data = pd.read_csv(
            nispace_source_data_path / "reference" / "grf" / "tab" /
            f"dset-grf_parc-{parc}.csv.gz", index_col=0
        )
        grf_data = grf_data.loc[grf_data.index.str.contains("alpha-0.0")]
        print("Shape of grf_data:", grf_data.shape)

        # compute cross-hemisphere correlation matrix (n_lh x n_rh)
        # np.corrcoef on stacked transposed arrays gives full (n_lh+n_rh)^2 matrix;
        # we take the top-right block
        X_lh = grf_data.loc[:, lh_labels].values  # n_maps x n_lh
        X_rh = grf_data.loc[:, rh_labels].values  # n_maps x n_rh
        full_corr = np.corrcoef(np.vstack([X_lh.T, X_rh.T]))
        C = pd.DataFrame(
            full_corr[:n_lh, n_lh:],
            index=pd.Index(lh_labels, name="lh"),
            columns=pd.Index(rh_labels, name="rh"),
            dtype=float
        )

        # threshold weak correlations
        C[C < 0.1] = 0

        # plot
        sn.heatmap(
            C.replace(0, np.nan),
            cmap="viridis",
            vmin=0,
            vmax=1
        )
        plt.title(parc)
        plt.show()

        # save
        C.to_csv(
            nispace_source_data_path / "parcellation" / parc / space /
            f"parc-{parc}_space-{space}.lrcorr.csv.gz"
        )


#%% DEPRECATED: Generate LR mapping from regression on random maps.
# Commented out: worked well (interhemi r > 0.8) but asymmetric by construction
# (OLS predicting RH from LH != OLS predicting LH from RH), which causes bias
# when used as cross-hemisphere weights in a symmetric W matrix.

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
            y = grf_data.loc[:, rh_label]
            model = LinearRegression()
            model.fit(X, y)
            betas = pd.Series(model.coef_, index=lh_labels)
            if np.isclose(betas.sum(), 0):
                raise ValueError(f"Sum of betas after fitting {rh_label} model is 0")
            betas[betas < 0.1] = 0
            if np.isclose(betas.sum(), 0):
                raise ValueError(f"Sum of betas after thresholding {rh_label} model is 0")
            betas = betas / betas.sum()
            if not np.isclose(betas.sum(), 1):
                raise ValueError(f"Sum of betas after adjusting {rh_label} model is not 1")
            betas_df.loc[lh_labels, rh_label] = betas

        # plot
        sn.heatmap(betas_df.replace(0, np.nan), cmap="viridis", vmin=0, vmax=1)
        plt.title(parc)
        plt.show()

        # save
        betas_df.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")



# %% DEPRECATED: Generate LR mapping from fsLR vertex overlap.
# Commented out: produces a diffuse mapping for asymmetric parcellations
# because non-symmetric parcels don't have clean vertex-to-vertex correspondence.
# Column normalization dilutes diagonal weights, giving ~0.2 interhemi r.

# for parc in PARCS:
#     print("Parcellation:", parc)
#
#     # fetch fsLR maps, labels, and contained data
#     fp = nispace_source_data_path / "parcellation" / parc / "fsLR" / f"parc-{parc}_space-fsLR_hemi-%s.label.gii.gz"
#     (img_lh, img_rh) = load_img(
#         (fp.as_posix() % "L", fp.as_posix() % "R"))
#     (labels_lh, labels_rh) = load_labels(
#         (fp.as_posix().replace(".gii.gz", ".txt") % "L", fp.as_posix().replace(".gii.gz", ".txt") % "R"), concat=False)
#     dat_lh, dat_rh = img_lh.agg_data().astype(int), img_rh.agg_data().astype(int)
#     idc_lh, idc_rh = np.trim_zeros(np.unique(dat_lh)), np.trim_zeros(np.unique(dat_rh))
#     assert len(idc_lh) == len(labels_lh), "Number of left indices does not match number of left labels"
#     assert len(idc_rh) == len(labels_rh), "Number of right indices does not match number of right labels"
#
#     # fetch fslr medial wall
#     den = _img_density_for_neuromaps(img_lh)
#     medial_lh, medial_rh = load_img(fetch_fslr(den)["medial"])
#
#     # mask
#     dat_lh[medial_lh.agg_data() == 0] = 0
#     dat_rh[medial_rh.agg_data() == 0] = 0
#
#     # iterate left indices
#     l2r_map = pd.DataFrame(np.zeros((len(idc_lh), len(idc_rh)), dtype=float), index=idc_lh, columns=idc_rh)
#     for lh_idx in tqdm(idc_lh):
#         lh_loc = np.where(dat_lh == lh_idx)[0]
#         rh_idc_at_loc, rh_counts_at_loc = np.unique(dat_rh[lh_loc], return_counts=True)
#         isnull = rh_idc_at_loc == 0
#         rh_idc_at_loc, rh_counts_at_loc = rh_idc_at_loc[~isnull], rh_counts_at_loc[~isnull]
#         assert 0 not in rh_idc_at_loc, "0 is in right indices"
#         l2r_map.loc[lh_idx, rh_idc_at_loc] = rh_counts_at_loc
#
#     # column-normalize
#     col_sums = l2r_map.sum(axis=0)
#     assert (col_sums > 0).all(), "Some RH parcels have zero total overlap"
#     l2r_map = l2r_map.div(col_sums, axis=1)
#     assert np.allclose(l2r_map.sum(axis=0), 1), "Column sums do not equal 1"
#
#     # assign labels
#     l2r_map.index = labels_lh
#     l2r_map.columns = labels_rh
#
#     # save
#     for space in parcellation_lib[parc]:
#         l2r_map.to_csv(nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.l2rmap.csv.gz")
#
#     # plot
#     sn.heatmap(l2r_map.replace(0, np.nan), cmap="viridis", vmin=0, vmax=1)
#     plt.title(parc)
#     plt.show()


# %%
