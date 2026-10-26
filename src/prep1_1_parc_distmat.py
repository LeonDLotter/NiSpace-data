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
from utils import load_parc_lists, save_csv_gz, parc_files, InputStamps

# nispace data path
nispace_source_data_path = wd

# all parcellations (aliases excluded)
PARCS, _, _ = load_parc_lists(wd)
print("PARCS:", PARCS)

# settings (part of the input stamps: changing them invalidates the matrices)
DIST_VERSION = 1
PARC_RESAMPLE = {"MNI152NLin2009cAsym": 2, "MNI152NLin6Asym": 2, "fsLR": "32k", "fsaverage": "41k"}

# %% Generate distance matrices

n_computed, n_fresh = 0, 0
for parc in PARCS:
    print("Parcellation:", parc)
    spaces = sorted(
        [s.name for s in (nispace_source_data_path / "parcellation" / parc).glob("*")
         if s.is_dir() and s.name != "plot"]
    )
    print("Available spaces:", spaces)
    stamps = InputStamps.for_parc(parc, scope="distmat")
    
    for space in spaces:
        print(parc, space)

        # skip if the parcellation image is unchanged (label names do not affect distances)
        out_base = nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}"
        outputs = ([Path(f"{out_base}.dist.csv.gz")] if "mni" in space.lower() else
                   [Path(f"{out_base}_hemi-{h}.dist.csv.gz") for h in ("L", "R")])
        stamp = stamps.check(
            f"{space}/dist",
            files=parc_files(wd, parc, space, labels=False),
            params={"version": DIST_VERSION, "resample": PARC_RESAMPLE[space], "centroids": False,
                    "surf_euclidean": False, "dtype": "float32"},
            outputs=outputs,
        )
        if stamp is None:
            n_fresh += 1
            continue
        n_computed += 1
        
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
            parc_resample=PARC_RESAMPLE[space],
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
        stamps.write(f"{space}/dist", stamp)

print(f"[inputs] distmat: {n_computed} computed, {n_fresh} up to date")

# %%
