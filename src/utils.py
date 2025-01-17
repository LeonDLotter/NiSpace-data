from pathlib import Path
import pickle
import gzip
import numpy as np
import pandas as pd

import sys

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

from nispace.modules.constants import _PARCS_NICE
from nispace.io import parcellate_data
from nispace.nulls import generate_null_maps
from nispace.datasets import fetch_parcellation
from nispace.utils.utils import _rm_ext
    

def parcellate_reference_dataset(reference_name, reference_files, reference_path=None,
                                 data_labels=None, 
                                 data_space="MNI152NLin2009cAsym", 
                                 parc_space="MNI152NLin2009cAsym",
                                 parcs=_PARCS_NICE, nulls=False,
                                 **kwargs):
    
    reference_path = Path(reference_path)

    for parc in parcs:
        print(parc)

        # get parcellation
        parc_loaded, parc_labels, parc_space, parc_distmat = \
            fetch_parcellation(
                parc, 
                space=parc_space, 
                return_dist_mat=True, 
                return_space=True, 
                return_loaded=True
            )

        # parcellate  
        tab = parcellate_data(
            parcellation=parc_loaded,
            parc_labels=parc_labels,
            parc_space=parc_space,
            parc_hemi=["L", "R"],
            resampling_target="data" if "mni" in parc_space.lower() else "parcellation",
            data=reference_files,
            data_labels=[_rm_ext(f.name) for f in reference_files] if data_labels is None else data_labels,
            data_space="MNI152",
            n_proc=-1,
            dtype=np.float32,
            **{
                "drop_background_parcels": True,
                "min_num_valid_datapoints": 5,
                "min_fraction_valid_datapoints": 0.3,
            } | kwargs
        )
        tab.index.name = "map"
        tab.to_csv(reference_path / "tab" / f"dset-{reference_name}_parc-{parc}.csv.gz")
        
        if nulls:
            # null maps
            null_maps, _ = generate_null_maps(
                "moran",
                tab, 
                parcellation=parc_loaded,
                dist_mat=parc_distmat,
                parc_space=parc_space,
                n_nulls=10000,
                dtype=np.float16,
                n_proc=-1,
                seed=42
            )
            with gzip.open(reference_path / "null" / f"{reference_name}_{parc}.pkl.gz", "wb") as f:
                pickle.dump(null_maps, f, pickle.HIGHEST_PROTOCOL)
        
        
        