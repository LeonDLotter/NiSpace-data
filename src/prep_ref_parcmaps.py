# %% Init

import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from utils import parcellate_reference_dataset 

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference, reference_lib, parcellation_lib

# nispace data path 
nispace_source_data_path = wd

# datasets with maps
DSETS_WITH_MAPS = [k for k, v in reference_lib.items() if "map" in v]
print("DSETS_WITH_MAPS: ", DSETS_WITH_MAPS)

# parcellations: MNI152NLin
PARCS_MNI152NLin2009cAsym = [k for k, v in parcellation_lib.items() if "MNI152NLin2009cAsym" in v]
print("PARCS_MNI152NLin2009cAsym: ", PARCS_MNI152NLin2009cAsym)

# parcelations: fsaverage only
PARCS_FSA = [k for k, v in parcellation_lib.items() if "fsaverage" in v]
print("PARCS_FSA: ", PARCS_FSA)


# %% Parcellate map-based image data ---------------------------------------------------------------

# iterate datasets
for dataset in DSETS_WITH_MAPS:
    print("-------- " + dataset.upper() + " --------")
    
    for parcs, parc_space, ref_space in [
        (PARCS_MNI152NLin2009cAsym, "MNI152NLin2009cAsym", "MNI152NLin2009cAsym"), 
        (PARCS_FSA, "fsaverage", "MNI152NLin6Asym")
    ]:
        
        # get files
        files = fetch_reference(dataset, space=ref_space if dataset != "rsn" else "MNI152")
    
        # parcellate
        parcellate_reference_dataset(
            reference_name=dataset,
            reference_files=files,
            reference_path=nispace_source_data_path / "reference" / dataset,
            data_space=ref_space,
            data_labels=[f.name.split("_space")[0] for f in files],
            parc_space=parc_space,
            parcs=parcs,
        )


# %%
