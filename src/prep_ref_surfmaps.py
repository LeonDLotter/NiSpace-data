# %% Init

import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import nibabel as nib
from neuromaps import images
from neuromaps.resampling import resample_images
from neuromaps import transforms
from sklearn.preprocessing import minmax_scale

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference, fetch_parcellation, reference_lib, parcellation_lib
from nispace.io import parcellate_data
from nispace.utils.utils_datasets import get_file

# nispace data path 
nispace_source_data_path = wd

# datasets with maps
DSETS_WITH_MAPS = [k for k, v in reference_lib.items() if "map" in v]
print("DSETS_WITH_MAPS: ", DSETS_WITH_MAPS)

# %% Convert to surface maps

for dataset in DSETS_WITH_MAPS:
    if dataset == "rsn":
        print("Skipping RSN for now...")
        continue
    
    print("Processing dataset:", dataset)
    
    # iterate maps
    for m in reference_lib[dataset]["map"]:
        from_mni_to_fsa, from_mni_to_fslr, from_fsa_to_fslr, from_fslr_to_fsa = False, False, False, False
        if "private" in reference_lib[dataset]["map"][m]["MNI152"]["host"]:
            print(f"Skipping private map: {m}")
            continue
        
        print("Processing map:", m)
        
        # Cases:
        # original fsaverage exists
        if "fsaverage" in reference_lib[dataset]["map"][m]:
            if not "desc-proc" in reference_lib[dataset]["map"][m]["fsaverage"]["L"]:
                from_fsa_to_fslr = True
            else:
                from_fsa_to_fslr = False
        # original fsLR exists
        elif "fsLR" in reference_lib[dataset]["map"][m]:
            if not "desc-proc" in reference_lib[dataset]["map"][m]["fsLR"]["L"]:
                from_fslr_to_fsa = True
            else:
                from_fslr_to_fsa = False
        # only MNI152NLin6Asym exists
        elif "MNI152NLin6Asym" in reference_lib[dataset]["map"][m]:
            from_mni_to_fsa = True
            from_mni_to_fslr = True
        else:
            raise ValueError(f"Something is wrong with the map: {m}")
        
        # check
        if not any([from_mni_to_fsa, from_mni_to_fslr, from_fsa_to_fslr, from_fslr_to_fsa]):
            raise ValueError(f"All conversions are False for {m}")
        
        # fsaverage to fsLR
        if from_fsa_to_fslr:
            print("Converting from fsaverage to fsLR")
            fp = fetch_reference("pet", m, space="fsaverage", verbose=False)
            if len(fp) == 0 or len(fp) > 1:
                raise ValueError(f"No or multiple files found for {m} in fsaverage")
            fp = fp[0]
            fn_save = fp[0].name.split("_space-")[0] + "_space-fsLR_desc-proc_hemi-%s.surf.gii"
            surf = transforms.fsaverage_to_fslr(
                data=fp,
                target_density="32k",
                method="linear"
            )
            surf[0].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "L"))
            surf[1].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "R"))
        
        # fsLR to fsaverage
        if from_fslr_to_fsa:
            print("Converting from fsLR to fsaverage")
            fp = fetch_reference("pet", m, space="fsLR", verbose=False)
            if len(fp) == 0 or len(fp) > 1:
                raise ValueError(f"No or multiple files found for {m} in fsLR")
            fp = fp[0]
            fn_save = fp[0].name.split("_space-")[0] + "_space-fsaverage_desc-proc_hemi-%s.surf.gii"
            surf = transforms.fslr_to_fsaverage(
                data=fp,
                target_density="41k",
                method="linear"
            )
            surf[0].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "L"))
            surf[1].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "R"))
            
        # MNI152NLin6Asym to fsaverage
        if from_mni_to_fsa or from_mni_to_fslr:
            fp = fetch_reference("pet", m, space="MNI152NLin6Asym", verbose=False)
            if len(fp) == 0 or len(fp) > 1:
                raise ValueError(f"No or multiple files found for {m} in MNI152NLin6Asym")
            fp = fp[0]
            
            if from_mni_to_fsa:
                print("Converting from MNI152NLin6Asym to fsaverage")
                fn_save = fp.name.split("_space-")[0] + "_space-fsaverage_desc-proc_hemi-%s.surf.gii"
                surf = transforms.mni152_to_fsaverage(
                    img=fp,
                    fsavg_density="41k",
                    method="linear"
                )
                surf[0].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "L"))
                surf[1].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "R"))

            if from_mni_to_fslr:
                print("Converting from MNI152NLin6Asym to fsLR")
                fn_save = fp.name.split("_space-")[0] + "_space-fsLR_desc-proc_hemi-%s.surf.gii"
                surf = transforms.mni152_to_fslr(
                    img=fp,
                    fslr_density="32k",
                    method="linear"
                )
                surf[0].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "L"))
                surf[1].to_filename(nispace_source_data_path / "reference" / dataset / "map" / m / (fn_save % "R"))
            
            
# %%
