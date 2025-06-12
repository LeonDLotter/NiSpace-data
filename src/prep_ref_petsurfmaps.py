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
from nispace.io import parcellate_data, load_img
from nispace.utils.utils_datasets import get_file

# nispace data path 
nispace_source_data_path = wd


# %% Convert to surface maps
dataset = "pet"    

# iterate maps
for m in reference_lib[dataset]["map"]:
    transform_to_fsaverage, transform_to_fslr = None, None
    if "private" in reference_lib[dataset]["map"][m]["MNI152"]["host"]:
        print(f"Skipping private map: {m}")
        continue
    
    print("Processing map:", m)
    
    # Cases:
    # original fsaverage exists
    if "fsaverageOriginal" in reference_lib[dataset]["map"][m]:
        # TODO: intuitively, source_space should be fsaverageOriginal, but it is NOT because we processed
        # the data in another script. So, source_space will be "fsaverage" and no fsaverage_to_fsaverage 
        # transform will be applied. CHANGE THIS BY IMPLEMENTING ALL PROCESSING IN ONE SCRIPT
        source_space = "fsaverage"
        transform_to_fsaverage = None
        transform_to_fslr = transforms.fsaverage_to_fslr
    # original fsLR exists
    elif "fsLROriginal" in reference_lib[dataset]["map"][m]:
        # TODO: See above, replace fsaverageOriginal with fsLROriginal
        source_space = "fsLR"
        transform_to_fsaverage = transforms.fslr_to_fsaverage
        transform_to_fslr = None
    # only MNI152NLin6Asym exists
    elif "MNI152NLin6Asym" in reference_lib[dataset]["map"][m]:
        source_space = "MNI152NLin6Asym"
        transform_to_fsaverage = transforms.mni152_to_fsaverage
        transform_to_fslr = transforms.mni152_to_fslr
    else:
        raise ValueError(f"Something is wrong with the map: {m}")
    
    # apply transform
    for target_space, transform_to_target, target_density in [
        ("fsLR", transform_to_fslr, "32k"), 
        ("fsaverage", transform_to_fsaverage, "41k")
    ]:
        print(f"Transforming {m} from {source_space} to {target_space}...")
        
        # get original map and check
        fp = fetch_reference(dataset, m, space=source_space, verbose=False)
        if len(fp) == 0 or len(fp) > 1:
            raise ValueError(f"No or multiple files found for {m} in {source_space}")
        fp = fp[0]
        if source_space == "MNI152NLin6Asym" and isinstance(fp, Path):
            pass
        elif len(fp) == 2:
            hemi = ["L", "R"]
        elif len(fp) == 1:
            raise ValueError(f"Only one hemisphere found for {m}: {fp}")
            #hemi = [fp[0].name.split("hemi-")[1].split(".")[0]]
        else:
            raise ValueError(f"Something is wrong with the map: {m}")
        
        # transform
        if transform_to_target is not None:
            map_target = transform_to_target(
                load_img(fp),
                target_density,
                method="linear",
                #**{"hemi": hemi} if source_space != "MNI152NLin6Asym" else {}
            )
            
            # save
            save_dir = nispace_source_data_path / "reference" / dataset / "map" / m
            save_dir.mkdir(parents=True, exist_ok=True)
            for i_h, h in enumerate(hemi):
                map_target[i_h].to_filename(save_dir / f"{m}_space-{target_space}_desc-proc_hemi-{h}.surf.gii.gz")
                print(f"Saved {m} to {target_space} {h}...")
        else:
            print(f"No transform to {target_space} for {m}")
                
# %%
