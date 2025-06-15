# %% Init

import sys
from pathlib import Path
import numpy as np
from neuromaps import transforms
import nibabel as nib

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference, fetch_parcellation, reference_lib, parcellation_lib
from nispace.io import parcellate_data, load_img
from nispace.utils.utils_datasets import get_file
from nispace.stats.misc import residuals 

# nispace data path 
nispace_source_data_path = wd

# %% Convert to NiSpace-format surface maps

for m in reference_lib["bigbrain"]["map"]:
    print("Processing map:", m)

    # get transforms
    if "fsaverageOriginal" in reference_lib["bigbrain"]["map"][m]:
        source_space = "fsaverageOriginal"
        transform_to_fslr = transforms.fsaverage_to_fslr
        transform_to_fsaverage = transforms.fsaverage_to_fsaverage
    elif "fsLROriginal" in reference_lib["bigbrain"]["map"][m]:
        source_space = "fsLROriginal"
        transform_to_fslr = transforms.fslr_to_fslr
        transform_to_fsaverage = transforms.fslr_to_fsaverage
    else:
        raise ValueError(f"Neither fsaverageOriginal nor fsLROriginal found for map {m}")

    # get original map
    fp = fetch_reference("bigbrain", maps=m, space=source_space, verbose=False)
    if len(fp) == 0 or len(fp) > 1:
        raise ValueError(f"No or multiple files found for {m} in fsLR: {fp}")
    fp = fp[0]
    if len(fp) == 1:
        print(f"Only one hemisphere found for {m}: {fp}")
    if len(fp) > 2:
        raise ValueError(f"More than two files found for {m} in fsLR: {fp}")
    
    # transform to fsLR and fsaverage
    for target_space, transform_to_target, target_density in [
        ("fsLR", transform_to_fslr, "32k"), 
        ("fsaverage", transform_to_fsaverage, "41k")
    ]:
        print(f"Transforming {m} to {target_space}...")
        
        # transform
        map_target = transform_to_target(
            data=load_img(fp),
            target_density=target_density,
            method="linear",
        )

        # save
        save_dir = nispace_source_data_path / "reference" / "bigbrain"/ "map" / m
        save_dir.mkdir(parents=True, exist_ok=True)
        ext = "shape.gii.gz" if not "func" in fp[0].name else "func.gii.gz"
        for i_h, h in enumerate(["L", "R"]):
            map_target[i_h].to_filename(save_dir / f"{m}_space-{target_space}_desc-proc_hemi-{h}.{ext}")
            print(f"Saved {m} to {target_space} {h}...")
        
# %% For checking, plot:
from nispace.datasets import fetch_template
from neuromaps.datasets import fetch_fslr
from nilearn.plotting import plot_surf_stat_map
import matplotlib.pyplot as plt

fsaverage = fetch_template("fsaverage", res="41k")[0]
fslr = fetch_fslr("32k")["midthickness"][0]

for m in reference_lib["bigbrain"]["map"]:
    print("Plotting map:", m)
    
    # fslr
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), subplot_kw={"projection": "3d"})
    for i, view in enumerate(["lateral", "medial"]):
        plot_surf_stat_map(
            fslr,
            wd / "reference" / "bigbrain" / "map" / m / f"{m}_space-fsLR_desc-proc_hemi-L.surf.gii.gz",
            title=f"{m}: fsLR",
            cmap="viridis",
            view=view,
            axes=axes[i],
        )
    plt.show()
    
    # fsaverage
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), subplot_kw={"projection": "3d"})
    for i, view in enumerate(["lateral", "medial"]):
        plot_surf_stat_map(
            fsaverage,
            wd / "reference" / "bigbrain" / "map" / m / f"{m}_space-fsaverage_desc-proc_hemi-L.surf.gii.gz",
            title=f"{m}: fsaverage",
            cmap="viridis",
            view=view,
            axes=axes[i],
        )
    plt.show()
    
    

    
    



# %%
