# %% Init

from pathlib import Path
import numpy as np
import pandas as pd
from neuromaps import transforms
import nibabel as nib

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.datasets import fetch_reference, reference_lib
from nispace.io import load_img
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
# from nispace.plotting import brainplot
# import matplotlib.pyplot as plt

# for m in reference_lib["bigbrain"]["map"]:
#     for space in ["fsLR", "fsaverage"]:
#         brainplot(
#             fetch_reference("bigbrain", maps=m, space=space, verbose=False)[0],
#             space=space,
#             title=f"{m} | {space}",
#         )
#         plt.show()


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "bigbrain"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])

pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

pd.Series([
    "feature-layer1_pub-wagstyl2020",
    "feature-layer2_pub-wagstyl2020",
    "feature-layer3_pub-wagstyl2020",
    "feature-layer4_pub-wagstyl2020",
    "feature-layer5_pub-wagstyl2020",
    "feature-layer6_pub-wagstyl2020",
], name="map").to_csv(ref_dir / "collection-CorticalLayers.collect", index=False)

pd.Series([
    "feature-histogradient1_pub-paquola2021",
    "feature-histogradient2_pub-paquola2021",
    "feature-microgradient1_pub-paquola2021",
    "feature-microgradient2_pub-paquola2021",
    "feature-funcgradient1_pub-paquola2021",
    "feature-funcgradient2_pub-paquola2021",
    "feature-funcgradient3_pub-paquola2021",
], name="map").to_csv(ref_dir / "collection-DifferentiationGradients.collect", index=False)

# %%
