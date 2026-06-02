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

# %% We only need to calculate glycolytic index for now
# "The GI is obtained by conventional linear regression of CMRGlu on CMRO2 and exhibiting the 
# residuals scaled by 1000.""
# https://www.pnas.org/doi/10.1073/pnas.1010459107

cmrglc = fetch_reference("cortexfeatures", space="fsLROriginal", maps="cmrglc")[0]
cmro2 = fetch_reference("cortexfeatures", space="fsLROriginal", maps="cmro2")[0]

# load cmrglc and cmro2
cmrglc = load_img(cmrglc)
cmro2 = load_img(cmro2)

# shape of one hemisphere dataset
n_vertices = cmrglc[0].agg_data().shape[0]
print("One hemisphere dataset shape: ", n_vertices)

# one vector for each
cmrglc_agg = np.concatenate([cmrglc[0].agg_data(), cmrglc[1].agg_data()])
cmro2_agg = np.concatenate([cmro2[0].agg_data(), cmro2[1].agg_data()])
print("Concatenated shape: ", cmrglc_agg.shape)

# calculate GI
gi_agg = residuals(y=cmrglc_agg, x=cmro2_agg) * 1000

# separate in hemispheres
gi_agg_L = gi_agg[:n_vertices]
gi_agg_R = gi_agg[n_vertices:]
assert gi_agg_L.shape == gi_agg_R.shape

# new giftis
gi_agg_L_img = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data=gi_agg_L)])
gi_agg_R_img = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data=gi_agg_R)])

# save giftis
save_dir = nispace_source_data_path / "reference" / "cortexfeatures"/ "map" / "feature-glycindex_pub-vaishnavi2010" 
save_dir.mkdir(parents=True, exist_ok=True)
gi_agg_L_img.to_filename(save_dir / "feature-glycindex_pub-vaishnavi2010_space-fsLROriginal_hemi-L.surf.gii.gz")
gi_agg_R_img.to_filename(save_dir / "feature-glycindex_pub-vaishnavi2010_space-fsLROriginal_hemi-R.surf.gii.gz")


# %% Convert to NiSpace-format surface maps

for m in reference_lib["cortexfeatures"]["map"]:
    print("Processing map:", m)

    # get transforms
    if "fsaverageOriginal" in reference_lib["cortexfeatures"]["map"][m]:
        source_space = "fsaverageOriginal"
        transform_to_fslr = transforms.fsaverage_to_fslr
        transform_to_fsaverage = transforms.fsaverage_to_fsaverage
    elif "fsLROriginal" in reference_lib["cortexfeatures"]["map"][m]:
        source_space = "fsLROriginal"
        transform_to_fslr = transforms.fslr_to_fslr
        transform_to_fsaverage = transforms.fslr_to_fsaverage
    else:
        raise ValueError(f"Neither fsaverageOriginal nor fsLROriginal found for map {m}")
    
    # get original map
    fp = fetch_reference("cortexfeatures", maps=m, space=source_space, verbose=False)
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
        
        # hemispheres
        hemi = ["L", "R"] if len(fp) == 2 else [fp[0].name.split("hemi-")[1].split(".")[0]]
        
        # transform
        map_target = transform_to_target(
            data=load_img(fp),
            target_density=target_density,
            method="linear",
            hemi=hemi
        )
        
        # save
        save_dir = nispace_source_data_path / "reference" / "cortexfeatures"/ "map" / m
        save_dir.mkdir(parents=True, exist_ok=True)
        for i_h, h in enumerate(hemi):
            map_target[i_h].to_filename(save_dir / f"{m}_space-{target_space}_desc-proc_hemi-{h}.surf.gii.gz")
            print(f"Saved {m} to {target_space} {h}...")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "cortexfeatures"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])

pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

pd.Series([
    "feature-megpoweralpha_pub-shafiei2022",
    "feature-megpowerbeta_pub-shafiei2022",
    "feature-megpowerdelta_pub-shafiei2022",
    "feature-megpowergamma1_pub-shafiei2022",
    "feature-megpowergamma2_pub-shafiei2022",
    "feature-megpowertheta_pub-shafiei2022",
    "feature-megtimescale_pub-shafiei2022",
], name="map").to_csv(ref_dir / "collection-MEG.collect", index=False)

pd.Series([
    "feature-cbf_pub-vaishnavi2010",
    "feature-cbv_pub-vaishnavi2010",
    "feature-cmro2_pub-vaishnavi2010",
    "feature-cmrglc_pub-vaishnavi2010",
    "feature-glycindex_pub-vaishnavi2010",
], name="map").to_csv(ref_dir / "collection-Metabolism.collect", index=False)

pd.Series([
    "feature-thickness_pub-hcps1200",
    "feature-t1t2_pub-hcps1200",
    "feature-saaxis_pub-sydnor2021",
    "feature-geneexpr-abagen",
    "feature-develexpansion_pub-hill2010",
    "feature-evolexpansion_pub-hill2010",
    "feature-evolexpansion_pub-xu2020",
    "feature-specieshomology_pub-xu2020",
], name="map").to_csv(ref_dir / "collection-CortexOrganisation.collect", index=False)


# %% Parcellate ------------------------------------------------------------------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import parcellate_mapref

parcellate_mapref(wd, "cortexfeatures", spaces=[
    "fsLROriginal",       # fsLR-native maps
    "fsaverageOriginal",  # fsaverage-native maps (no overlap → no overwrite)
])

# %%
