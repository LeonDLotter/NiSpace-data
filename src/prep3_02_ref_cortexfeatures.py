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


# %% HCP S1200 group-average structural maps
# Source: "Human HCP 1200 Group Average + Individuals; Structural + fMRI Atlas" on BALSA
# (HCP_S1200_GroupAvg_BALSA.scene, ID: D4w0w; login required), stored in _archive/hcps1200.
# 32k fsLR CIFTI dscalars (59412 cortical vertices) -> per-hemisphere fsLR GIFTIs (medial wall = 0,
# like the neuromaps maps) -> fsaverage 41k. Sulc/curvature keep the HCP sign (negative = sulcal).

hcp_dir = wd / "_archive" / "hcps1200"
HCP_MAPS = {
    "feature-thickness_pub-hcps1200": "S1200.thickness_MSMAll.32k_fs_LR.dscalar.nii",
    "feature-t1t2_pub-hcps1200": "S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii",
    "feature-sulc_pub-hcps1200": "S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii",
    "feature-curv_pub-hcps1200": "S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii",
}
CIFTI_HEMIS = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}

for m, fn in HCP_MAPS.items():
    print("Processing HCP map:", m)
    cifti = nib.load(hcp_dir / fn)
    data = cifti.get_fdata(dtype=np.float32).ravel()

    # CIFTI -> fsLR 32k GIFTIs
    map_fslr = {}
    for name, slc, model in cifti.header.get_axis(1).iter_structures():
        if name not in CIFTI_HEMIS:
            continue
        arr = np.zeros(model.nvertices[name], dtype=np.float32)
        arr[model.vertex] = data[slc]
        map_fslr[CIFTI_HEMIS[name]] = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(data=arr)])
    assert list(map_fslr) == ["L", "R"]

    # fsLR -> fsaverage 41k
    map_fsa = transforms.fslr_to_fsaverage(
        data=(map_fslr["L"], map_fslr["R"]),
        target_density="41k",
        method="linear",
    )

    # save
    save_dir = nispace_source_data_path / "reference" / "cortexfeatures" / "map" / m
    save_dir.mkdir(parents=True, exist_ok=True)
    for i_h, h in enumerate(["L", "R"]):
        map_fslr[h].to_filename(save_dir / f"{m}_space-fsLR_desc-proc_hemi-{h}.surf.gii.gz")
        map_fsa[i_h].to_filename(save_dir / f"{m}_space-fsaverage_desc-proc_hemi-{h}.surf.gii.gz")
    print(f"Saved {m} to fsLR and fsaverage")


# %% Convert neuromaps maps to NiSpace-format surface maps

for m in reference_lib["cortexfeatures"]["map"]:
    if m in HCP_MAPS:
        continue
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
    "feature-sulc_pub-hcps1200",
    "feature-curv_pub-hcps1200",
    "feature-saaxis_pub-sydnor2021",
    "feature-geneexpr_pub-markello2021",
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
    "fsLR",               # all maps (HCP maps have no Original space)
    "fsaverageOriginal",  # fsaverage-native maps (overwrites their fsLR pass)
])

# %%
