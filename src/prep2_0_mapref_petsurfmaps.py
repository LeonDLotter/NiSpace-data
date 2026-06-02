# %% Init

import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import nibabel as nib
from neuromaps import images
from neuromaps.resampling import resample_images
from neuromaps import transforms
from sklearn.preprocessing import minmax_scale

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.datasets import fetch_reference, reference_lib
from nispace.io import load_img, write_json

# nispace data path 
nispace_source_data_path = wd


# %% Convert to surface maps
dataset = "pet"    

# iterate maps
for m in reference_lib[dataset]["map"]:
    transform_to_fsaverage, transform_to_fslr = None, None
    if "private" in reference_lib[dataset]["map"][m]["MNIOriginal"]["host"]:
        print(f"Skipping private map: {m}")
        continue
    
    print("Processing map:", m)
    
    # Cases:
    # original fsaverage exists
    if "fsaverageOriginal" in reference_lib[dataset]["map"][m]:
        # TODO: intuitively, source_space should be fsaverageOriginal, but it is NOT because we processed
        # the data in another script. So, source_space will be "fsaverage" and no fsaverage_to_fsaverage 
        # transform will be applied. CHANGE THIS BY IMPLEMENTING ALL PROCESSING IN ONE SCRIPT
        # TODO: get the rescaling from external script to apply it here
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
        fp = fetch_reference(dataset, m, space=source_space, verbose=False, check_file_hash=False)
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


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "pet"
maps = [m for m in reference_lib["pet"]["map"]
        if "private" not in reference_lib["pet"]["map"][m].get("MNIOriginal", {}).get("host", "")]

pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

pd.DataFrame({
    "set":    [m.split("_")[0].split("-")[1] for m in maps],
    "x":      maps,
    "weight": [m.split("_")[2].split("-")[1] for m in maps],
}).to_csv(ref_dir / "collection-AllTargetSets.collect", index=False)

write_json(
    {
        "General": [
            "target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023",
            "target-rCPS_tracer-leucine_n-42_dx-hc_pub-smith2023",
            "target-SV2A_tracer-ucbj_n-76_dx-hc_pub-finnema2016",
            "target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016",
            "target-VMAT2_tracer-dtbz_n-76_dx-hc_pub-larsen2020",
        ],
        "Immunity": [
            "target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018",
            "target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020",
        ],
        "Glutamate": [
            "target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019",
            "target-NMDA_tracer-ge179_n-29_dx-hc_pub-galovic2021",
        ],
        "GABA": [
            "target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022",
            "target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018",
        ],
        "Dopamine": [
            "target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018",
            "target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017",
            "target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015",
            "target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018",
        ],
        "Serotonin": [
            "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012",
            "target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012",
            "target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012",
            "target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017",
            "target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018",
            "target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012",
        ],
        "Noradrenaline/Acetylcholine": [
            "target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017",
            "target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016",
            "target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020",
            "target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017",
        ],
        "Opioids/Endocannabinoids": [
            "target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020",
            "target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018",
            "target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015",
        ],
        "Histamine": [
            "target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017",
        ],
    },
    ref_dir / "collection-UniqueTracers.collect",
)

unique_tracer_sets = [
    "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012",
    "target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012",
    "target-5HT1b_tracer-p943_n-65_dx-hc_pub-gallezot2010",
    "target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012",
    "target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017",
    "target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018",
    "target-5HTT_tracer-dasb_n-100_dx-hc_pub-beliveau2017",
    "target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012",
    "target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016",
    "target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015",
    "target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023",
    "target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020",
    "target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017",
    "target-D23_tracer-flb457_n-37_dx-hc_pub-smith2017",
    "target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015",
    "target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018",
    "target-DAT_tracer-fpcit_n-30_dx-hc_pub-garciagomez2013",
    "target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018",
    "target-GABAa_tracer-flumazenil_n-16_dx-hc_pub-norgaard2021",
    "target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018",
    "target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022",
    "target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017",
    "target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016",
    "target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018",
    "target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020",
    "target-mGluR5_tracer-abp688_n-22_dx-hc_pub-rosaneto",
    "target-mGluR5_tracer-abp688_n-28_dx-hc_pub-dubois2015",
    "target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019",
    "target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020",
    "target-MOR_tracer-carfentanil_n-39_dx-hc_pub-turtonen2021",
    "target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017",
    "target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010",
    "target-rCPS_tracer-leucine_n-42_dx-hc_pub-smith2023",
    "target-SV2A_tracer-ucbj_n-76_dx-hc_pub-finnema2016",
    "target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018",
    "target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017",
    "target-VAChT_tracer-feobv_n-4_dx-hc_pub-tuominen",
    "target-VAChT_tracer-feobv_n-5_dx-hc_pub-bedard2019",
    "target-VMAT2_tracer-dtbz_n-76_dx-hc_pub-larsen2020",
]
pd.DataFrame({
    "set":    [m.split("_")[0].split("-")[1] for m in unique_tracer_sets],
    "x":      unique_tracer_sets,
    "weight": [m.split("_")[2].split("-")[1] for m in unique_tracer_sets],
}).to_csv(ref_dir / "collection-UniqueTracerSets.collect", index=False)

# %%
