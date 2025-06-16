# %% Init

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from neuromaps import images
from neuromaps.resampling import resample_images
from sklearn.preprocessing import minmax_scale

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import reference_lib
from nispace.io import parcellate_data

# nispace data path 
nispace_source_data_path = wd

# datasets with maps
DSETS_WITH_MAPS = [k for k, v in reference_lib.items() if "map" in v]
print("DSETS_WITH_MAPS: ", DSETS_WITH_MAPS)

# all parcellations 
PARCS = sorted(
    [p.name for p in (nispace_source_data_path / "parcellation").glob("*") if p.is_dir()]
)
print("PARCS:", PARCS)

# function to fetch reference maps
def fetch_reference(dataset, maps, space):
    
    map_paths = []
    map_dir = nispace_source_data_path / "reference" / dataset / "map"
    for m in maps:
        fp = sorted((map_dir / m).glob(f"{m}_space-{space}*"))
        if len(fp) == 0:
            raise ValueError(f"No map found for {m} in {space}")
        elif len(fp) == 1:
            fp = fp[0]
        elif len(fp) == 2:
            fp = tuple(fp)
        elif len(fp) > 2:
            raise ValueError(f"Over two maps found for {m} in {space}: {fp}")
        map_paths.append(fp)
    return map_paths
        

# %% Parcellate map-based image data ---------------------------------------------------------------

# iterate datasets
for dataset in DSETS_WITH_MAPS:
    print("-------- " + dataset.upper() + " --------")
    
    # get files: 
    ref_maps = {}
    # here, we iterate original spaces, not the processed ones
    for space in ["MNI152", "MNI152NLin6Asym", "MNI152NLin2009cAsym", 
                  "fsaverageOriginal", "fsaverage", 
                  "fsLROriginal", "fsLR"]:
        ref_maps[space] = {}
        
        print("Checking space:", space)
        ref_maps[space] = []
        for m in reference_lib[dataset]["map"].keys():
            if space in reference_lib[dataset]["map"][m]:
                
                # skip private
                private = False
                if "host" in reference_lib[dataset]["map"][m][space]:
                    if "private" in reference_lib[dataset]["map"][m][space]["host"]: private = True
                elif "L" in reference_lib[dataset]["map"][m][space]:
                    if "private" in reference_lib[dataset]["map"][m][space]["L"]["host"]: private = True
                elif "R" in reference_lib[dataset]["map"][m][space]:
                    if "private" in reference_lib[dataset]["map"][m][space]["R"]["host"]: private = True
                
                # append if not private
                if not private:
                    ref_maps[space].append(m)
        
        if len(ref_maps[space]) == 0:
            ref_maps.pop(space)
            print("No maps found for space:", space) 
        else:
            print(f"{len(ref_maps[space])} maps")
            
    # check all available maps across spaces
    ref_maps_avail_all = set()
    for space in ref_maps.keys():
        ref_maps_avail_all.update(ref_maps[space])
    print(f"-> {len(ref_maps_avail_all)} unique maps")
    # restore order from reference_lib
    ref_maps_avail_all = [m for m in reference_lib[dataset]["map"] if m in ref_maps_avail_all]

    # iterate parcellations:
    print("Parcellating...")
    for parc_name in PARCS: 
        print(parc_name)
        
        # iterate spaces: 
        # TODO: THIS IS A MESS: NOT NECESSARY WHEN WE PROVIDE ALL SPACES FOR ALL REFS/PARCS
        
        # dataset: pet
        if dataset == "pet":
            # parcellations available in all spaces: fsaverage > MNI152NLin6Asym
            if parc_name in ["Schaefer100", "Schaefer200", "Schaefer400", "DesikanKilliany", "Destrieux"]:
                ref_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
                parc_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
                ref_maps_to_iterate = [ref_maps["MNI152NLin6Asym"], ref_maps["fsaverageOriginal"]]
            # parcellations only available in fsLR
            elif parc_name in ["Glasser"]:
                ref_spaces_to_iterate = ["fsLR", "fsaverage"]
                parc_spaces_to_iterate = ["fsLR", "fsaverage"]
                ref_maps_to_iterate = [ref_maps["fsLR"], ref_maps["fsaverageOriginal"]]
            # parcellations only available in MNI spaces
            elif parc_name in ["DesikanKillianyTourville", "TianS1", "TianS2", "TianS3", "Aseg"]:
                ref_spaces_to_iterate = ["MNI152NLin6Asym"]
                parc_spaces_to_iterate = ["MNI152NLin6Asym"]
                ref_maps_to_iterate = [ref_maps["MNI152NLin6Asym"]]
            # the rest
            else:
                raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
                
        # dataset: rsn
        # RSN dataset has only cortical MNI152 space for now and fits better to MNI152NLin2009cAsym 
        elif dataset == "rsn":
            # parcellations availabe in MNI152NLin2009cAsym
            if parc_name in ["Schaefer100", "Schaefer200", "Schaefer400", "DesikanKilliany", "Destrieux", "DesikanKillianyTourville"]:
                ref_spaces_to_iterate = ["MNI152"]
                parc_spaces_to_iterate = ["MNI152NLin2009cAsym"]
                ref_maps_to_iterate = [ref_maps["MNI152"]]
            # subcortical parcellations
            elif parc_name in ["TianS1", "TianS2", "TianS3", "Aseg"]:
                continue
            # cortical parcellations but not available in MNI152NLin2009cAsym
            elif parc_name in ["Glasser"]:
                continue
            # the rest
            else:
                raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
            
        # dataset: cortexfeatures
        elif dataset == "cortexfeatures":
            # cortical parcellations available in surface spaces:
            if parc_name in ["Schaefer100", "Schaefer200", "Schaefer400", "DesikanKilliany", "Destrieux", "Glasser"]:
                ref_spaces_to_iterate = ["fsLR", "fsaverage"]
                parc_spaces_to_iterate = ["fsLR", "fsaverage"]
                ref_maps_to_iterate = [ref_maps["fsLROriginal"], ref_maps["fsaverageOriginal"]]
            # subcortical parcellations
            elif parc_name in ["TianS1", "TianS2", "TianS3", "Aseg"]:
                continue
            # cortical parcellations but not available in fsLR
            elif parc_name in ["DesikanKillianyTourville"]:
                continue
            # the rest
            else:
                raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
            
        # dataset: bigbrain
        elif dataset == "bigbrain":
            # cortical parcellations available in surface spaces:
            if parc_name in ["Schaefer100", "Schaefer200", "Schaefer400", "DesikanKilliany", "Destrieux", "Glasser"]:
                ref_spaces_to_iterate = ["fsaverage"]
                parc_spaces_to_iterate = ["fsaverage"]
                ref_maps_to_iterate = [ref_maps["fsaverageOriginal"]]
            # subcortical parcellations
            elif parc_name in ["TianS1", "TianS2", "TianS3", "Aseg"]:
                continue
            # cortical parcellations but not available in fsaverage
            elif parc_name in ["DesikanKillianyTourville"]:
                continue
            # the rest
            else:
                raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
            
        # parcel labels
        try:
            labels = np.loadtxt(nispace_source_data_path / "parcellation" / parc_name / "MNI152NLin6Asym" / 
                                f"parc-{parc_name}_space-MNI152NLin6Asym.label.txt", dtype=str)
        except:
            labels = np.concatenate([
                np.loadtxt(nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
                           f"parc-{parc_name}_space-fsLR_hemi-L.label.txt", dtype=str),
                np.loadtxt(nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
                           f"parc-{parc_name}_space-fsLR_hemi-R.label.txt", dtype=str)
            ])
        
        # Initiate dataframe
        ref_maps_df = pd.DataFrame(
            index=pd.Index(list(ref_maps_avail_all), name="map"),
            columns=labels,
        )
        print("Pre-parcellation dataframe shape: ", ref_maps_df.shape)
            
        # run parcellation
        for ref_space, parc_space, ref_maps_avail in zip(ref_spaces_to_iterate, parc_spaces_to_iterate, ref_maps_to_iterate):
                     
            # Load the parcellation
            if "mni" in parc_space.lower():
                parc = images.load_nifti(
                    nispace_source_data_path / "parcellation" / parc_name / parc_space / 
                    f"parc-{parc_name}_space-{parc_space}.label.nii.gz"
                )
            else:
                parc = (
                    images.load_gifti(nispace_source_data_path / "parcellation" / parc_name / parc_space / 
                                      f"parc-{parc_name}_space-{parc_space}_hemi-L.label.gii.gz"),
                    images.load_gifti(nispace_source_data_path / "parcellation" / parc_name / parc_space / 
                                      f"parc-{parc_name}_space-{parc_space}_hemi-R.label.gii.gz")
                )
            print(f"Parcellation loaded for space {parc_space}")
            
            # Load the reference maps
            # local fetch_reference function; works only if all maps are available in the source_data folder
            ref_paths = fetch_reference(dataset, maps=ref_maps_avail, space=ref_space)
            print(f"-> {len(ref_paths)} maps available for space {ref_space}")
            
            # parcellate
            # both hemispheres
            if ("MNI152" in parc_space) or ("MNI152" not in parc_space and all([len(fp)==2 for fp in ref_paths])):
                parc = [parc]
                parc_labels = [labels]
                parc_hemi = [None]
                ref_paths = [ref_paths]
                ref_maps_avail = [ref_maps_avail]
            # mix of both hemispheres and single hemisphere
            else:
                parc = [parc, parc[0], parc[1]]
                parc_labels = [labels, [l for l in labels if "hemi-L" in l], [l for l in labels if "hemi-R" in l]]
                parc_hemi = [None, "L", "R"]
                ref_paths = [
                    [fp for fp in ref_paths if len(fp)==2], 
                    [fp for fp in ref_paths if len(fp)==1 and "hemi-L" in fp[0].name],
                    [fp for fp in ref_paths if len(fp)==1 and "hemi-R" in fp[0].name]
                ]
                ref_maps_avail = [
                    [fp[0].parent.name for fp in ref_paths[0]], 
                    [fp[0].parent.name for fp in ref_paths[1]],
                    [fp[0].parent.name for fp in ref_paths[2]]
                ]
                
            for p, p_l, p_h, r_p, r_m_a in zip(parc, parc_labels, parc_hemi, ref_paths, ref_maps_avail):
                if len(r_p) == 0:
                    continue
                
                tab = parcellate_data(
                    parcellation=p,
                    parc_hemi=p_h,
                    parc_labels=p_l,
                    parc_space=parc_space,
                    data=r_p,
                    data_labels=r_m_a,
                    data_space=ref_space,
                    n_proc=-1,
                    dtype=np.float32,
                    drop_background_parcels=True,
                    min_num_valid_datapoints=5,
                    min_fraction_valid_datapoints=0.3,
                )
                
                # save into dataframe
                print(f"Parcellated data of shape: {tab.shape}, while dataframe has shape: {ref_maps_df.shape}")
                ref_maps_df.loc[r_m_a, tab.columns] = tab
        
        # save
        #ref_maps_df = ref_maps_df.sort_index()
        print(ref_maps_df.head())
        save_dir = nispace_source_data_path / "reference" / dataset / "tab"
        save_dir.mkdir(parents=True, exist_ok=True)
        ref_maps_df.to_csv(save_dir / f"dset-{dataset}_parc-{parc_name}.csv.gz")
        print("-"*100)
        print("\n")



# %%
