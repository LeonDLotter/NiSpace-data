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
from sklearn.preprocessing import minmax_scale

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference, fetch_parcellation, reference_lib
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


# %% Parcellate map-based image data ---------------------------------------------------------------

# iterate datasets
for dataset in DSETS_WITH_MAPS:
    print("-------- " + dataset.upper() + " --------")
    
    # get files: 
    ref_maps = {}
    # here, we iterate original spaces, not the processed ones
    for space in ["MNI152", "MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsa", "hcp"]:
        ref_maps[space] = {}
        
        print("Checking space:", space)
        ref_maps[space] = [
            m for m in reference_lib[dataset]["map"].keys()
            if space in reference_lib[dataset]["map"][m]
            and "private" not in (reference_lib[dataset]["map"][m][space]["host"] if "mni" in space.lower() 
                                    else reference_lib[dataset]["map"][m][space]["L"]["host"])
        ]
        if len(ref_maps[space]) == 0:
            ref_maps.pop(space)
            print("No maps found for space:", space)
        else:
            print(f"{len(ref_maps[space])} maps")
            
    # all available maps across spaces
    ref_maps_avail_all = set()
    for space in ref_maps.keys():
        ref_maps_avail_all.update(ref_maps[space])
    print(f"-> {len(ref_maps_avail_all)} unique maps")
    
    # replace space keys -> fsa to fsaverage, hcp to fsLR
    ref_maps = {
        k.replace("fsa", "fsaverage").replace("hcp", "fsLR"): v
        for k, v in ref_maps.items()
    }

    # iterate parcellations:
    print("Parcellating...")
    for parc_name in PARCS: 
        print(parc_name)
        
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
        
        # iterate spaces: 
        # TODO: THIS IS A MESS: NOT NECESSARY WHEN WE PROVIDE ALL SPACES FOR ALL REFS/PARCS
        # for PET data and Schaefer parcellations: fsaverage > MNI152NLin6Asym
        if dataset == "pet" and parc_name in ["Schaefer100", "Schaefer200", "Schaefer400", 
                                              "Glasser",
                                              "DesikanKilliany", "Destrieux"]:
            ref_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
            parc_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
        # parcellations only available in MNI
        elif dataset == "pet" and parc_name in ["DesikanKillianyTourville", "TianS1", "TianS2", "TianS3", "Aseg"]:
            ref_spaces_to_iterate = ["MNI152NLin6Asym"]
            parc_spaces_to_iterate = ["MNI152NLin6Asym"]
        # RSN dataset has only MNI152 space for now and fits better to MNI152NLin2009cAsym 
        elif dataset == "rsn" and parc_name != "Glasser":
            ref_spaces_to_iterate = ["MNI152"]
            parc_spaces_to_iterate = ["MNI152NLin2009cAsym"]
        # RSN only on cortex and in MNI
        elif dataset == "rsn" and parc_name in ["Glasser", "TianS1", "TianS2", "TianS3", "Aseg"]:
            continue
        else:
            raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
            
        # run parcellation
        for ref_space, parc_space in zip(ref_spaces_to_iterate, parc_spaces_to_iterate):
            
            # only if Glasser, load all fsLR transformed versions instead of MNI
            # TODO: terrible, change that
            ref_maps_avail = ref_maps[ref_space]
            if parc_name == "Glasser" and ref_space == "MNI152NLin6Asym":
                print("Glasser: loading fsLR versions instead of MNI")
                parc_space = "fsLR"
                ref_space = "fsLR"
                
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
            ref_paths = fetch_reference(dataset, maps=ref_maps_avail, space=ref_space, print_references=False)
            print(f"-> {len(ref_paths)} maps available for space {ref_space}")
            
            # parcellate
            tab = parcellate_data(
                parcellation=parc,
                parc_labels=labels,
                parc_space=parc_space,
                data=ref_paths,
                data_labels=ref_maps_avail,
                data_space=ref_space,
                n_proc=-1,
                dtype=np.float32,
                drop_background_parcels=True,
                min_num_valid_datapoints=5,
                min_fraction_valid_datapoints=0.3,
            )
            
            # save into dataframe
            print(f"Parcellated data of shape: {tab.shape}, while dataframe has shape: {ref_maps_df.shape}")
            ref_maps_df.loc[ref_maps_avail, tab.columns] = tab
        
        # save
        ref_maps_df = ref_maps_df.sort_index()
        ref_maps_df.to_csv(nispace_source_data_path / "reference" / dataset / "tab" / f"dset-{dataset}_parc-{parc_name}.csv.gz")



# %%
