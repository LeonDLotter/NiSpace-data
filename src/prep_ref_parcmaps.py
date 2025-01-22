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
from nispace.datasets import fetch_reference, fetch_parcellation, reference_lib, parcellation_lib
from nispace.io import parcellate_data

# nispace data path 
nispace_source_data_path = wd

# datasets with maps
DSETS_WITH_MAPS = [k for k, v in reference_lib.items() if "map" in v]
print("DSETS_WITH_MAPS: ", DSETS_WITH_MAPS)

# all parcellations (original, not aliases)
PARCS = [k for k in parcellation_lib.keys() if "alias" not in parcellation_lib[k]]
print("PARCS:", PARCS)


# %% Parcellate map-based image data ---------------------------------------------------------------

# iterate datasets
for dataset in DSETS_WITH_MAPS:
    print("-------- " + dataset.upper() + " --------")
    
    # get files: 
    ref_maps = {}
    for space in ["MNI152", "MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsaverage", "fsLR"]:
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
    
    # iterate parcellations:
    print("Parcellating...")
    for parc_name in PARCS:
        print(parc_name)
        
        # Initiate dataframe
        ref_maps_df = pd.DataFrame(
            index=pd.Index(list(ref_maps_avail_all), name="map"),
            columns=fetch_parcellation(
                parc_name, 
                space="MNI152NLin2009cAsym" if parc_name not in ["DesikanKilliany", "Destrieux"] else "fsaverage", 
                return_loaded=True
            )[1],
        )
        print("Pre-parcellation dataframe shape: ", ref_maps_df.shape)
        
        # iterate spaces: 
        # TODO: THIS IS A MESS: NOT NECESSARY WHEN WE PROVIDE ALL SPACES FOR ALL REFS/PARCS
        # Desikan and Destrieux parcellations only available in fsaverage, transform function expects MNI152NLin6Asym
        if dataset == "pet" and parc_name in ["DesikanKilliany", "Destrieux"]:
            ref_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
            parc_spaces_to_iterate = ["fsaverage", "fsaverage"]
        # RSN dataset has only MNI152 space for now
        # TODO: REGISTER RSN DATASET (OR WAIT 'TILL AVAILABLE VIA MIDB )
        elif dataset == "rsn" and parc_name in ["DesikanKilliany", "Destrieux"]:
            ref_spaces_to_iterate = ["MNI152"]
            parc_spaces_to_iterate = ["fsaverage"]
        # RSN dataset fits better to MNI152NLin2009cAsym 
        elif dataset == "rsn" and parc_name not in ["DesikanKilliany", "Destrieux"]:
            ref_spaces_to_iterate = ["MNI152"]
            parc_spaces_to_iterate = ["MNI152NLin2009cAsym"]
        # for PET data and Schaefer parcellations: fsaverage > MNI152NLin6Asym
        elif dataset == "pet" and parc_name in ["Schaefer100MelbourneS1", "Schaefer200MelbourneS2", "Schaefer400MelbourneS3"]:
            ref_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
            parc_spaces_to_iterate = ["MNI152NLin6Asym", "fsaverage"]
        # HCPex parcellation exists only in MNI152NLin2009cAsym
        elif dataset == "pet" and parc_name == "HCPex":
            ref_spaces_to_iterate = ["MNI152NLin2009cAsym", "fsaverage"]
            parc_spaces_to_iterate = ["MNI152NLin2009cAsym", "fsaverage"]
        else:
            raise ValueError(f"We missed a case: Dataset: {dataset}; parcellation: {parc_name}")
            
        # run parcellation
        for ref_space, parc_space in zip(ref_spaces_to_iterate, parc_spaces_to_iterate):
            
            # Load the parcellation
            parc, labels = fetch_parcellation(parc_name, space=parc_space, return_loaded=True)
            print(f"Parcellation loaded for space {parc_space}")
            
            # Load the reference maps
            ref_maps_avail = ref_maps[ref_space]
            ref_paths = fetch_reference(dataset, maps=ref_maps_avail, space=ref_space, print_references=False)
            print(f"-> {len(ref_paths)} maps available for space {ref_space}")
            
            # TODO: fix this: we will now delete all potential subcortical data for the fsaverage maps
            # this makes sense as the MNI data of these maps is somewhat problematic
            # We could handle it by using MNI data only for subcortex and fsaverage for cortex
            # but this is not working yet as we have scaled the MNI data from 0-1 but we did not
            # scale the fsaverage data. This is not straightforward as we need to scale fsaverage data
            # to the range of the MNI data in cortical regions only.
            if ref_space == "fsaverage":
                labels_cx = ref_maps_df.columns[ref_maps_df.columns.str.contains("_SC_")]
                print(f"Dropping {len(labels_cx)} subcortical parcels")
                ref_maps_df.loc[ref_maps_avail, labels_cx] = np.nan
                
                # load fsaverage data and scale from 1e-6 - 1
                print("Scaling fsaverage data... CHANGE THIS")
                tmp_dir = Path(tempfile.mkdtemp())
                for i, (lh, rh) in enumerate(ref_paths):
                    # load img
                    lh, rh = images.load_gifti(lh), images.load_gifti(rh)
                    # resample parc to img
                    (parc_lh, parc_rh), _ = resample_images(
                        src=parc,
                        trg=(lh, rh),
                        src_space="fsaverage",
                        trg_space="fsaverage",
                        method="nearest",
                        resampling="transform_to_trg"
                    )
                    # get data
                    lh_dat, rh_dat = lh.agg_data(), rh.agg_data()
                    parc_lh_dat, parc_rh_dat = parc_lh.agg_data(), parc_rh.agg_data()
                    # rescale
                    lh_dat_rescaled, rh_dat_rescaled = np.zeros_like(lh_dat), np.zeros_like(rh_dat)
                    lh_dat_rescaled[parc_lh_dat > 0] = minmax_scale(lh_dat[parc_lh_dat > 0], feature_range=(1e-6, 1))
                    rh_dat_rescaled[parc_rh_dat > 0] = minmax_scale(rh_dat[parc_rh_dat > 0], feature_range=(1e-6, 1))
                    # to nifti
                    lh_rescaled = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(lh_dat_rescaled)])
                    rh_rescaled = nib.GiftiImage(darrays=[nib.gifti.GiftiDataArray(rh_dat_rescaled)])
                    lh_rescaled.to_filename(tmp_dir / f"lh_{i}.gii")
                    rh_rescaled.to_filename(tmp_dir / f"rh_{i}.gii")
                    ref_paths[i] = (tmp_dir / f"lh_{i}.gii", 
                                    tmp_dir / f"rh_{i}.gii")
            # that was terrible, but it works for now

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
        ref_maps_df.to_csv(nispace_source_data_path / "reference" / dataset / "tab" / f"dset-{dataset}_parc-{parc_name}.csv.gz")



# %%
