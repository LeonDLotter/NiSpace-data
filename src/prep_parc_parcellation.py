# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from nilearn import plotting, image
from neuromaps import images, transforms
from netneurotools.datasets import fetch_mmpall
import templateflow.api as tflow

import matplotlib.pyplot as plt
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.utils.utils import relabel_nifti_parc, merge_parcellations
from nispace.utils.utils_datasets import download, download_file

# nispace data path 
nispace_source_data_path = wd
print(f"NiSpace source data path: {nispace_source_data_path}")

# plot for mni
def plot_mni(parc, name, space):
    plotting.plot_stat_map(
        parc, 
        bg_img=download(f"https://templateflow.s3.amazonaws.com/tpl-{space}/tpl-{space}_res-01_T1w.nii.gz"),
        title=f"{name} | {space}", 
        cut_coords=(0, 0, 0)
    )
    plt.show()


# %% Get parcellations

# parcellation info
parc_info = {}

# ==================================================================================================
# Schaefer + Melbourne

for schaefer, tian in [(100, "S1"), (200, "S2"), (400, "S3")]:
    print(f"Schaefer {schaefer} + Melbourne {tian}")
    
    # name
    name = f"Schaefer{schaefer}Melbourne{tian}"
    
    # labels Melbourne
    labs_tian = pd.read_csv(
        f"https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/"
        f"Tian_Subcortex_{tian}_3T_label.txt", 
        header=None
    )[0].to_list()
    
    # labels Schaefer
    labs_schaefer = pd.read_table(
        tflow.get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"{schaefer}Parcels7Networks", suffix="dseg")[0]
    ).name.to_list()
    
    # combine and rename
    labs_old = []
    for l in labs_tian:
        l = l.split("-")
        labs_old.append(l[-1].upper() + "_SC_" + "-".join(l[:-1]))
    for l in labs_schaefer:
        l = l.split("_")
        labs_old.append(l[1] + "_CX_" + "_".join(l[2:]))
    print("old labels: ", labs_old[:5], "...")
    
    # REORDER LABELS: LH CX -> LH SC -> RH CX -> RH SC
    labs = \
        [l for l in labs_old if "LH_CX_" in l] + \
        [l for l in labs_old if "RH_CX_" in l] + \
        [l for l in labs_old if "LH_SC_" in l] + \
        [l for l in labs_old if "RH_SC_" in l]
    #labs = [f"{i}_{l}" for i, l in enumerate(labs, start=1)]
    print("new labels: ", labs[:5], "...")

    # Load parcellation and reorder labels and save all data
    for space in ["MNI152NLin6Asym", "MNI152NLin2009cAsym", "fsaverage", "fsLR", ]:
        print(space)
        save_dir = nispace_source_data_path / "parcellation" / name / space
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
                
        # MNI spaces
        if "MNI152" in space:
            save_path = save_dir / f"parc-{name}_space-{space}.label.nii.gz"
            print(f"Loading parcellation {name}, {space}...")
            
            # download cortical
            parc_cx = images.load_nifti(
                download(
                    "https://templateflow.s3.amazonaws.com/"
                    f"tpl-{space}/tpl-{space}_res-01_atlas-Schaefer2018_desc-{schaefer}Parcels7Networks_dseg.nii.gz")
            )
            
            # download subcortical
            fn = f"Tian_Subcortex_{tian}_3T{'_2009cAsym.nii.gz' if space == 'MNI152NLin2009cAsym' else '.nii'}"
            parc_sc = images.load_nifti(
                download_file(
                    host="github",
                    remote=("yetianmed/subcortex", "master", 
                            f"Group-Parcellation/3T/Subcortex-Only/{fn}"),
                )
            )
            parc_sc = image.resample_to_img(parc_sc, parc_cx, interpolation="nearest")
            
            # combine
            parc, _ = merge_parcellations([parc_cx, parc_sc])
            # save
            print(f"Saving relabeled parcellation {name}, {space}...")
            parc.to_filename(save_path)
            # plot
            plot_mni(parc, name, space)
        
            # LABELS
            save_path = save_dir / f"parc-{name}_space-{space}.label.txt"
            # add index to labels and save
            with open(save_path, "w") as f:
                f.write("\n".join([f"{i}_{l}" for i, l in enumerate(labs, start=1)]))
                
            # Resolution
            res = "1mm"
   
                
        # fsaverage space
        elif space == "fsaverage":
            
            # PARCELLATION
            save_path = (save_dir / f"parc-{name}_space-fsaverage_hemi-L.label.gii.gz",
                         save_dir / f"parc-{name}_space-fsaverage_hemi-R.label.gii.gz")
            print(f"Loading parcellation {name}, {space}...")
            
            # load 
            parc = images.relabel_gifti(
                (download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                          f"tpl-fsaverage_hemi-L_den-164k_atlas-Schaefer2018_seg-7n_scale-{schaefer}_dseg.label.gii"),
                 download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                          f"tpl-fsaverage_hemi-R_den-164k_atlas-Schaefer2018_seg-7n_scale-{schaefer}_dseg.label.gii"))
            )
            # resample
            parc = transforms.fsaverage_to_fsaverage(
                data=parc,
                target_density="41k",
                method="nearest"
            )
            # save
            parc[0].to_filename(save_path[0])
            parc[1].to_filename(save_path[1])
            
            # LABELS
            save_path = (save_dir / f"parc-{name}_space-fsaverage_hemi-L.label.txt",
                         save_dir / f"parc-{name}_space-fsaverage_hemi-R.label.txt")
            with open(save_path[0], "w") as f:
                f.write("\n".join([f"{i}_{l}" for i, l in enumerate([l for l in labs if "LH_CX_" in l], start=1)]))
            with open(save_path[1], "w") as f:
                f.write("\n".join([f"{i}_{l}" for i, l in enumerate([l for l in labs if "RH_CX_" in l], start=schaefer // 2 + 1)]))
            
            # Resolution
            res = "41k"
            
        # fslr space
        elif space == "fsLR":
            
            # PARCELLATION
            save_path = (save_dir / f"parc-{name}_space-fsLR_hemi-L.label.gii.gz",
                         save_dir / f"parc-{name}_space-fsLR_hemi-R.label.gii.gz")
            print(f"Loading parcellation {name}, {space}...")
            # download, convert to gifti and relabel
            parc = images.relabel_gifti(
                images.dlabel_to_gifti(
                    download_file(
                    host="github",
                    remote=("ThomasYeoLab/CBIG", 
                            "master", 
                            f"stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/"
                            f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"),
                    )
                )                
            )
            # save
            parc[0].to_filename(save_path[0])
            parc[1].to_filename(save_path[1])
            
            # LABELS: same as fsaverage
            shutil.copy(
                nispace_source_data_path / "parcellation" / name / "fsaverage" / f"parc-{name}_space-fsaverage_hemi-L.label.txt", 
                save_dir / f"parc-{name}_space-fsLR_hemi-L.label.txt"
            )
            shutil.copy(
                nispace_source_data_path / "parcellation" / name / "fsaverage" / f"parc-{name}_space-fsaverage_hemi-R.label.txt", 
                save_dir / f"parc-{name}_space-fsLR_hemi-R.label.txt"
            )
            # Resolution
            res = "32k"

        # write info and labels, we have the same labels (obviously) for both resolutions
        parc_info[name, space] = {
            "n_parcels": len(labs), 
            "resolution": res, 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        
# ==================================================================================================


# ==================================================================================================
#   Glasser

print("Glasser")
# name
name = "Glasser"

## SPACE: fsLR (original) ------------------------------------------------------------
space = "fsLR"
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
    
# PARCELLATION
save_path = (save_dir / f"parc-Glasser_space-fsLR_hemi-L.label.gii.gz",
             save_dir / f"parc-Glasser_space-fsLR_hemi-R.label.gii.gz")

# load and relabel
parc = fetch_mmpall()
parc = images.relabel_gifti((parc.lh, parc.rh))

# save
parc[0].to_filename(save_path[0])
parc[1].to_filename(save_path[1])

# LABELS
# save path
save_path = (save_dir / f"parc-Glasser_space-fsLR_hemi-L.label.txt",
             save_dir / f"parc-Glasser_space-fsLR_hemi-R.label.txt")
# one hemisphere, but symmetric
labs = [l[1].replace("_ROI","").replace("L_","").replace("R_","") 
        for l in parc[0].labeltable.get_labels_as_dict().items() 
        if l[0] != 0]
# save
with open(save_path[0], "w") as f:
    f.write("\n".join([f"{i}_LH_CX_{l}" for i, l in enumerate(labs, start=1)]))
with open(save_path[1], "w") as f:
    f.write("\n".join([f"{i}_RH_CX_{l}" for i, l in enumerate(labs, start=len(labs) + 1)]))

# info
parc_info[name, space] = {
    "n_parcels": len(labs), 
    "resolution": "32k", 
    "publication": "10.1038/nature18933",
    "license": "https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms"
}


## SPACE: fsaverage ----------------------------------------------------------------
space = "fsaverage"
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
    
# PARCELLATION
save_path = (save_dir / f"parc-Glasser_space-fsaverage_hemi-L.label.gii.gz",
             save_dir / f"parc-Glasser_space-fsaverage_hemi-R.label.gii.gz")

# load and relabel
parc = images.relabel_gifti(
    images.annot_to_gifti(
        (download_file(remote="https://figshare.com/ndownloader/files/5528816"),
         download_file(remote="https://figshare.com/ndownloader/files/5528819"))
    )
)
parc = transforms.fsaverage_to_fsaverage(
    data=parc,
    target_density="41k",
    method="nearest"
)

# save
parc[0].to_filename(save_path[0])
parc[1].to_filename(save_path[1])

# LABELS (copy from fslr)
shutil.copy(
    nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-Glasser_space-fsLR_hemi-L.label.txt", 
    save_dir / f"parc-{name}_space-fsaverage_hemi-L.label.txt"
)
shutil.copy(
    nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-Glasser_space-fsLR_hemi-R.label.txt", 
    save_dir / f"parc-{name}_space-fsaverage_hemi-R.label.txt"
)

# info
parc_info[name, space] = {
    "n_parcels": len(labs), 
    "resolution": "41k", 
    "publication": "10.1038/nature18933; https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446/2",
    "license": "CC-BY-4.0"
}

# ==================================================================================================


# ==================================================================================================
# DesikanKilliany & Destrieux

for name in ["DesikanKillianyAseg", "DestrieuxAseg"]:
    print(name)

    # SPACE: mni ---------------------------------------------------------------------------------------
    for space in ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]:
        print(space)
        save_dir = nispace_source_data_path / "parcellation" / name / space
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        
        seg = "aparc.aseg" if "Desikan" in name else "aparc.a2009s.aseg"
        # load parcellation from G-Node
        parc = images.load_gifti(
            download(
                "https://gin.g-node.org/llotter/mni_freesurfer/raw/c406d4f41aca04f6497649a85bfcfc4de93ab7a2/"
                f"parcellations/{space}/seg-{seg}_space-{space}_desc-smoothed.nii.gz"
            )
        )
        labs_tmp = pd.read_table(
            "https://gin.g-node.org/llotter/mni_freesurfer/raw/c406d4f41aca04f6497649a85bfcfc4de93ab7a2/"
            f"parcellations/{space}/seg-{seg}_space-{space}.tsv",
            index_col=0,
            header=None
        )
        labs_tmp = labs_tmp[1].to_list()
        labs = []
        for i, l in enumerate(labs_tmp, start=1):
            if l.startswith("ctx-"):
                labs.append(
                    f"{i}_{l.split('-')[1].upper()}_CX_{l.replace('ctx-lh-', '').replace('ctx-rh-', '')}"
                )
            else:
                labs.append(
                    f"{i}_{'LH' if 'Left' in l else 'RH'}_SC_{l.replace('Left-', '').replace('Right-', '')}"
                )
                
        # save
        save_path = save_dir / f"parc-{name}_space-{space}.label.nii.gz"
        parc.to_filename(save_path)

        # LABELS
        save_path = save_dir / f"parc-{name}_space-{space}.label.txt"
        with open(save_path, "w") as f:
            f.write("\n".join(labs))

        # info
        parc_info[name, space] = {
            "n_parcels": len(labs), 
            "resolution": "1mm", 
            "publication": "10.1016/j.neuroimage.2006.01.021" if "Desikan" in name else "10.1016/j.neuroimage.2010.06.010",
            "license": "free"
        }
        
    # SPACE: fsaverage and fslr --------------------------------------------------------------------
    for space in ["fsaverage", "fsLR"]:
        print(space)
        save_dir = nispace_source_data_path / "parcellation" / name / space
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        # load parcellation from templateflow
        parc = (
            images.load_gifti(download(
                f"https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                f"tpl-fsaverage_hemi-L_den-41k_atlas-{'Desikan2006_seg-aparc' if 'Desikan' in name else 'Destrieux2009'}_dseg.label.gii"
            )),
            images.load_gifti(download(
                f"https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                f"tpl-fsaverage_hemi-R_den-41k_atlas-{'Desikan2006_seg-aparc' if 'Desikan' in name else 'Destrieux2009'}_dseg.label.gii"
            ))
        )
        
        # get labels
        labs = [l for _, l in parc[0].labeltable.get_labels_as_dict().items()]
        labs_bg = ["unknown", "corpuscallosum", "medial_wall"]
        labs_lh = [f"{i}_LH_CX_{l}" for i, l in enumerate([l for l in labs if l.lower() not in labs_bg], start=1)]
        labs_rh = [f"{i}_RH_CX_{l}" for i, l in enumerate([l for l in labs if l.lower() not in labs_bg], start=len(labs_lh)+1)]
        
        # convert to fslr
        if space == "fsLR":
            parc = transforms.fsaverage_to_fslr(
                data=parc,
                target_density="32k",
                method="nearest"
            )

        # relabel giftis
        parc = images.relabel_gifti(
            (images.construct_shape_gii(images.load_data(parc[0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
            images.construct_shape_gii(images.load_data(parc[1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
            background=labs_bg
        )

        # check
        print("LH: ", np.unique(parc[0].agg_data()))
        print("RH: ", np.unique(parc[1].agg_data()))

        # save maps and labels
        # left
        parc[0].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-L.label.gii.gz")
        with open(save_dir / f"parc-{name}_space-{space}_hemi-L.label.txt", "w") as f:
            f.write("\n".join(labs_lh))
        # right
        parc[1].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-R.label.gii.gz")
        with open(save_dir / f"parc-{name}_space-{space}_hemi-R.label.txt", "w") as f:
            f.write("\n".join(labs_rh))
            
        # info
        parc_info[name, space] = {
            "n_parcels": len(labs_lh) + len(labs_rh), 
            "resolution": "41k" if space == "fsaverage" else "32k", 
            "publication": "10.1016/j.neuroimage.2006.01.021" if "Desikan" in name else "10.1016/j.neuroimage.2010.06.010",
            "license": "free"
        }


# ==================================================================================================

# Save info

parc_info = pd.DataFrame.from_dict(parc_info).T
parc_info.index.names = ["parcellation", "space"]
parc_info.to_csv(nispace_source_data_path / "parcellation" / "metadata.csv")

# ==================================================================================================


# %%