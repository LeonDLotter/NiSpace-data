# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from neuromaps import images, transforms
from netneurotools.datasets import fetch_schaefer2018, fetch_mmpall

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.utils.utils import relabel_nifti_parc
from nispace.utils.utils_datasets import download, download_file

# nispace data path 
nispace_source_data_path = wd


# %% Get parcellations

# parcellation info
parc_info = {}
#pd.DataFrame(columns=["parcellation", "n_parcels", "space", "resolution", "publication"])


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
    labs_schaefer = pd.read_csv(
        f"https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/"
        f"Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"
        f"Schaefer2018_{schaefer}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )["ROI Name"].to_list()
    
    # combine and rename
    labs_old = []
    for l in labs_tian:
        l = l.split("-")
        labs_old.append(l[-1].upper() + "_SC_" + "-".join(l[:-1]))
    for l in labs_schaefer:
        l = l.split("_")
        labs_old.append(l[1] + "_CX_" + "_".join(l[2:]))
    print("old labels: ", labs_old[:5], "...")
    #labs_old = [f"{i}_{l}" for i, l in enumerate(labs_old, start=1)]
    
    # REORDER LABELS: LH CX -> LH SC -> RH CX -> RH SC
    labs = \
        [l for l in labs_old if "LH_CX_" in l] + \
        [l for l in labs_old if "RH_CX_" in l] + \
        [l for l in labs_old if "LH_SC_" in l] + \
        [l for l in labs_old if "RH_SC_" in l]
    #labs = [f"{i}_{l}" for i, l in enumerate(labs, start=1)]
    print("new labels: ", labs[:5], "...")
    
    # Load parcellation and reorder labels and save all data
    for space in ["MNI152NLin2009cAsym", "MNI152NLin6Asym", "fsaverage", "fsLR", ]:
        save_dir = nispace_source_data_path / "parcellation" / name / space
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
                
        # MNI spaces
        if "MNI152" in space:
            
            # PARCELLATION
            save_path = save_dir / f"parc-{name}_space-{space}.label.nii.gz"
            print(f"Loading parcellation {name}, {space}...")
            # download image
            parc = images.load_nifti(
                download_file(
                    host="github",
                    remote=("yetianmed/subcortex", 
                            "master", 
                            f"Group-Parcellation/3T/Cortex-Subcortex/MNIvolumetric/"
                            f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}_{'3T_' if space == 'MNI152NLin2009cAsym' else ''}{space}_1mm.nii.gz"),
                )
            )
            # reorder labels
            parc = relabel_nifti_parc(parc, new_order=[labs_old.index(l_new) + 1 for l_new in labs])
            # save
            print(f"Saving relabeled parcellation {name}, {space}...")
            parc.to_filename(save_path)
        
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
            
            # load via netneurotools, convert to gifti and relabel
            parc = fetch_schaefer2018("fsaverage5")[f"{schaefer}Parcels7Networks"]
            parc = images.relabel_gifti(
                images.annot_to_gifti(
                    (parc.lh, parc.rh)
                )
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
            res = "10k"
            
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
                    remote=("yetianmed/subcortex", 
                            "master", 
                            f"Group-Parcellation/3T/Cortex-Subcortex/"
                            f"Schaefer2018_{schaefer}Parcels_7Networks_order_Tian_Subcortex_{tian}.dlabel.nii"),
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
#   HCP

print("HCPex")
# name
name = "HCPex"

## SPACE: fsLR (original) ------------------------------------------------------------
space = "fsLR"
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
    
# PARCELLATION
save_path = (save_dir / f"parc-HCPex_space-fsLR_hemi-L.label.gii.gz",
             save_dir / f"parc-HCPex_space-fsLR_hemi-R.label.gii.gz")

# load and relabel
parc = fetch_mmpall()
parc = images.relabel_gifti((parc.lh, parc.rh))

# save
parc[0].to_filename(save_path[0])
parc[1].to_filename(save_path[1])

# LABELS
# save path
save_path = (save_dir / f"parc-HCPex_space-fsLR_hemi-L.label.txt",
             save_dir / f"parc-HCPex_space-fsLR_hemi-R.label.txt")
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
save_path = (save_dir / f"parc-HCPex_space-fsaverage_hemi-L.label.gii.gz",
             save_dir / f"parc-HCPex_space-fsaverage_hemi-R.label.gii.gz")

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
    nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-HCPex_space-fsLR_hemi-L.label.txt", 
    save_dir / f"parc-{name}_space-fsaverage_hemi-L.label.txt"
)
shutil.copy(
    nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-HCPex_space-fsLR_hemi-R.label.txt", 
    save_dir / f"parc-{name}_space-fsaverage_hemi-R.label.txt"
)

# info
parc_info[name, space] = {
    "n_parcels": len(labs), 
    "resolution": "41k", 
    "publication": "10.1038/nature18933; https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446/2",
    "license": "CC-BY-4.0"
}


## SPACE: MNI152NLin2009cAsym  ------------------------------------------------------------

space = "MNI152NLin2009cAsym"
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)

# source PARCELLATION and LABELS

# load 
parc_mni = images.load_nifti(
    download_file(
        host="github",
        remote=("wayalan/HCPex", "main", "HCPex_v1.1/HCPex.nii.gz"),
    )
)
labs_mni = np.loadtxt(
    download("https://github.com/wayalan/HCPex/raw/main/HCPex_v1.1/HCPex.nii.txt"), 
    str
).tolist()

# new PARCELLATION and LABELS

# reload fslr labels
labs_fslr = np.concatenate([
    np.loadtxt(
        nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-HCPex_space-fsLR_hemi-L.label.txt", 
        str
    ),
    np.loadtxt(
        nispace_source_data_path / "parcellation" / name / "fsLR" / "parc-HCPex_space-fsLR_hemi-R.label.txt", 
        str
    )
])

# we make a dataframe with the labels for mni and fslr
labs_matching = pd.DataFrame({
    "mni_idx": [int(l[0]) for l in labs_mni],
    "mni_label": [f"{l[1]}H_{'CX' if int(l[0]) <= 360 else 'SC'}_{l[2]}" for l in labs_mni],
    "fslr_idx": [int(l.split("_")[0]) for l in labs_fslr] + [""] * 66,
    "fslr_label": ["_".join(l.split("_")[1:]) for l in labs_fslr] + [""] * 66,
})
# One single label is different: H in fsLR is Hipp in mni
labs_matching.replace({"LH_CX_Hipp": "LH_CX_H", "RH_CX_Hipp": "RH_CX_H"}, inplace=True)
print(labs_matching)

# check if all fslr (original) labels in mni
print("fsLR (original) labels that are not in mni:")
print([l for l in labs_matching["fslr_label"] if l not in labs_matching["mni_label"].to_list() and l != ""])
print("mni labels that are not in fsLR (original):")
print([l for l in labs_matching["mni_label"] if l not in labs_matching["fslr_label"].to_list() and "_SC_" not in l])

# new label_order in MNI
labs_matching["mni_idx_new"] = \
    [
        labs_matching.loc[labs_matching["mni_label"] == l_fslr]["mni_idx"].values[0] 
        for l_fslr in labs_matching["fslr_label"] 
        if l_fslr != ""
    ] + \
    labs_matching["mni_idx"].iloc[360:].to_list()
labs_matching["mni_label_new"] = \
    [
        labs_matching.loc[labs_matching["mni_idx"] == i]["mni_label"].values[0] 
        for i in labs_matching["mni_idx_new"] 
    ]
print(labs_matching)

# PARCELLATION
save_path = save_dir / f"parc-{name}_space-{space}.label.nii.gz"

# Well, that was a mess. now make the new volume
parc = relabel_nifti_parc(
    parc_mni, 
    new_order=labs_matching["mni_idx_new"], 
    new_labels=np.arange(1, len(labs_matching["mni_idx_new"]) + 1)
)

# save
parc.to_filename(save_path)

# LABELS
save_path = save_dir / f"parc-{name}_space-{space}.label.txt"
with open(save_path, "w") as f:
    f.write("\n".join([f"{i}_{l}" for i, l in enumerate(labs_matching["mni_label_new"], start=1)]))

# info
parc_info[name, space] = {
    "n_parcels": len(labs_matching["mni_label_new"]), 
    "resolution": "1mm", 
    "publication": "10.1038/nature18933; 10.1007/s00429-021-02421-6",
    "license": "GPL-3.0"
}


# ==================================================================================================


# ==================================================================================================
# DesikanKilliany

print("DesikanKilliany")

# name
name = "DesikanKilliany"
space = "fsaverage"

# load parcellation from templateflow
parc = (
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-41k_atlas-Desikan2006_seg-aparc_dseg.label.gii")),
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-R_den-41k_atlas-Desikan2006_seg-aparc_dseg.label.gii"))
)

# get labels
labs = [l for _, l in parc[0].labeltable.get_labels_as_dict().items()]
labs_bg = ["unknown", "corpuscallosum"]

# relabel giftis
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
     background=labs_bg
)

# info
parc_info[name, space] = {
    "n_parcels": len([l for l in labs if l != "unknown"]) * 2, 
    "resolution": "41k", 
    "publication": "https://doi.org/10.1016/j.neuroimage.2006.01.021",
    "license": "free"
}

# save maps and labels
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
# left
parc[0].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-L.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-L.label.txt", "w") as f:
    f.write("\n".join([f"{i}_LH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=1)]))
# right
parc[1].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-R.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-R.label.txt", "w") as f:
    f.write("\n".join([f"{i}_RH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=35)]))

# ==================================================================================================


# ==================================================================================================

# Destrieux
print("Destrieux")
# name
name = "Destrieux"
space = "fsaverage"

# load parcellation from templateflow
parc = (
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-41k_atlas-Destrieux2009_dseg.label.gii")),
    images.load_gifti(download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-R_den-41k_atlas-Destrieux2009_dseg.label.gii"))
)

# get labels
labs = [l for l in parc[0].labeltable.get_labels_as_dict().values()]
labs_bg = ["Unknown", "Medial_wall"]

# relabel giftis
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=labs, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=labs, intent='NIFTI_INTENT_LABEL')), 
     background=labs_bg
)

# info
parc_info[name, space] = {
    "n_parcels": len([l for l in labs if l != "unknown"]) * 2, 
    "resolution": "41k", 
    "publication": "10.1016/j.neuroimage.2010.06.010",
    "license": "free"
}

# save maps and labels
save_dir = nispace_source_data_path / "parcellation" / name / space
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
# left
parc[0].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-L.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-L.label.txt", "w") as f:
    f.write("\n".join([f"{i}_LH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=1)]))
# right
parc[1].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-R.label.gii.gz")
with open(save_dir / f"parc-{name}_space-{space}_hemi-R.label.txt", "w") as f:
    f.write("\n".join([f"{i}_RH_CX_{l}" for i, l in enumerate([l for l in labs if l not in labs_bg], start=75)]))

# ==================================================================================================

# Save info

parc_info = pd.DataFrame.from_dict(parc_info).T
parc_info.index.names = ["parcellation", "space"]
parc_info.to_csv(nispace_source_data_path / "parcellation" / "metadata.csv")

# ==================================================================================================


# %%