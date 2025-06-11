# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn import plotting, image
from neuromaps import images, transforms
from netneurotools.datasets import fetch_mmpall
import templateflow.api as tflow

import matplotlib.pyplot as plt
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.utils.utils import relabel_nifti_parc
from nispace.utils.utils_datasets import download, download_file
from nispace.datasets import template_lib

# nispace data path 
nispace_source_data_path = wd
print(f"NiSpace source data path: {nispace_source_data_path}")
    
# MNI spaces
mni_spaces = ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]

# templates
mni_templates = {
    "MNI152NLin6Asym": {
        "mask": image.load_img(download(template_lib["MNI152NLin6Asym"]["1mm"]["mask"]["remote"])),
        "T1w": image.load_img(download(template_lib["MNI152NLin6Asym"]["1mm"]["T1w"]["remote"])),
    },
    "MNI152NLin2009cAsym": {
        "mask": image.load_img(download(template_lib["MNI152NLin2009cAsym"]["1mm"]["mask"]["remote"])),
        "T1w": image.load_img(download(template_lib["MNI152NLin2009cAsym"]["1mm"]["T1w"]["remote"])),
    },
}

# plot for mni
def plot_mni(parc, name, space):
    plotting.plot_stat_map(
        parc, 
        bg_img=mni_templates[space]["T1w"],
        title=f"{name} | {space}", 
        cut_coords=(0, 0, 0)
    )
    plt.show()

# function to get save dir
def get_save_dir(name, space):
    fp = nispace_source_data_path / "parcellation" / name / space
    if not fp.exists():
        fp.mkdir(parents=True, exist_ok=True)
    return fp

# function to save parcellation and labels
def save_parc(parc, name, space, labels, save_dir=None, plot=True):
    if save_dir is None:
        save_dir = get_save_dir(name, space)
    
    # volumetric
    if "mni" in space.lower():
        
        # save parcellation
        parc.to_filename(save_dir / f"parc-{name}_space-{space}.label.nii.gz")
        
        # save labels
        with open(save_dir / f"parc-{name}_space-{space}.label.txt", "w") as f:
            f.write("\n".join(labels))
            
        # plot
        if plot:
            plot_mni(parc, name, space)
            
    # surface
    elif "fsaverage" in space.lower() or "fslr" in space.lower():
        if not isinstance(parc, tuple) or not isinstance(labels, tuple):
            raise ValueError("parc and labels must be tuples")
        
        for i_hemi, hemi in enumerate(["L", "R"]):
            
            # save parcellation
            parc[i_hemi].to_filename(save_dir / f"parc-{name}_space-{space}_hemi-{hemi}.label.gii.gz")
            
            # save labels
            with open(save_dir / f"parc-{name}_space-{space}_hemi-{hemi}.label.txt", "w") as f:
                f.write("\n".join(labels[i_hemi]))
                
            # plot
            # not implemented yet
            
        
# resample to template bbox
def resample_to_template(img, space):
    img_resampled = image.resample_to_img(
        source_img=img,
        target_img=mni_templates[space]["mask"],
        interpolation="nearest"
    )
    return img_resampled


# %% Get parcellations
# Label naming convention:
# hemi + "division" + label name
# division is optional and can be, e.g., a network or a broader anatomical division like a lobe
# e.g.: "hemi-{L|R|B}_lab-{label+name}"
# e.g.: "hemi-{L|R|B}_div-{network+name}_lab-{label+name}"

# parcellation info
parc_info = {}

# %% ===============================================================================================
# Schaefer

for schaefer in [100, 200, 400]:
    print(f"Schaefer {schaefer}")
    
    # name
    name = f"Schaefer{schaefer}"
    
    # labels 
    # get original
    labels = pd.read_table(
        tflow.get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"{schaefer}Parcels7Networks", suffix="dseg")[0]
    ).name.to_list()
    # rename
    labels = [f"hemi-{l.split('_')[1][0]}_div-{l.split('_')[2]}_lab-{'+'.join(l.split('_')[3:])}" for l in labels]
    print(labels)
    
    
    # MNI ------------------------------------------------------------------------------------------
    for space in mni_spaces:
        print(space)
            
        # download parcellation
        parc = images.load_nifti(
            download(
                "https://templateflow.s3.amazonaws.com/"
                f"tpl-{space}/tpl-{space}_res-01_atlas-Schaefer2018_desc-{schaefer}Parcels7Networks_dseg.nii.gz"
            )
        )
        
        # make sure parcel indices are in consecutive order
        parc = relabel_nifti_parc(parc, np.arange(1, len(labels) + 1))

        # make sure we are in the correct space
        parc = resample_to_template(parc, space)
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=labels, 
            plot=True
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "resolution": "1mm", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }

        
    # fsaverage ------------------------------------------------------------------------------------
    print("fsaverage")
    
    # download
    parc = (
        download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                 f"tpl-fsaverage_hemi-L_den-164k_atlas-Schaefer2018_seg-7n_scale-{schaefer}_dseg.label.gii"),
        download("https://templateflow.s3.amazonaws.com/tpl-fsaverage/"
                 f"tpl-fsaverage_hemi-R_den-164k_atlas-Schaefer2018_seg-7n_scale-{schaefer}_dseg.label.gii")
    )
    
    # make sure parcel indices are in consecutive order
    parc = images.relabel_gifti(parc)
    
    # resample
    parc = transforms.fsaverage_to_fsaverage(
        data=parc,
        target_density="41k",
        method="nearest"
    )
    
    # save and plot
    save_parc(
        parc=parc, 
        name=name, 
        space="fsaverage", 
        labels=([l for l in labels if "hemi-L_" in l], [l for l in labels if "hemi-R_" in l]), 
        plot=True
    )
    
    # add to parc_info
    parc_info[name, "fsaverage"] = {
        "level": "cortex",
        "n_parcels": len(labels), 
        "resolution": "41k", 
        "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
        "license": "MIT"
    }
    
    
    # fsLR -----------------------------------------------------------------------------------------
    print("fsLR")
    
    # download, convert to gifti
    parc = images.dlabel_to_gifti(
        download_file(
            host="github",
            remote=("ThomasYeoLab/CBIG", 
                    "master", 
                    f"stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/"
                    f"Schaefer2018_{schaefer}Parcels_7Networks_order.dlabel.nii"),
        )
    ) 
    
    # make sure parcel indices are in consecutive order
    parc = images.relabel_gifti(parc)
    
    # save and plot
    save_parc(
        parc=parc, 
        name=name, 
        space="fsLR", 
        labels=([l for l in labels if "hemi-L_" in l], [l for l in labels if "hemi-R_" in l]), 
        plot=True
    )
    
    # add to parc_info
    parc_info[name, "fsLR"] = {
        "level": "cortex",
        "n_parcels": len(labels), 
        "resolution": "32k", 
        "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
        "license": "MIT"
    }

# %% ===============================================================================================
# Tian 

for tian in ["S1", "S2", "S3"]: 
    print(f"Tian {tian}")
    
    # name
    name = f"Tian{tian}"
    
    # labels
    labels = pd.read_csv(
        f"https://github.com/yetianmed/subcortex/raw/master/Group-Parcellation/3T/Subcortex-Only/"
        f"Tian_Subcortex_{tian}_3T_label.txt", 
        header=None
    )[0].to_list()
    # rename    
    labels = [f"hemi-{l[-2].upper()}_lab-{'+'.join(l.split('-')[:-1])}" for l in labels]
    print(labels)
    # order does not correspond to our convention, so we need to reorder
    labels_old_order = labels.copy()
    labels = [l for l in labels if "hemi-L_" in l] + [l for l in labels if "hemi-R_" in l]
    print(labels)
    
    # MNI ------------------------------------------------------------------------------------------
    for space in mni_spaces:
        print(space)
        
        # download parcellation
        fn = f"Tian_Subcortex_{tian}_3T{'_2009cAsym.nii.gz' if space == 'MNI152NLin2009cAsym' else '.nii'}"
        parc = images.load_nifti(
            download_file(
                host="github",
                remote=("yetianmed/subcortex", "master", f"Group-Parcellation/3T/Subcortex-Only/{fn}"),
            )
        )
        
        # make sure parcel indices are in consecutive order 
        parc = relabel_nifti_parc(parc, new_order=[labels_old_order.index(l)+1 for l in labels])
        
        # resample
        parc = resample_to_template(parc, space)
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=labels, 
            plot=True
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": "subcortex",
            "n_parcels": len(labels), 
            "resolution": "1mm", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        

# %% ===============================================================================================
# Glasser
print("Glasser")

# name
name = "Glasser"


# fsLR ---------------------------------------------------------------------------------------------
print("fsLR")
    
# load
parc = fetch_mmpall()

# make sure parcel indices are in consecutive order
parc = images.relabel_gifti((parc.lh, parc.rh))

# labels from fslr file
# get for one hemisphere (parcellation is symmetric)
labels = [
    f"lab-{l[1].replace('_ROI', '').replace('L_', '').replace('R_', '')}"
    for l in parc[0].labeltable.get_labels_as_dict().items() 
    if l[0] != 0
]
labels = (["hemi-L_" + l for l in labels], ["hemi-R_" + l for l in labels])
print(labels)

# save
save_parc(
    parc=parc, 
    name=name, 
    space="fsLR", 
    labels=labels, 
    plot=True
)

# add to parc_info
parc_info[name, "fsLR"] = {
    "level": "cortex",
    "n_parcels": len(labels[0]) * 2, 
    "resolution": "32k", 
    "publication": "10.1038/nature18933",
    "license": "https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms"
}


# fsaverage ----------------------------------------------------------------------------------------
print("fsaverage")
    
# load, convert to gifti
parc = images.annot_to_gifti(
    (download("https://figshare.com/ndownloader/files/5528816"),
     download("https://figshare.com/ndownloader/files/5528819"))
)

# make sure parcel indices are in consecutive order
parc = images.relabel_gifti(parc)

# resample
parc = transforms.fsaverage_to_fsaverage(
    data=parc,
    target_density="41k",
    method="nearest"
)

# save and plot
save_parc(
    parc=parc, 
    name=name, 
    space="fsaverage", 
    labels=labels, 
    plot=True
)

# info
parc_info[name, "fsaverage"] = {
    "level": "cortex",
    "n_parcels": len(labels[0]) * 2, 
    "resolution": "41k", 
    "publication": "10.1038/nature18933; https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446/2",
    "license": "CC-BY-4.0"
}


# %% ===============================================================================================
# DesikanKilliany & Destrieux

for name, fs_name, doi in [
    ("DesikanKilliany", "aparc", "10.1016/j.neuroimage.2006.01.021; doi.org/10.12751/g-node.2mnxpm"), 
    ("DesikanKillianyTourville", "aparc.DKTatlas", "10.3389/fnins.2012.00171; doi.org/10.12751/g-node.2mnxpm"), 
    ("Destrieux", "aparc.a2009s", "10.1016/j.neuroimage.2010.06.010; doi.org/10.12751/g-node.2mnxpm")
]:
    print(name)


    # MNI ------------------------------------------------------------------------------------------
    for space in mni_spaces:
        print(space)
        
        # download parcellation
        parc = images.load_nifti(
            download(
                "https://gin.g-node.org/llotter/mni_freesurfer/raw/a75429767d3939b8fad005adb814c25f23c78c85/"
                f"parcellations/{space}/seg-{fs_name}_space-{space}_desc-smoothed.nii.gz"
            )
        )
        
        # download labels
        labels = pd.read_table(
            "https://gin.g-node.org/llotter/mni_freesurfer/raw/a75429767d3939b8fad005adb814c25f23c78c85/"
            f"parcellations/{space}/seg-{fs_name}_space-{space}.tsv",
            index_col=0,
            header=None
        )[1].to_list()
        labels = [f"hemi-{l.split('-')[1][0].upper()}_lab-{('+'.join(l.split('-')[2:])).replace('_', '+')}" for l in labels]
        print(labels)
        
        # make sure parcel indices are in consecutive order
        parc = relabel_nifti_parc(parc, np.arange(1, len(labels) + 1))
        
        # resample
        parc = resample_to_template(parc, space)
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=labels, 
            plot=True
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "resolution": "1mm", 
            "publication": doi,
            "license": "free"
        }
        
        
    # fsaverage & fsLR -----------------------------------------------------------------------------
    for space in ["fsaverage", "fsLR"]:
        print(space)
        
        # skip if DKT
        # TODO: manage adding DKT atlas
        if name == "DesikanKillianyTourville":
            print("Skipping DKT atlas")
            continue

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
        
        # labels
        labels = [l for _, l in parc[0].labeltable.get_labels_as_dict().items()]
        labels_bg = ["unknown", "corpuscallosum", "medial_wall"]
        labels_lh = [f"hemi-L_{l.replace('-', '+').replace('_', '+')}" for l in [l for l in labels if l.lower() not in labels_bg]]
        labels_rh = [f"hemi-R_{l.replace('-', '+').replace('_', '+')}" for l in [l for l in labels if l.lower() not in labels_bg]]
        print(labels_lh)
        print(labels_rh)
        
        # convert to fslr
        if space == "fsLR":
            parc = transforms.fsaverage_to_fslr(
                data=parc,
                target_density="32k",
                method="nearest"
            )

        # relabel giftis
        parc = images.relabel_gifti(
            (images.construct_shape_gii(images.load_data(parc[0]), labels=labels, intent='NIFTI_INTENT_LABEL'), 
             images.construct_shape_gii(images.load_data(parc[1]), labels=labels, intent='NIFTI_INTENT_LABEL')), 
            background=labels_bg
        )

        # check
        print("LH: ", np.unique(parc[0].agg_data()))
        print("RH: ", np.unique(parc[1].agg_data()))

        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=(labels_lh, labels_rh), 
            plot=True
        )
        
        # info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels_lh) + len(labels_rh), 
            "resolution": "41k" if space == "fsaverage" else "32k", 
            "publication": "10.1016/j.neuroimage.2006.01.021" if "Desikan" in name else "10.1016/j.neuroimage.2010.06.010",
            "license": "free"
        }
        
        
# %% ===============================================================================================
# Aseg
name = "Aseg"

# MNI ----------------------------------------------------------------------------------------------
for space in mni_spaces:
    print(space)
    
    # download parcellation
    parc = images.load_gifti(download(
        "https://gin.g-node.org/llotter/mni_freesurfer/raw/c406d4f41aca04f6497649a85bfcfc4de93ab7a2/"
        f"parcellations/{space}/seg-aseg_space-{space}.nii.gz"
    ))
     
    # download labels
    labels = pd.read_table(
        "https://gin.g-node.org/llotter/mni_freesurfer/raw/c406d4f41aca04f6497649a85bfcfc4de93ab7a2/"
        f"parcellations/{space}/seg-aseg_space-{space}.tsv",
        index_col=0,
        header=None
    )[1].to_list()
    labels = [f"hemi-{l[0]}_lab-{l.replace('Left-', '').replace('Right-', '').replace('-', '+')}" for l in labels]
    print(labels)

    # make sure parcel indices are in consecutive order
    parc = relabel_nifti_parc(parc, np.arange(1, len(labels) + 1))
    
    # resample
    parc = resample_to_template(parc, space)
    
    # save and plot
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=labels, 
        plot=True
    )
    
    # add to parc_info
    parc_info[name, space] = {
        "level": "subcortex",
        "n_parcels": len(labels), 
        "resolution": "1mm", 
        "publication": "10.1016/s0896-6273(02)00569-x",
        "license": "free"
    }
    

# %% ===============================================================================================

# Save info

parc_info = pd.DataFrame.from_dict(parc_info).T
parc_info.index.names = ["parcellation", "space"]
parc_info.to_csv(nispace_source_data_path / "parcellation" / "metadata.csv")

# ==================================================================================================


# %%