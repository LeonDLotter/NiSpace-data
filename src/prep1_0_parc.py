# %% Init

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets
from neuromaps import images, transforms
import templateflow.api as tflow
import matplotlib.pyplot as plt

# import NiSpace functions
from nispace.utils.utils import relabel_nifti_parc, apply_transform
from nispace.utils.utils_datasets import download, download_file
from utils import tflow_get
from nispace.datasets import template_lib, fetch_template
from nispace.plotting import brainplot

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# nispace data path 
nispace_source_data_path = wd
print(f"NiSpace source data path: {nispace_source_data_path}")
    
# MNI spaces
mni_spaces = ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]

# templates
mni_templates = {
    "MNI152NLin6Asym": {
        "mask": image.load_img(fetch_template("MNI152NLin6Asym", res="1mm", desc="mask")),
    },
    "MNI152NLin2009cAsym": {
        "mask": image.load_img(fetch_template("MNI152NLin2009cAsym", res="1mm", desc="mask")),
    },
}

# function to get save dir
def get_save_dir(name, space):
    fp = nispace_source_data_path / "parcellation" / name / space
    if not fp.exists():
        fp.mkdir(parents=True, exist_ok=True)
    return fp

# function to save parcellation and labels
def save_parc(parc, name, space, labels, save_dir=None, plot=False):
    if save_dir is None:
        save_dir = get_save_dir(name, space)

    # volumetric
    if "mni" in space.lower():
        n_expected = len(labels)
        
        print(f"  Expected: {n_expected} labels — {list(labels)[:3]} ...")
        
        # check parcel counts before saving
        _check_parcel_counts(parc, n_expected)

        # save
        nii_path = save_dir / f"parc-{name}_space-{space}.label.nii.gz"
        txt_path = save_dir / f"parc-{name}_space-{space}.label.txt"
        parc.to_filename(nii_path)
        with open(txt_path, "w") as f:
            f.write("\n".join(labels))

        # verify from disk
        saved_labels = np.loadtxt(txt_path, dtype=str).tolist()
        if isinstance(saved_labels, str):
            saved_labels = [saved_labels]
        if any(re.search(r"[./\?*', ]", l.split("lab-")[1]) for l in saved_labels):
            raise ValueError(f"Invalid characters found in labels: {saved_labels}")
        if len(saved_labels) != len(labels):
            raise ValueError(f"Label count mismatch: saved {len(saved_labels)}, expected {len(labels)}")
        if saved_labels != list(labels):
            first_diff = next(i for i, (a, b) in enumerate(zip(saved_labels, list(labels))) if a != b)
            raise ValueError(f"Label content mismatch at index {first_diff}: {saved_labels[first_diff]!r} != {list(labels)[first_diff]!r}")
        
        saved_nii = nib.load(nii_path)
        unique_vals = np.unique(saved_nii.get_fdata())[:5]
        _check_parcel_counts(saved_nii, n_expected)
        print(f"  Verified: {len(saved_labels)} labels — {saved_labels[:3]} ...")
        print(f"  Unique vals (first 5): {unique_vals}")

        # plot
        if plot:
            brainplot(data=parc, kind="slice", space=space, title=f"{name} | {space}")
            plt.show()

    # surface
    elif "fsaverage" in space.lower() or "fslr" in space.lower():
        if not isinstance(parc, tuple) or not isinstance(labels, tuple):
            raise ValueError("parc and labels must be tuples")

        unique_vals_hemi = {}
        for i_hemi, hemi in enumerate(["L", "R"]):
            n_expected = len(labels[i_hemi])
            
            print(f"  [{hemi}] Expected: {n_expected} labels — {list(labels[i_hemi])[:3]} ...")

            # check parcel counts before saving
            _check_parcel_counts(parc[i_hemi], n_expected)
            
            # save
            gii_path = save_dir / f"parc-{name}_space-{space}_hemi-{hemi}.label.gii.gz"
            txt_path = save_dir / f"parc-{name}_space-{space}_hemi-{hemi}.label.txt"
            parc[i_hemi].to_filename(gii_path)
            with open(txt_path, "w") as f:
                f.write("\n".join(labels[i_hemi]))

            # verify from disk
            saved_labels = np.loadtxt(txt_path, dtype=str).tolist()
            if isinstance(saved_labels, str):
                saved_labels = [saved_labels]
            if len(saved_labels) != len(labels[i_hemi]):
                raise ValueError(f"[{hemi}] Label count mismatch: saved {len(saved_labels)}, expected {len(labels[i_hemi])}")
            if saved_labels != list(labels[i_hemi]):
                first_diff = next(i for i, (a, b) in enumerate(zip(saved_labels, list(labels[i_hemi]))) if a != b)
                raise ValueError(f"[{hemi}] Label content mismatch at index {first_diff}: {saved_labels[first_diff]!r} != {list(labels[i_hemi])[first_diff]!r}")

            saved_gii = nib.load(gii_path)
            lt = saved_gii.labeltable.get_labels_as_dict()
            unique_vals_hemi[hemi] = np.unique(saved_gii.agg_data())
            unique_vals = unique_vals_hemi[hemi][:5]
            _check_parcel_counts(saved_gii, n_expected)
            print(f"  [{hemi}] Verified: {len(saved_labels)} labels — {saved_labels[:3]} ...")
            print(f"  [{hemi}] Labeltable (first 3): {dict(list(lt.items())[:3])}")
            print(f"  [{hemi}] Unique vals (first 5): {unique_vals}")

        # final test: any of lh in rh or vice versa: raise error
        unique_vals_overlap = np.intersect1d(unique_vals_hemi["L"], unique_vals_hemi["R"])
        print(f"  Unique vals overlap between hemispheres: {unique_vals_overlap}")
        if len(unique_vals_overlap) != 1:
            raise ValueError(f"LH and RH have overlapping unique values!")

        # plot
        if plot:
            brainplot(data=parc, kind="surface", space=space, title=f"{name} | {space}")
            plt.show()


# resample to template bbox
def resample_to_template(img, space):
    img_resampled = image.resample_to_img(
        source_img=img,
        target_img=mni_templates[space]["mask"],
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    return img_resampled

# check parcel counts
def _check_parcel_counts(parc, n_expected):
    if isinstance(parc, tuple):
        n_expected = n_expected // 2  # expect half the parcels in each hemisphere
        n_lh = len(np.trim_zeros(np.unique(parc[0].agg_data())))
        n_rh = len(np.trim_zeros(np.unique(parc[1].agg_data())))
        if not (n_lh == n_rh == n_expected):
            raise ValueError(f"Wrong unique index count: lh: {n_lh}, rh: {n_rh}, expected: {n_expected}")
    else:
        try:
            n = len(np.trim_zeros(np.unique(parc.get_fdata())))
        except Exception as e:
            n = len(np.trim_zeros(np.unique(parc.agg_data())))
        if n != n_expected:
            raise ValueError(f"Wrong unique index count: is: {n}, expected: {n_expected}")
        
# Get parcellations
# Label naming convention:
# hemi + "division" + label name
# division is optional and can be, e.g., a network or a broader anatomical division like a lobe
# e.g.: "hemi-{L|R|B}_lab-{label+name}"
# e.g.: "hemi-{L|R|B}_div-{network+name}_lab-{label+name}"

# parcellation info
parc_info = {}

# %% ===============================================================================================
# Schaefer

for schaefer_parcels in [100, 200, 400, 1000]:
    for schaefer_networks in [7, 17]:
        print(f"Schaefer {schaefer_parcels}-{schaefer_networks}")
        
        # name
        name = f"Schaefer{schaefer_parcels}Parcels{schaefer_networks}Networks"
        
        # labels 
        # get original
        labels = pd.read_table(
            tflow_get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"{schaefer_parcels}Parcels{schaefer_networks}Networks", suffix="dseg", extension="tsv")
        ).name.to_list()
        # rename
        labels = [f"hemi-{l.split('_')[1][0]}_div-{l.split('_')[2]}_lab-{'+'.join(l.split('_')[3:])}" for l in labels]
        labels_lh = [l for l in labels if "hemi-L" in l]
        labels_rh = [l for l in labels if "hemi-R" in l]
        print(labels_lh[:3], "...", labels_lh[-3:])
        print(labels_rh[:3], "...", labels_rh[-3:])
        
        
        # MNI ------------------------------------------------------------------------------------------
        for space in mni_spaces:
            print(space)
                
            # download parcellation
            parc = images.load_nifti(
                tflow_get(space, resolution="01", atlas="Schaefer2018",
                          desc=f"{schaefer_parcels}Parcels{schaefer_networks}Networks",
                          suffix="dseg", extension="nii.gz")
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
            )
            
            # add to parc_info
            parc_info[name, space] = {
                "level": "cortex",
                "n_parcels": len(labels), 
                "symmetric": False,
                "resolution": "1mm", 
                "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
                "license": "MIT"
            }

            
        # fsaverage ------------------------------------------------------------------------------------
        print("fsaverage")
        
        # download
        parc_fsa = (
            tflow_get("fsaverage", hemi="L", density="164k", atlas="Schaefer2018",
                      suffix="dseg", extension="label.gii",
                      name_filter=f"seg-{schaefer_networks}n_scale-{schaefer_parcels}_"),
            tflow_get("fsaverage", hemi="R", density="164k", atlas="Schaefer2018",
                      suffix="dseg", extension="label.gii",
                      name_filter=f"seg-{schaefer_networks}n_scale-{schaefer_parcels}_"),
        )
        
        # resample
        parc = transforms.fsaverage_to_fsaverage(
            data=parc_fsa,
            target_density="41k",
            method="nearest"
        )
        
        # make sure parcel indices are in consecutive order and have our labels
        parc = images.relabel_gifti(
            (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'), 
            images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
        )
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space="fsaverage", 
            labels=([l for l in labels if "hemi-L_" in l], [l for l in labels if "hemi-R_" in l]), 
        )
        
        # add to parc_info
        parc_info[name, "fsaverage"] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "symmetric": False,
            "resolution": "41k", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        
        
        # fsLR -------------------------------------------------------------------------------------
        # we'll convert from the high-resolution fsaverage because the fsLR version on GitHub 
        # misses two labels!
        print("fsLR")
        
        # resample and convert -> necessary, otherwise workbench adds labels for whatever reason
        # source (parc_fsa) is the downloaded paths from above
        parc = transforms.fsaverage_to_fslr(
            transforms.fsaverage_to_fsaverage(
                data=parc_fsa,
                target_density="41k",
                method="nearest"
            ), 
            target_density="32k",
            method="nearest"
        )
        
        # make sure parcel indices are in consecutive order and have our labels
        parc = images.relabel_gifti(
            (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'), 
             images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
        )
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space="fsLR", 
            labels=([l for l in labels if "hemi-L_" in l], [l for l in labels if "hemi-R_" in l]), 
        )
        
        # add to parc_info
        parc_info[name, "fsLR"] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "symmetric": False,
            "resolution": "32k", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        

# %% ===============================================================================================
# Yan2023 (homotopic, Kong17 17 networks)

CBIG_REPO = ("ThomasYeoLab/CBIG", "cb2e5bd8f5587485669f14e723c691ba83d0ae26")
YAN_BASE = "stable_projects/brain_parcellation/Yan2023_homotopic/parcellations"
doi_yan = "10.1016/j.neuroimage.2023.120010"

def _parse_yan_label(raw):
    # "17networks_LH_DefaultC_PHC"  → "hemi-L_div-DefaultC_lab-PHC"
    # "17networks_LH_SomMotA_1"     → "hemi-L_div-SomMotA_lab-1"
    # "17networks_LH_VisualC"       → "hemi-L_div-VisualC_lab-VisualC"  (no sub-region)
    parts = raw.split("_")
    hemi = "L" if parts[1] == "LH" else "R"
    div = parts[2]
    lab = "+".join(parts[3:]) if len(parts) > 3 else parts[2]
    return f"hemi-{hemi}_div-{div}_lab-{lab}"


for n_parcels in [100, 200, 400, 1000]:
    print(f"\n{'='*60}\nYan{n_parcels}")
    name = f"Yan{n_parcels}"

    # labels from dlabel label table ---------------------------------------------------------------
    dlabel_path = download_file(
        host="github",
        remote=(*CBIG_REPO, f"{YAN_BASE}/HCP/fsLR32k/kong17/{n_parcels}Parcels_Kong2022_17Networks.dlabel.nii")
    )
    label_dict = nib.load(dlabel_path).header.get_axis(0).label[0]  # {int: (name, rgba)}
    all_labels = [_parse_yan_label(label_dict[k][0]) for k in sorted(k for k in label_dict if k > 0)]
    labels_lh = [l for l in all_labels if "hemi-L" in l]
    labels_rh = [l for l in all_labels if "hemi-R" in l]
    print(f"  {len(labels_lh)} L + {len(labels_rh)} R parcels")
    print(f"  sample: {all_labels[:2]} ... {all_labels[-2:]}")

    # MNI152NLin6Asym ------------------------------------------------------------------------------
    space = "MNI152NLin6Asym"
    print(space)

    parc = images.load_nifti(
        download(
            f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/cb2e5bd8f5587485669f14e723c691ba83d0ae26/"
            f"{YAN_BASE}/MNI/kong17/{n_parcels}Parcels_Kong2022_17Networks_FSLMNI152_1mm.nii.gz"
        )
    )
    parc = relabel_nifti_parc(parc, np.arange(1, len(all_labels) + 1))
    parc = resample_to_template(parc, space)

    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=all_labels, 
    )
    parc_info[name, space] = {
        "level": "cortex", 
        "n_parcels": len(all_labels), 
        "symmetric": True,
        "resolution": "1mm",
        "publication": doi_yan, 
        "license": "free"
    }

    # MNI152NLin2009cAsym --------------------------------------------------------------------------
    space = "MNI152NLin2009cAsym"
    print(space)

    parc = apply_transform(
        img=wd / "parcellation" / name / "MNI152NLin6Asym" / f"parc-{name}_space-MNI152NLin6Asym.label.nii.gz",
        mni_from="MNI152NLin6Asym",
        mni_to="MNI152NLin2009cAsym",
        order=0,
    )

    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=all_labels, 
    )
    parc_info[name, space] = {
        "level": "cortex", 
        "n_parcels": len(all_labels), 
        "symmetric": True,
        "resolution": "1mm",
        "publication": doi_yan, 
        "license": "free"
    }

    # fsLR (native 32k dlabel) ---------------------------------------------------------------------
    space = "fsLR"
    print(space)

    parc = images.dlabel_to_gifti(dlabel_path)
    parc = images.relabel_gifti(
        (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'),
         images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
    )
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=(labels_lh, labels_rh), 
    )
    parc_info[name, space] = {
        "level": "cortex", 
        "n_parcels": len(all_labels), 
        "symmetric": True,
        "resolution": "32k",
        "publication": doi_yan, 
        "license": "free"
    }

    # fsaverage (annot → 164k → 41k) ---------------------------------------------------------------
    space = "fsaverage"
    print(space)

    parc = images.annot_to_gifti((
        download_file(host="github", remote=(*CBIG_REPO,
            f"{YAN_BASE}/FreeSurfer/fsaverage/label/kong17/lh.{n_parcels}Parcels_Kong2022_17Networks.annot")),
        download_file(host="github", remote=(*CBIG_REPO,
            f"{YAN_BASE}/FreeSurfer/fsaverage/label/kong17/rh.{n_parcels}Parcels_Kong2022_17Networks.annot"))
    ))
    parc = transforms.fsaverage_to_fsaverage(data=parc, target_density="41k", method="nearest")
    parc = images.relabel_gifti(
        (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'),
         images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
    )
    save_parc(
        parc=parc, 
        name=name,
        space=space, 
        labels=(labels_lh, labels_rh), 
    )
    parc_info[name, space] = {
        "level": "cortex", 
        "n_parcels": len(all_labels), 
        "symmetric": True,
        "resolution": "41k",
        "publication": doi_yan, 
        "license": "free"
    }


# %% ===============================================================================================
# Tian 

for tian in ["S1", "S2", "S3", "S4"]: 
    print(f"Tian {tian}")
    
    # name
    name = f"Tian{tian}"
    
    # labels
    labels = pd.read_csv(
        f"https://github.com/yetianmed/subcortex/raw/dcad93421ea8021d6c5738df0a915a2223cd82aa/Group-Parcellation/3T/Subcortex-Only/"
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
                remote=("yetianmed/subcortex", "dcad93421ea8021d6c5738df0a915a2223cd82aa", f"Group-Parcellation/3T/Subcortex-Only/{fn}"),
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
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": "subcortex",
            "n_parcels": len(labels), 
            "symmetric": True,
            "resolution": "1mm", 
            "publication": "10.1093/cercor/bhx179; 0.1038/s41593-020-00711-6",
            "license": "MIT"
        }
        
        
# %% ===============================================================================================
# FreeSurfer DK, DKT, Destrieux, and Glasser/HCPMMP
# All loaded from associated FreeSurfer GIN repo: https://gin.g-node.org/llotter/mni_freesurfer
gin_commit = "5839f92fdf207714fae18088da7a114b21a015c8"
        
for name, fs_name, doi in [
    ("DesikanKilliany", "aparc", "10.1016/j.neuroimage.2006.01.021; doi.org/10.12751/g-node.2mnxpm"), 
    ("DesikanKillianyTourville", "aparc.DKTatlas", "10.3389/fnins.2012.00171; doi.org/10.12751/g-node.2mnxpm"), 
    ("Destrieux", "aparc.a2009s", "10.1016/j.neuroimage.2010.06.010; doi.org/10.12751/g-node.2mnxpm"),
    ("Glasser", "HCPMMP1", "10.1038/nature18933; https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446")
]:
    print(name)
    
    # MNI ------------------------------------------------------------------------------------------
    for space in mni_spaces:
        print(space)
        
        # download parcellation
        parc = images.load_nifti(
            download(
                f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
                f"parcellations/{space}/seg-{fs_name}_space-{space}_desc-smoothed.nii.gz"
            )
        )
        
        # download labels
        labels = pd.read_table(
            f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
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
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "symmetric": True,
            "resolution": "1mm", 
            "publication": doi,
            "license": "free"
        }
        
        
    # fsaverage & fsLR -----------------------------------------------------------------------------
    for space, transform_fun, target_density in [
        ("fsaverage", transforms.fsaverage_to_fsaverage, "41k"), 
        ("fsLR", transforms.fslr_to_fslr, "32k")
    ]:
        print(space)
        
        # download parcellation
        parc = (
            images.load_gifti(download(
                f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
                f"parcellations/{space}/seg-{fs_name}_space-{space}_hemi-L.label.gii.gz"
            )),
            images.load_gifti(download(
                f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
                f"parcellations/{space}/seg-{fs_name}_space-{space}_hemi-R.label.gii.gz"
            ))
        )
        
        # download labels
        labels = pd.read_table(
            f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
            f"parcellations/{space}/seg-{fs_name}_space-{space}.tsv",
            index_col=0,
            header=None
        )[1].to_list()
        labels = [f"hemi-{l.split('-')[1][0].upper()}_lab-{('+'.join(l.split('-')[2:])).replace('_', '+')}" for l in labels]
        labels_lh = [l for l in labels if "hemi-L" in l]
        labels_rh = [l for l in labels if "hemi-R" in l]
        print(labels_lh)
        print(labels_rh)
        
        # to target density
        parc = transform_fun(
            parc,
            target_density=target_density,
            method="nearest"
        )
        
        # check
        print("Before relabeling")
        print("LH: ", np.unique(parc[0].agg_data()))
        print(parc[0].labeltable.get_labels_as_dict())
        print("RH: ", np.unique(parc[1].agg_data()))
        print(parc[1].labeltable.get_labels_as_dict())

        # # relabel giftis: first pass -> will remove all unwanted labels
        # parc = images.relabel_gifti(
        #     (images.construct_shape_gii(images.load_data(parc[0]), labels=labels, intent='NIFTI_INTENT_LABEL'), 
        #      images.construct_shape_gii(images.load_data(parc[1]), labels=labels, intent='NIFTI_INTENT_LABEL')), 
        #     background=labels_bg
        # )
        # insert our label names
        parc = images.relabel_gifti(
            (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'), 
             images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL')), 
        )

        # check
        print("After relabeling")
        print("LH: ", np.unique(parc[0].agg_data()))
        print(parc[0].labeltable.get_labels_as_dict())
        print("RH: ", np.unique(parc[1].agg_data()))
        print(parc[1].labeltable.get_labels_as_dict())

        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=(labels_lh, labels_rh), 
        )
        
        # info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels_lh) + len(labels_rh), 
            "symmetric": True,
            "resolution": target_density, 
            "publication": doi,
            "license": "free"
        }
        

# %% ===============================================================================================
# Aseg (FreeSurfer subcortical parcellation)
# Also sourced from FreeSurfer GIN repo
name = "Aseg"

# MNI ----------------------------------------------------------------------------------------------
for space in mni_spaces:
    print(space)
    
    # download parcellation
    parc = images.load_gifti(download(
        f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
        f"parcellations/{space}/seg-aseg_space-{space}.nii.gz"
    ))
     
    # download labels
    labels = pd.read_table(
        f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/"
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
    )
    
    # add to parc_info
    parc_info[name, space] = {
        "level": "subcortex",
        "n_parcels": len(labels), 
        "symmetric": True,
        "resolution": "1mm", 
        "publication": "10.1016/s0896-6273(02)00569-x",
        "license": "free"
    }
    
# %% ===============================================================================================
# Harvard-Oxford Cortical
for name, tf_name, nl_name, level, doi in [
    ("HarvardOxfordCortical", "HOCPAL", "cortl", "cortex", "doi.org/10.1016/j.schres.2005.11.020; doi.org/10.1016/j.neuroimage.2006.01.021"), 
    ("HarvardOxfordSubcortical", "HOSPA", "sub", "subcortex", "doi.org/10.1176/appi.ajp.162.7.1256; doi.org/10.1016/j.biopsych.2006.06.027"), 
]:
    print(name)


    # MNI ------------------------------------------------------------------------------------------
    for space in mni_spaces:
        print(space)
        
        # download parcellation
        parc = images.load_nifti(
            tflow_get(space, resolution="01", atlas=tf_name, desc="th25",
                      suffix="dseg", extension="nii.gz")
        )
        
        # get labels from nilearn (templateflow does not provide them)
        labels = datasets.fetch_atlas_harvard_oxford(f"{nl_name}-prob-1mm")["labels"]
        label_idc = [
            i for i, l in enumerate(labels) # we start from 0 because the background==0 label is included!
            if l not in ["Background", 
                         "Left Cerebral White Matter", "Right Cerebral White Matter",
                         "Left Cerebral Cortex", "Right Cerebral Cortex",
                         "Left Lateral Ventricle", "Right Lateral Ventricle",
                         "Brain-Stem"]
        ]
        label_idc_lh = [i for i in label_idc if "Left " in labels[i]]
        print("LH indices:", label_idc_lh)
        label_idc_rh = [i for i in label_idc if "Right " in labels[i]]
        print("RH indices:", label_idc_rh)
        assert 0 not in label_idc_lh and 0 not in label_idc_rh, "Background label should not be included"
        labels = [
            f"hemi-{labels[i][0]}_lab-" + \
                labels[i].split(" (")[0].replace("Left ", "").replace("Right ", "").replace(",", "").replace("'","").replace(" " , "+") 
            for i in label_idc_lh + label_idc_rh
        ]
        print(labels)
        
        # keep the right parcel indices and relabel consecutively
        parc = relabel_nifti_parc(parc, new_order=label_idc_lh + label_idc_rh)
        
        # resample
        parc = resample_to_template(parc, space)
        
        # save and plot
        save_parc(
            parc=parc, 
            name=name, 
            space=space, 
            labels=labels, 
        )
        
        # add to parc_info
        parc_info[name, space] = {
            "level": level,
            "n_parcels": len(labels), 
            "symmetric": True,
            "resolution": "1mm", 
            "publication": doi,
            "license": "free"
        }
        
        
    # fsaverage & fsLR -----------------------------------------------------------------------------
    # will be converted from MNI152NLin6Asym
    name = "HarvardOxfordCortical"
    print(name)

    # load MNI152NLin6Asym parcellation that we prepared above
    parc = images.load_nifti(
        get_save_dir(name, "MNI152NLin6Asym") / f"parc-{name}_space-MNI152NLin6Asym.label.nii.gz"
    )
    labels = pd.read_csv(
        get_save_dir(name, "MNI152NLin6Asym") / f"parc-{name}_space-MNI152NLin6Asym.label.txt",
        header=None
    )[0].to_list()

    # dict to store surfaces
    tmp = {"fsaverage": {"L": {}, "R": {}}, "fsLR": {"L": {}, "R": {}}}
    
    # iterate hemispheres
    for hemi, hemi_idx in [("L", 0), ("R", 1)]:
        print("transforming hemi:", hemi)
        
        # one-hemisphere volume
        label_idc_hemi = [i for i, l in enumerate(labels, start=1) if f"hemi-{hemi}_" in l]
        parc_hemi = relabel_nifti_parc(parc, new_order=label_idc_hemi, new_labels=label_idc_hemi)
        
        # convert to surface
        for space, transform_fun, target_density in [
            ("fsaverage", transforms.mni152_to_fsaverage, "41k"), 
            ("fsLR", transforms.mni152_to_fslr, "32k")
        ]:
            print("transforming space:", space)
            
            # transform    
            parc_surf = transform_fun(parc_hemi, target_density, method="nearest")
            parc_surf = parc_surf[hemi_idx] # left/right hemi
            
            # labels
            labels_hemi = [l for l in labels if f"hemi-{hemi}_" in l]
            
            # new labels
            parc_surf = images.construct_shape_gii(
                images.load_data(parc_surf), labels=["unknown"] + labels_hemi, intent='NIFTI_INTENT_LABEL'
            )
            if isinstance(parc_surf, tuple):
                if len(parc_surf) == 1:
                    parc_surf = parc_surf[0]
                elif len(parc_surf) > 1:
                    raise ValueError("parc_surf too long:", parc_surf)

            # save
            tmp[space][hemi]["surf"] = parc_surf
            tmp[space][hemi]["labels"] = labels_hemi
    
    # save and plot
    for space in ["fsaverage", "fsLR"]:
        print("saving space:", space)
            
        # save parc
        save_parc(
            parc=(tmp[space]["L"]["surf"], tmp[space]["R"]["surf"]), 
            name=name, 
            space=space, 
            labels=(tmp[space]["L"]["labels"], tmp[space]["R"]["labels"]), 
        )
        # save info
        parc_info[name, space] = {
            "level": "cortex",
            "n_parcels": len(labels), 
            "symmetric": True,
            "resolution": "41k" if space == "fsaverage" else "32k",
            "publication": parc_info[name, "MNI152NLin6Asym"]["publication"],
            "license": "free"
        }
    
 
# %% ===============================================================================================

# Brainnetome
# available from: http://www.brainnetome.org/resource/ or https://atlas.brainnetome.org/download.html
# download urls are generated by a cloud service and thus not stable over time
# we therefore load the files from a local source not uploaded to this repo
# if accessed from the webpages above, source files can be found as:
# MNI152NLin6Asym: BN_Atlas_246_1mm.nii.gz
# region labels: BN_Atlas_246_LUT.txt
# first 210 labels are cortical, last 36 are subcortical
# downloaded on 20. Jan 2026
print("BrainnetomeCortical and ...Subcortical")
doi = "10.1093/cercor/bhw157"

# MNI152NLin6Asym ----------------------------------------------------------------------------------
space = "MNI152NLin6Asym"
print(space)

# load parcellation from local
parc = images.load_nifti(wd / "_archive" / "brainnetome" / "BN_Atlas_246_1mm.nii.gz")

# load labels from local
labels = pd.read_table(
    wd / "_archive" / "brainnetome" / "BN_Atlas_246_LUT.txt",
    header=None,
    sep=r"\s+",
    skiprows=1
)[1].to_list()

# format label_L -> hemi-L_lab-label (lots of fall-backs here)
labels = [
    f"hemi-{l[-1]}_lab-" + l.replace("_L","").replace("_R","") \
        .replace("-","+").replace("_","+").replace("/","+").replace(".","+")
    for l in labels
]
labels_cx = labels[:210]
labels_sc = labels[210:]

# sanity check
print(f"All: {len(labels)} labels, first/last 5:", labels[:5], labels[-5:])
print(f"Cx: {len(labels_cx)} labels, first/last 5:", labels_cx[:5], labels_cx[-5:])
print(f"Sc: {len(labels_sc)} labels, first/last 5:", labels_sc[:5], labels_sc[-5:])

# load indices from image
idc_old = np.trim_zeros(np.unique(parc.get_fdata()))
assert len(labels) == len(idc_old), "Len of labels and idc does not match!"

# relabel image
parc_cx = relabel_nifti_parc(
    parc,
    new_order= \
        [i for i, l in zip(idc_old, labels) if "hemi-L" in l and i<=210] + \
        [i for i, l in zip(idc_old, labels) if "hemi-R" in l and i<=210]
)
parc_sc = relabel_nifti_parc(
    parc,
    new_order= \
        [i for i, l in zip(idc_old, labels) if "hemi-L" in l and i>210] + \
        [i for i, l in zip(idc_old, labels) if "hemi-R" in l and i>210]
)

# to target space
parc_cx = resample_to_template(parc_cx, space)
parc_sc = resample_to_template(parc_sc, space)

# sanity check
assert len(labels_cx) == len(np.trim_zeros(np.unique(parc_cx.get_fdata())))
assert len(labels_sc) == len(np.trim_zeros(np.unique(parc_sc.get_fdata())))

# save and plot
for name, parc, labels, level in [
    ("BrainnetomeCortical", parc_cx, labels_cx, "cortex"),
    ("BrainnetomeSubcortical", parc_sc, labels_sc, "subcortex")
]:
    
    # save
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=labels, 
    )

    # add to parc_info
    parc_info[name, space] = {
        "level": level,
        "n_parcels": len(labels), 
        "symmetric": True,
        "resolution": "1mm", 
        "publication": doi,
        "license": "non-commercial, attribution, share alike"
    }
    
# MNI152NLin2009cAsym ------------------------------------------------------------------------------
space = "MNI152NLin2009cAsym"  
print(space)

# transform to MNI...2009cAsym with templateflow transform

# Cx/Sc
for name, labels, level in [
    ("BrainnetomeCortical", labels_cx, "cortex"),
    ("BrainnetomeSubcortical", labels_sc, "subcortex")
]:
    
    # run transform
    parc = apply_transform(
        img=wd / "parcellation" / name / "MNI152NLin6Asym" / f"parc-{name}_space-MNI152NLin6Asym.label.nii.gz",
        mni_from="MNI152NLin6Asym",
        mni_to="MNI152NLin2009cAsym",
        order=0,
    )
    
    # save and plot
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=labels, 
    )

    # add to parc_info
    parc_info[name, space] = {
        "level": level,
        "n_parcels": len(labels), 
        "symmetric": True,
        "resolution": "1mm", 
        "publication": doi,
        "license": "non-commercial, attribution, share alike"
    }

# fsaverage ----------------------------------------------------------------------------------------
name = "BrainnetomeCortical"
space = "fsaverage"
print(space)

# load parc from local annot
parc = images.annot_to_gifti(
    (wd / "_archive" / "brainnetome" / "BN_Atlas_freesurfer" / "fsaverage" / "label" / "lh.BN_Atlas.annot",
     wd / "_archive" / "brainnetome" / "BN_Atlas_freesurfer" / "fsaverage" / "label" / "rh.BN_Atlas.annot")
)

# labels from above
labels_lh = [l for l in labels_cx if "hemi-L" in l]
labels_rh = [l for l in labels_cx if "hemi-R" in l]

# resample
parc = transforms.fsaverage_to_fsaverage(
    data=parc,
    target_density="41k",
    method="nearest"
)

# make sure parcel indices are in consecutive order and have our labels
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
)

# sanity check
print("LH labels:", labels_lh)
print("LH labels dict:", parc[0].labeltable.get_labels_as_dict())
print("LH values:", np.unique(parc[0].agg_data()))
print("RH labels:", labels_rh)
print("RH labels dict:", parc[1].labeltable.get_labels_as_dict())
print("RH values:", np.unique(parc[1].agg_data()))

# save and plot
save_parc(
    parc=parc, 
    name=name, 
    space=space, 
    labels=(labels_lh, labels_rh), 
)

# add to parc_info
parc_info[name, space] = {
    "level": "cortex",
    "n_parcels": len(labels_cx),
    "symmetric": True,
    "resolution": "41k", 
    "publication": doi,
    "license": "non-commercial, attribution, share alike"
}

# fsLR ---------------------------------------------------------------------------------------------
space = "fsLR"
print(space)

# load parc from local annot
parc_fsa = images.annot_to_gifti(
    (wd / "_archive" / "brainnetome" / "BN_Atlas_freesurfer" / "fsaverage" / "label" / "lh.BN_Atlas.annot",
     wd / "_archive" / "brainnetome" / "BN_Atlas_freesurfer" / "fsaverage" / "label" / "rh.BN_Atlas.annot")
)

# resample and transform
# necessary, will insert extra labels otherwise
parc = transforms.fsaverage_to_fslr(
    transforms.fsaverage_to_fsaverage(
        data=parc_fsa,
        target_density="41k",
        method="nearest"
    ), 
    target_density="32k", 
    method="nearest"
)

# make sure parcel indices are in consecutive order and have our labels
parc = images.relabel_gifti(
    (images.construct_shape_gii(images.load_data(parc[0]), labels=["unknown"] + labels_lh, intent='NIFTI_INTENT_LABEL'), 
     images.construct_shape_gii(images.load_data(parc[1]), labels=["unknown"] + labels_rh, intent='NIFTI_INTENT_LABEL'))
)

# sanity check
print("LH labels:", labels_lh)
print("LH labels dict:", parc[0].labeltable.get_labels_as_dict())
print("LH values:", np.unique(parc[0].agg_data()))
print("RH labels:", labels_rh)
print("RH labels dict:", parc[1].labeltable.get_labels_as_dict())
print("RH values:", np.unique(parc[1].agg_data()))

# save and plot
save_parc(
    parc=parc, 
    name=name, 
    space=space, 
    labels=(labels_lh, labels_rh), 
)

# add to parc_info
parc_info[name, space] = {
    "level": "cortex",
    "n_parcels": len(labels_cx), 
    "symmetric": True,
    "resolution": "32k", 
    "publication": doi,
    "license": "non-commercial, attribution, share alike"
}


# %% ===============================================================================================
# AAL (Automated Anatomical Labeling 2)
# Source: _archive/AAL2/... (local, not uploaded to repo)
# Downloaded from https://www.gin.cnrs.fr/fr/outils/aal/ on 20. Jan 2026
# direct download link: https://www.gin.cnrs.fr/wp-content/uploads/aal2_for_SPM12.tar.gz
print("AALCortical and AALSubcortical")
doi = "10.1016/j.neuroimage.2015.07.075"

# MNI152NLin6Asym ----------------------------------------------------------------------------------
space = "MNI152NLin6Asym"
print(space)

# load parc
parc = images.load_nifti(wd / "_archive" / "AAL2" / "atlas" / "AAL2.nii")

# load labels (here dataframe with label and idx columns)
labels_df = pd.read_csv(wd / "_archive" / "AAL2" / "aal2.nii.txt", header=None, index_col=0, sep=r"\s+")
labels_df.columns = ["label", "idx"]
print("Length of labels_df:", len(labels_df))

# subset: we keep only left/right cortical and subcortical labels 
# has any idx; contains _L, _R; does not contain Cerebellum, Vermis
labels_df = labels_df[labels_df.label.str.contains(r"_[LR]$")]  # keep only left/right labels
labels_df = labels_df[~labels_df.label.str.contains(r"Cerebelum|Cerebellum|Vermis")]  # exclude cerebellum and vermis
print("Length of filtered labels_df:", len(labels_df))

# re-label
labels_df.label = [f"hemi-{l[-1]}_lab-{l[:-2].replace('_', '+').replace(' ', '+').replace('-', '+')}" 
                   for l in labels_df.label] 

# insert cortex/subcortex index
labels_df["level"] = np.where(labels_df.label.str.contains(
    r"Hippocampus|Amygdala|Caudate|Putamen|Pallidum|Thalamus"), "subcortex", "cortex")

# resample parc
parc = resample_to_template(parc, space)

# create parcellations
labels_idc_cx = labels_df[(labels_df.level=="cortex") & labels_df.label.str.contains("hemi-L")].index.tolist() + \
    labels_df[(labels_df.level=="cortex") & labels_df.label.str.contains("hemi-R")].index.tolist()
labels_idc_sc = labels_df[(labels_df.level=="subcortex") & labels_df.label.str.contains("hemi-L")].index.tolist() + \
    labels_df[(labels_df.level=="subcortex") & labels_df.label.str.contains("hemi-R")].index.tolist()
labels_cx = labels_df.loc[labels_idc_cx].label.tolist()
labels_sc = labels_df.loc[labels_idc_sc].label.tolist()
parc_cx = relabel_nifti_parc(parc, new_order=labels_df.loc[labels_idc_cx].idx.tolist())
parc_sc = relabel_nifti_parc(parc, new_order=labels_df.loc[labels_idc_sc].idx.tolist(), new_labels=range(1, len(labels_sc) + 1))

# create parcellations
for name, parc, labels, level in [
    ("AALCortical", parc_cx, labels_cx, "cortex"),
    ("AALSubcortical", parc_sc, labels_sc, "subcortex")
]:
    
    # save
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=labels, 
    )
    
    # add to parc_info
    parc_info[name, space] = {
        "level": level,
        "n_parcels": len(labels), 
        "symmetric": True,
        "resolution": "1mm", 
        "publication": doi,
        "license": "free"
    }
    

# MNI152NLin2009cAsym ------------------------------------------------------------------------------
space = "MNI152NLin2009cAsym"
print(space)

for name, labels, level in [
    ("AALCortical",    labels_cx, "cortex"),
    ("AALSubcortical", labels_sc, "subcortex"),
]:
    parc = apply_transform(
        img=wd / "parcellation" / name / "MNI152NLin6Asym" / f"parc-{name}_space-MNI152NLin6Asym.label.nii.gz",
        mni_from="MNI152NLin6Asym",
        mni_to="MNI152NLin2009cAsym",
        order=0,
    )
    save_parc(
        parc=parc, 
        name=name, 
        space=space, 
        labels=labels, 
    )
    parc_info[name, space] = {
        "level": level, 
        "n_parcels": len(labels), 
        "symmetric": True,
        "resolution": "1mm",
        "publication": doi, 
        "license": "free"
    }

# fsaverage & fsLR (AALCortical only) --------------------------------------------------------------
name = "AALCortical"
print(name, "— surface spaces")

parc_cx_mni6 = images.load_nifti(
    get_save_dir(name, "MNI152NLin6Asym") / f"parc-{name}_space-MNI152NLin6Asym.label.nii.gz"
)
labels_cx_aal_loaded = np.loadtxt(
    get_save_dir(name, "MNI152NLin6Asym") / f"parc-{name}_space-MNI152NLin6Asym.label.txt", dtype=str
).tolist()

tmp_aal = {"fsaverage": {"L": {}, "R": {}}, "fsLR": {"L": {}, "R": {}}}

for hemi, hemi_idx in [("L", 0), ("R", 1)]:
    print(f"  hemi: {hemi}")

    # one-hemi volume with original consecutive indices for that hemi
    label_idc_hemi = [i for i, l in enumerate(labels_cx_aal_loaded, start=1) if f"hemi-{hemi}_" in l]
    parc_hemi = relabel_nifti_parc(parc_cx_mni6, new_order=label_idc_hemi, new_labels=label_idc_hemi)

    # check before relabeling
    print(f"    Before relabeling — unique: {np.unique(parc_hemi.get_fdata())[1:6]} ...")

    for space, transform_fun, target_density in [
        ("fsaverage", transforms.mni152_to_fsaverage, "41k"),
        ("fsLR",      transforms.mni152_to_fslr,      "32k"),
    ]:
        parc_surf = transform_fun(parc_hemi, target_density, method="nearest")
        parc_surf = parc_surf[hemi_idx]

        labels_hemi = [l for l in labels_cx_aal_loaded if f"hemi-{hemi}_" in l]
        parc_surf = images.construct_shape_gii(
            images.load_data(parc_surf), labels=["unknown"] + labels_hemi, intent="NIFTI_INTENT_LABEL"
        )
        if isinstance(parc_surf, tuple):
            parc_surf = parc_surf[0]

        tmp_aal[space][hemi]["surf"]   = parc_surf
        tmp_aal[space][hemi]["labels"] = labels_hemi

for space in ["fsaverage", "fsLR"]:
    print(f"  saving {space}")
    save_parc(
        parc=(tmp_aal[space]["L"]["surf"], tmp_aal[space]["R"]["surf"]),
        name=name,
        space=space,
        labels=(tmp_aal[space]["L"]["labels"], tmp_aal[space]["R"]["labels"]),
    )
    parc_info[name, space] = {
        "level": "cortex",
        "n_parcels": len(labels_cx),
        "symmetric": True,
        "resolution": "41k" if space == "fsaverage" else "32k",
        "publication": doi,
        "license": "free"
    }


# %% ===============================================================================================

# Save info
parc_info = pd.DataFrame.from_dict(parc_info).T
parc_info.index.names = ["parcellation", "space"]
parc_info.to_csv(nispace_source_data_path / "parcellation" / "metadata.csv")

# ==================================================================================================


# %%