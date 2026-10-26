# %% Init

import json
import shutil
import yaml
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import image, datasets
import templateflow.api as tflow

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

import hashlib
import tarfile
from nispace.transform import mni_to_mni
from nispace.utils.utils_datasets import download
from nispace.utils.utils_transforms import barycentric_weights
from utils import tflow_get

nispace_data_path = wd
nispace_toolbox_path = Path.home() / "projects" / "nispace"

# Load canonical geometry reference built by prep0_0_affines.py
with open(nispace_data_path / "template" / "affines.yaml") as _f:
    affines_db = yaml.safe_load(_f)

FSLR_DENSITIES = ["4k", "8k", "32k", "164k"]
FSAVERAGE_DENSITIES = ["3k", "10k", "41k", "164k"]

# Liberal cortex gray matter masks sourced from Schaefer parcellation (templateflow)
mask_liberalcortex_MNI6_1mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="01", suffix="dseg", extension="nii.gz")
)
mask_liberalcortex_MNI6_2mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin6Asym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="02", suffix="dseg", extension="nii.gz")
)
mask_liberalcortex_MNI9_1mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin2009cAsym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="01", suffix="dseg", extension="nii.gz")
)
mask_liberalcortex_MNI9_2mm = image.math_img(
    "img > 0", 
    img=tflow_get("MNI152NLin2009cAsym", atlas="Schaefer2018", desc=f"100Parcels7Networks", resolution="02", suffix="dseg", extension="nii.gz")
)

# Conservative cortex gray matter masks sourced from freesurfer segmentation of MNI templates (gin)
gin_commit = "5839f92fdf207714fae18088da7a114b21a015c8"
mask_tightcortex_MNI6_1mm = image.math_img(
    "img > 0",
    img=download(f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/parcellations/MNI152NLin6Asym/seg-aparc_space-MNI152NLin6Asym.nii.gz")
)
mask_tightcortex_MNI9_1mm = image.math_img(
    "img > 0",
    img=download(f"https://gin.g-node.org/llotter/mni_freesurfer/raw/{gin_commit}/parcellations/MNI152NLin2009cAsym/seg-aparc_space-MNI152NLin2009cAsym.nii.gz")
)

# Subcortex mask from HCP CIFTI grayordinates (natively MNI152NLin6Asym 2mm)
# Atlas_ROIs.2.nii.gz defines exactly which subcortical voxels are included in fsLR grayordinates
hcp_commit = "e2d5bbad6d48452deb15c4293556b535c1973a75"
hcp_subcortex = download(f"https://github.com/Washington-University/HCPpipelines/raw/{hcp_commit}/global/templates/91282_Greyordinates/Atlas_ROIs.2.nii.gz")
subcortexmask_MNI6_2mm = image.resample_img(
    image.math_img("(img != 0) & (img != 8) & (img != 47)", img=hcp_subcortex),
    target_affine=np.array(affines_db["MNI152NLin6Asym"]["2mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin6Asym"]["2mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
)
cbmask_MNI6_2mm = image.resample_img(
    image.math_img("(img == 8) | (img == 47)", img=hcp_subcortex),
    target_affine=np.array(affines_db["MNI152NLin6Asym"]["2mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin6Asym"]["2mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
) 
subcortexmask_MNI6_1mm = image.resample_img(
    subcortexmask_MNI6_2mm,
    target_affine=np.array(affines_db["MNI152NLin6Asym"]["1mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin6Asym"]["1mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
)
cbmask_MNI6_1mm = image.resample_img(
    cbmask_MNI6_2mm,
    target_affine=np.array(affines_db["MNI152NLin6Asym"]["1mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin6Asym"]["1mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
)
# Cross-space: MNI9 subcortexmask must be pre-computed (MNI9 is processed before MNI6 in the loop)
subcortexmask_MNI9_2mm = mni_to_mni(
    subcortexmask_MNI6_2mm, "MNI152NLin6Asym", "MNI152NLin2009cAsym", res="2mm", order=0
)
cbmask_MNI9_2mm = mni_to_mni(
    cbmask_MNI6_2mm, "MNI152NLin6Asym", "MNI152NLin2009cAsym", res="2mm", order=0
)
subcortexmask_MNI9_1mm = image.resample_img(
    subcortexmask_MNI9_2mm,
    target_affine=np.array(affines_db["MNI152NLin2009cAsym"]["1mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin2009cAsym"]["1mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
)
cbmask_MNI9_1mm = image.resample_img(
    cbmask_MNI9_2mm,
    target_affine=np.array(affines_db["MNI152NLin2009cAsym"]["1mm"]["affine"]),
    target_shape=tuple(affines_db["MNI152NLin2009cAsym"]["1mm"]["shape"]),
    interpolation="nearest", copy_header=True, force_resample=True,
)

# Combined GM masks: cortex | subcortex | cerebellum (computed at 1mm and 2mm; 3mm resampled via tuple)
gmmask_MNI6_1mm      = image.math_img("a | b | c", a=mask_liberalcortex_MNI6_1mm, b=subcortexmask_MNI6_1mm, c=cbmask_MNI6_1mm)
gmmask_MNI6_2mm      = image.math_img("a | b | c", a=mask_liberalcortex_MNI6_2mm, b=subcortexmask_MNI6_2mm, c=cbmask_MNI6_2mm)
gmmask_MNI9_1mm      = image.math_img("a | b | c", a=mask_liberalcortex_MNI9_1mm, b=subcortexmask_MNI9_1mm, c=cbmask_MNI9_1mm)
gmmask_MNI9_2mm      = image.math_img("a | b | c", a=mask_liberalcortex_MNI9_2mm, b=subcortexmask_MNI9_2mm, c=cbmask_MNI9_2mm)
tightgmmask_MNI6_1mm = image.math_img("a | b | c", a=mask_tightcortex_MNI6_1mm,   b=subcortexmask_MNI6_1mm, c=cbmask_MNI6_1mm)
tightgmmask_MNI9_1mm = image.math_img("a | b | c", a=mask_tightcortex_MNI9_1mm,   b=subcortexmask_MNI9_1mm, c=cbmask_MNI9_1mm)

# Sources for MNI templates.
# Values are either a URL string (load/download directly) or a (space, res, desc) tuple
# referencing an already-processed entry (resample if same space, transform if different).
MNI_TEMPLATE_SOURCES = {
    "MNI152NLin2009cAsym": {
        "1mm": {
            "T1w":             tflow_get("MNI152NLin2009cAsym", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":           tflow_get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="T1w", extension="nii.gz"),
            "gmprob":          tflow_get("MNI152NLin2009cAsym", resolution="01", label="GM", suffix="probseg", extension="nii.gz"),
            "brainmask":       tflow_get("MNI152NLin2009cAsym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
            "cortexmask":      mask_liberalcortex_MNI9_1mm,
            "tightcortexmask": mask_tightcortex_MNI9_1mm,
            "subcortexmask":   subcortexmask_MNI9_1mm,
            "cerebellummask":  cbmask_MNI9_1mm,
            "gmmask":          gmmask_MNI9_1mm,
            "tightgmmask":     tightgmmask_MNI9_1mm,
        },
        "2mm": {
            "T1w":             tflow_get("MNI152NLin2009cAsym", resolution="02", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":           tflow_get("MNI152NLin2009cAsym", resolution="02", desc="brain", suffix="T1w", extension="nii.gz"),
            "gmprob":          tflow_get("MNI152NLin2009cAsym", resolution="02", label="GM", suffix="probseg", extension="nii.gz"),
            "brainmask":       tflow_get("MNI152NLin2009cAsym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
            "cortexmask":      mask_liberalcortex_MNI9_2mm,
            "tightcortexmask": ("MNI152NLin2009cAsym", "1mm", "tightcortexmask"),
            "subcortexmask":   subcortexmask_MNI9_2mm,
            "cerebellummask":  cbmask_MNI9_2mm,
            "gmmask":          gmmask_MNI9_2mm,
            "tightgmmask":     ("MNI152NLin2009cAsym", "1mm", "tightgmmask"),
        },
        "3mm": {
            "T1w":             ("MNI152NLin2009cAsym", "1mm", "T1w"),
            "brain":           ("MNI152NLin2009cAsym", "1mm", "brain"),
            "gmprob":          ("MNI152NLin2009cAsym", "1mm", "gmprob"),
            "brainmask":       ("MNI152NLin2009cAsym", "1mm", "brainmask"),
            "cortexmask":      ("MNI152NLin2009cAsym", "1mm", "cortexmask"),
            "tightcortexmask": ("MNI152NLin2009cAsym", "1mm", "tightcortexmask"),
            "subcortexmask":   ("MNI152NLin2009cAsym", "1mm", "subcortexmask"),
            "cerebellummask":  ("MNI152NLin2009cAsym", "1mm", "cerebellummask"),
            "gmmask":          ("MNI152NLin2009cAsym", "1mm", "gmmask"),
            "tightgmmask":     ("MNI152NLin2009cAsym", "1mm", "tightgmmask"),
        },
    },
    "MNI152NLin6Asym": {
        "1mm": {
            "T1w":             tflow_get("MNI152NLin6Asym", resolution="01", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":           tflow_get("MNI152NLin6Asym", resolution="01", desc="brain", suffix="T1w", extension="nii.gz"),
            "gmprob":          ("MNI152NLin2009cAsym", "1mm", "gmprob"),  # MNI6 has no TPMs in templateflow
            "brainmask":       tflow_get("MNI152NLin6Asym", resolution="01", desc="brain", suffix="mask", extension="nii.gz"),
            "cortexmask":      mask_liberalcortex_MNI6_1mm,
            "tightcortexmask": mask_tightcortex_MNI6_1mm,
            "subcortexmask":   subcortexmask_MNI6_1mm,
            "cerebellummask":  cbmask_MNI6_1mm,
            "gmmask":          gmmask_MNI6_1mm,
            "tightgmmask":     tightgmmask_MNI6_1mm,
        },
        "2mm": {
            "T1w":             tflow_get("MNI152NLin6Asym", resolution="02", desc=None, suffix="T1w", extension="nii.gz"),
            "brain":           tflow_get("MNI152NLin6Asym", resolution="02", desc="brain", suffix="T1w", extension="nii.gz"),
            "gmprob":          ("MNI152NLin6Asym", "1mm", "gmprob"),
            "brainmask":       tflow_get("MNI152NLin6Asym", resolution="02", desc="brain", suffix="mask", extension="nii.gz"),
            "cortexmask":      mask_liberalcortex_MNI6_2mm,
            "tightcortexmask": ("MNI152NLin6Asym", "1mm", "tightcortexmask"),
            "subcortexmask":   subcortexmask_MNI6_2mm,
            "cerebellummask":  cbmask_MNI6_2mm,
            "gmmask":          gmmask_MNI6_2mm,
            "tightgmmask":     ("MNI152NLin6Asym", "1mm", "tightgmmask"),
        },
        "3mm": {
            "T1w":             ("MNI152NLin6Asym", "1mm", "T1w"),
            "brain":           ("MNI152NLin6Asym", "1mm", "brain"),
            "gmprob":          ("MNI152NLin6Asym", "1mm", "gmprob"),
            "brainmask":       ("MNI152NLin6Asym", "1mm", "brainmask"),
            "cortexmask":      ("MNI152NLin6Asym", "1mm", "cortexmask"),
            "tightcortexmask": ("MNI152NLin6Asym", "1mm", "tightcortexmask"),
            "subcortexmask":   ("MNI152NLin6Asym", "1mm", "subcortexmask"),
            "cerebellummask":  ("MNI152NLin6Asym", "1mm", "cerebellummask"),
            "gmmask":          ("MNI152NLin6Asym", "1mm", "gmmask"),
            "tightgmmask":     ("MNI152NLin6Asym", "1mm", "tightgmmask"),
        },
    },
}


def dst_path_mni(space, desc, res):
    fname = f"tpl-{space}_desc-{desc}_res-{res}.nii.gz"
    return nispace_data_path / "template" / space / "map" / desc / fname


def get_gii_ext(src_path):
    """Infer .surf.gii / .shape.gii / .label.gii from source filename."""
    name = str(src_path)
    for prefix in [".surf.", ".shape.", ".label.", ".func."]:
        if prefix in name:
            return prefix[1:] + "gii"
    return "gii"


# Surface atlases come from the neuromaps OSF tarballs, pinned by file ID and md5 (from neuromaps' osf.json,
# commit f0ed67c), NOT from the local ~/neuromaps-data cache: neuromaps never re-verifies cached files, and caches
# from before neuromaps PR #92 (2023) hold broken fsLR 4k/8k right-hemisphere medial-wall masks.
NEUROMAPS_OSF = {
    ("fsLR", "4k"):        ("4mw3a", "63eadf00cb544b03fa9e6f00", "72b7b17d389a04774ca6fd4ca28b4087"),
    ("fsLR", "8k"):        ("4mw3a", "63eadef9b3fed60444e3434f", "a1226aee262475e23d2cc37b84ef261d"),
    ("fsLR", "32k"):       ("4mw3a", "60b684b53a6df1021bd4df2d", "7932b4418f63d28935b5adf67150b16f"),
    ("fsLR", "164k"):      ("4mw3a", "60b684b9cb2a5e01fd68c115", "e86a2e9d998e07d621e8f31f0205bf3e"),
    ("fsaverage", "3k"):   ("4mw3a", "60b684ac9096b7021c63d9c5", "019dbf849671f5bed5a42476e5359997"),
    ("fsaverage", "10k"):  ("4mw3a", "60b684ab9096b7021b63cf6b", "c61384c271ee2e6b5449222281137414"),
    ("fsaverage", "41k"):  ("4mw3a", "60b684aecb2a5e01fc68b7e1", "0cc48e9d5d5bb0216502888c954805fd"),
    ("fsaverage", "164k"): ("4mw3a", "60b684b2cb2a5e01e968c918", "f4969a33b8cd88e46821ef581e492282"),
    ("regfusion", ""):     ("4mw3a", "60b684c03a6df1020ed525f6", "8e6c4679366031d44b030e9566e101a0"),
}
neuromaps_archive = nispace_data_path / "_archive" / "neuromaps_atlases"


def fetch_neuromaps_tarball(atlas, density=""):
    """Download (once) + md5-check + extract a pinned neuromaps atlas tarball; return its atlas dir."""
    project, file_id, md5 = NEUROMAPS_OSF[(atlas, density)]
    name = f"{atlas}{density}"
    tar_path = neuromaps_archive / f"{name}.tar.gz"
    out_dir = neuromaps_archive / name
    if not tar_path.exists():
        neuromaps_archive.mkdir(parents=True, exist_ok=True)
        download(f"https://files.osf.io/v1/resources/{project}/providers/osfstorage/{file_id}", tar_path)
    got = hashlib.md5(tar_path.read_bytes()).hexdigest()
    assert got == md5, f"md5 mismatch for {tar_path.name}: {got} != {md5}"
    if not out_dir.exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(out_dir, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(out_dir)
    return out_dir / "atlases" / atlas


def neuromaps_surface_files(space, density):
    """{desc: (L, R)} of the neuromaps atlas files for one space/density (same descs as neuromaps' fetchers)."""
    atlas_dir = fetch_neuromaps_tarball(space, density)
    surfs = ["midthickness", "inflated", "veryinflated", "sphere"] if space == "fsLR" else \
            ["white", "pial", "inflated", "sphere"]
    names = {s: f"{s}.surf.gii" for s in surfs}
    names.update(medial="desc-nomedialwall_dparc.label.gii", sulc="desc-sulc_midthickness.shape.gii",
                 vaavg="desc-vaavg_midthickness.shape.gii")
    files = {}
    for desc, suffix in names.items():
        pair = tuple(atlas_dir / f"tpl-{space}_den-{density}_hemi-{h}_{suffix}" for h in ("L", "R"))
        if all(p.exists() for p in pair):  # veryinflated only exists for fsLR 32k/164k
            files[desc] = pair
    # fsLR sphere registered to fsaverage: needed for fsLR <-> fsaverage transforms
    if space == "fsLR":
        files["sphere_space-fsaverage"] = tuple(
            atlas_dir / f"tpl-fsLR_space-fsaverage_den-{density}_hemi-{h}_sphere.surf.gii" for h in ("L", "R"))
    return files


def write_regfusion_meshes(space, density, sphere_files):
    """Registration fusion (Wu et al. 2018, via neuromaps) vertex positions in MNI152NLin6Asym as GIFTI meshes
    (coordinates + the faces of that density). neuromaps provides no fsLR 4k/8k tables: those are derived from
    the fsLR 32k table by barycentric resampling of the coordinates on the fsLR spheres (as wb_command
    -surface-resample does), i.e. the mapping is composed a priori."""
    rf_dir = fetch_neuromaps_tarball("regfusion")
    out = []
    for h, sphere_file in zip(("L", "R"), sphere_files):
        coords_sph, faces = [d.data for d in nib.load(sphere_file).darrays]
        table = rf_dir / f"tpl-MNI152_space-{space}_den-{density}_hemi-{h}_regfusion.txt"
        if table.exists():
            coords = np.loadtxt(table)
        else:
            assert space == "fsLR", f"no regfusion table for {space} {density}"
            sph_32k = fetch_neuromaps_tarball("fsLR", "32k") / f"tpl-fsLR_den-32k_hemi-{h}_sphere.surf.gii"
            c32, f32 = [d.data for d in nib.load(sph_32k).darrays]
            coords = barycentric_weights(c32, f32, coords_sph) @ np.loadtxt(
                rf_dir / f"tpl-MNI152_space-fsLR_den-32k_hemi-{h}_regfusion.txt")
        assert len(coords) == len(coords_sph)
        gii = nib.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(coords.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"),
        ])
        tmp = nispace_data_path / "_archive" / "neuromaps_atlases" / f"regfusion_{space}_{density}_{h}.surf.gii"
        gii.to_filename(tmp)
        out.append(tmp)
    return tuple(out)


def copy_and_register(space, density, atlas):
    """Copy surface files {desc: (L, R)} to nispace-data and return template.json sub-dict."""
    entries = {}
    for desc, files in atlas.items():
        entries[desc] = {}
        for hemi, src in zip(("L", "R"), map(Path, files)):
            ext = get_gii_ext(src)
            dst_name = f"tpl-{space}_desc-{desc}_res-{density}_hemi-{hemi}.{ext}"
            dst = nispace_data_path / "template" / space / "map" / desc / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"  {dst_name}")
            shutil.copy(src, dst)
            entries[desc][hemi] = {
                "host": "github-nispace",
                "remote": f"template/{space}/map/{desc}/{dst_name}",
            }
    return entries


# %% Build complete template.json

template_json = {}

# %% MNI templates — unified loop: download URLs, resample (same space) or transform (cross-space)

print("\n=== MNI templates ===")
processed = {}   # (space, res, desc) -> Path

for space, space_data in MNI_TEMPLATE_SOURCES.items():
    template_json[space] = {}
    for res, descs in space_data.items():
        print(f"\n{space} {res}:")
        template_json[space][res] = {}

        # Reference geometry from affines.json — no T1w-first ordering needed
        ref_aff   = np.array(affines_db[space][res]["affine"])
        ref_shape = tuple(affines_db[space][res]["shape"])
        interp    = lambda desc: "nearest" if "mask" in desc else "continuous"

        for desc, source in descs.items():
            dst = dst_path_mni(space, desc, res)
            dst.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(source, (str, Path, nib.spatialimages.SpatialImage)):
                if isinstance(source, Path):
                    print(f"  Copying {dst.name} ...")
                    shutil.copy(source, dst)
                    img = image.load_img(dst)
                elif isinstance(source, str):
                    print(f"  Downloading {dst.name} ...")
                    download(source, dst)
                    img = image.load_img(dst)
                else:
                    print(f"  Saving {dst.name} ...")
                    source.to_filename(dst)
                    img = source

                if img.shape[:3] != ref_shape or not np.allclose(img.affine, ref_aff):
                    print(
                        f"  WARNING: geometry mismatch for '{desc}' — "
                        f"shape {img.shape[:3]} (expected {ref_shape}), "
                        f"voxsize {np.abs(np.diag(img.affine)[:3]).round(4)} "
                        f"(expected {np.abs(np.diag(ref_aff)[:3]).round(4)}) — resampling ..."
                    )
                    image.resample_img(
                        img,
                        target_affine=ref_aff,
                        target_shape=ref_shape,
                        interpolation=interp(desc),
                        copy_header=True,
                        force_resample=True,
                    ).to_filename(dst)

            elif isinstance(source, tuple):
                src_space, src_res, src_desc = source
                src_path = processed[(src_space, src_res, src_desc)]

                if src_space == space:
                    # Same space — resample to canonical reference geometry from affines.json
                    print(f"  Resampling {src_path.name} -> {dst.name} ...")
                    image.resample_img(
                        image.load_img(src_path),
                        target_affine=ref_aff,
                        target_shape=ref_shape,
                        interpolation=interp(desc),
                        copy_header=True,
                        force_resample=True,
                    ).to_filename(dst)

                else:
                    # Different space — apply MNI-to-MNI transform, then check geometry
                    print(f"  Transforming {src_path.name} ({src_space} -> {space}) ...")
                    mni_to_mni(
                        img=src_path,
                        mni_from=src_space,
                        mni_to=space,
                        res=res,
                        order=0 if "mask" in desc else 3,
                    ).to_filename(dst)
                    img = image.load_img(dst)
                    if img.shape[:3] != ref_shape or not np.allclose(img.affine, ref_aff):
                        print(
                            f"  WARNING: geometry mismatch after transform for '{desc}' — "
                            f"shape {img.shape[:3]} (expected {ref_shape}), "
                            f"voxsize {np.abs(np.diag(img.affine)[:3]).round(4)} "
                            f"(expected {np.abs(np.diag(ref_aff)[:3]).round(4)}) — resampling ..."
                        )
                        image.resample_img(
                            img,
                            target_affine=ref_aff,
                            target_shape=ref_shape,
                            interpolation=interp(desc),
                            copy_header=True,
                            force_resample=True,
                        ).to_filename(dst)

            processed[(space, res, desc)] = dst
            template_json[space][res][desc] = {
                "host": "github-nispace",
                "remote": f"template/{space}/map/{desc}/{dst.name}",
            }

# %% fsLR — all densities from neuromaps, stored in nispace-data

print("\n=== fsLR ===")
template_json["fsLR"] = {}
for density in FSLR_DENSITIES:
    print(f"\nfsLR {density}:")
    atlas = neuromaps_surface_files("fsLR", density)
    atlas["regfusion_space-MNI152NLin6Asym"] = write_regfusion_meshes("fsLR", density, atlas["sphere"])
    template_json["fsLR"][density] = copy_and_register("fsLR", density, atlas)

# %% fsaverage — all densities from neuromaps, stored in nispace-data

print("\n=== fsaverage ===")
template_json["fsaverage"] = {}
for density in FSAVERAGE_DENSITIES:
    print(f"\nfsaverage {density}:")
    atlas = neuromaps_surface_files("fsaverage", density)
    atlas["regfusion_space-MNI152NLin6Asym"] = write_regfusion_meshes("fsaverage", density, atlas["sphere"])
    template_json["fsaverage"][density] = copy_and_register("fsaverage", density, atlas)

# %% Write template.yaml per space
# build_datalib.py runs automatically via pre-commit hook on nispace-data commit

for space, resolutions in template_json.items():
    yaml_path = nispace_data_path / "template" / space / "template.yaml"
    cfg = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {"name": space}
    cfg["resolutions"] = resolutions
    yaml_path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True))
    print(f"Written → {yaml_path}")
