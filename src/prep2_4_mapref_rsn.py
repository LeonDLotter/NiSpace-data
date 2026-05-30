# %% Init

import zipfile
from pathlib import Path
import nibabel as nib
import pandas as pd
from nilearn import image
from neuromaps import transforms
wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.datasets import fetch_template
from nispace.utils.utils import apply_transform
from nispace.utils.utils_datasets import download

nispace_source_data_path = wd


# %% Download RSN maps from Dworetsky et al. 2021
# source: https://github.com/GrattonLab/Dworetsky_etal_ConsensusNetworks
# paper: https://doi.org/10.1016/j.neuroimage.2020.117678

url = "https://raw.githubusercontent.com/GrattonLab/Dworetsky_etal_ConsensusNetworks/4521ead/Probabilistic_Network_Maps_t88_333.zip"
archive_dir = nispace_source_data_path / "_archive" / "rsn"
archive_dir.mkdir(parents=True, exist_ok=True)
fp_zip = archive_dir / "Probabilistic_Network_Maps_t88_333.zip"
download(url, fp_zip)
with zipfile.ZipFile(fp_zip, "r") as z:
    z.extractall(archive_dir)


# %% Process each network map

template_MNI6 = fetch_template("MNI152NLin6Asym", desc="mask", res="2mm")
mask_MNI6 = fetch_template("MNI152NLin6Asym", desc="mask_gm", res="2mm")
mask_MNI9 = fetch_template("MNI152NLin2009cAsym", desc="mask_gm", res="2mm")

for nii_fp in sorted(archive_dir.rglob("*.nii")):
    if nii_fp.name == "711-2B_333.nii":
        continue
    network_name = nii_fp.stem.replace("_", "").replace("-", "")
    map_id = f"nw-{network_name}_pub-dworetsky2021"
    print(f"Processing: {map_id}")

    map_dir = nispace_source_data_path / "reference" / "rsn" / "map" / map_id
    map_dir.mkdir(parents=True, exist_ok=True)

    # space: MNI152 — save original
    img = nib.load(nii_fp)
    if len(img.shape) > 3:
        img = image.index_img(img, 0)
    nib.save(img, map_dir / f"{map_id}_space-MNI152.nii.gz")

    # space: MNI152NLin6Asym — resample to 2mm (maps are already in this space)
    # also apply mask and scale to [0, 1]
    map_MNI6 = image.resample_to_img(img, template_MNI6, interpolation="continuous",
                                     force_resample=True, copy_header=True)
    map_MNI6 = image.math_img("(img * mask / 100).astype(np.float32)", img=map_MNI6, mask=mask_MNI6)
    map_MNI6.to_filename(map_dir / f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz")

    # space: MNI152NLin2009cAsym
    # transform from MNI6 to MNI2009c, apply mask and scale to [0, 1]
    map_2009c = apply_transform(map_MNI6, mni_from="MNI152NLin6Asym", mni_to="MNI152NLin2009cAsym", order=3)
    map_2009c = image.math_img("(img * mask / 100).astype(np.float32)", img=map_2009c, mask=mask_MNI9)
    map_2009c.to_filename(map_dir / f"{map_id}_space-MNI152NLin2009cAsym_desc-proc.nii.gz")

    # space: fsLR 32k
    map_fsLR = transforms.mni152_to_fslr(map_MNI6, fslr_density="32k", method="linear")
    map_fsLR[0].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-L.surf.gii.gz")
    map_fsLR[1].to_filename(map_dir / f"{map_id}_space-fsLR_desc-proc_hemi-R.surf.gii.gz")

    # space: fsaverage 41k
    map_fsavg = transforms.mni152_to_fsaverage(map_MNI6, fsavg_density="41k", method="linear")
    map_fsavg[0].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-L.surf.gii.gz")
    map_fsavg[1].to_filename(map_dir / f"{map_id}_space-fsaverage_desc-proc_hemi-R.surf.gii.gz")

    print(f"  Saved all spaces for {map_id}")


# %% Collections

ref_dir = nispace_source_data_path / "reference" / "rsn"
maps = sorted([d.name for d in (ref_dir / "map").iterdir() if d.is_dir()])
pd.Series(maps, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

# %%
