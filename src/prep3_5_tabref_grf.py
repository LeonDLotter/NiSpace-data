# %% Init

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import nibabel as nib
from scipy import fftpack
from joblib import Parallel, delayed
from tqdm import tqdm
from nilearn.image import math_img

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.parcellate import Parcellater

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, load_parc, load_parc_labels, DATASET_PARCELLATE_KWARGS

# nispace data path
nispace_source_data_path = wd

# external drive where raw GRF NIfTI maps are stored
grf_data_path = Path("/Volumes/data_m2_2tb/data/grf")
if not grf_data_path.exists():
    raise FileNotFoundError(f"GRF data path not found: {grf_data_path}")

# parcellations
PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
print("PARCS:", PARCS)

# %% Functions to generate random voxel-wise maps
# Copied from: https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/spatial.py
# Cite: https://doi.org/10.1016/j.neuroimage.2021.118052

def _fftind(x, y, z):
    k_ind = np.mgrid[:x, :y, :z]
    zero = np.array([int((n + 1) / 2) for n in [x, y, z]])
    while zero.ndim < k_ind.ndim:
        zero = np.expand_dims(zero, -1)
    k_ind = fftpack.fftshift(k_ind - zero)
    return k_ind


def gaussian_random_field(x, y, z, noise=None, alpha=3.0, normalize=True, seed=None):
    """Generate a Gaussian random field with k-space power law |k|^(-alpha/2).

    Based on code from Burt et al., 2020, NeuroImage.
    """
    rs = np.random.default_rng(seed)

    if not alpha:
        return rs.normal(size=(x, y, z))

    assert alpha > 0

    k_idx = _fftind(x, y, z)
    amplitude = np.power(
        np.sum([k ** 2 for k in k_idx], axis=0) + 1e-10, -alpha / 2.0
    )
    amplitude[0, 0, 0] = 0

    if noise is None:
        noise = rs.normal(size=(x, y, z))
    elif noise.shape != (x, y, z):
        try:
            noise = noise.reshape(x, y, z)
        except ValueError:
            raise ValueError(f"Provided noise cannot be reshaped to ({x}, {y}, {z})")

    gfield = np.fft.ifftn(np.fft.fftn(noise) * amplitude).real

    if normalize:
        return (gfield - gfield.mean()) / gfield.std()
    return gfield


# %% Setup

# reference space: 2mm MNI152NLin2009cAsym
mask_img = nib.load(wd / "template" / "MNI152NLin2009cAsym" / "map" / "mask" / "tpl-MNI152NLin2009cAsym_desc-mask_res-2mm.nii.gz")

# generate + mask one GRF and return a float32 NIfTI
def generate_grf(alpha, seed, mirrored=True):
    rng = np.random.default_rng(seed)
    grf = gaussian_random_field(*mask_img.shape, alpha=alpha, normalize=True, seed=rng)
    if mirrored:
        idx = grf.shape[0] // 2
        grf_mirr = np.zeros(grf.shape, dtype=grf.dtype)
        grf_mirr[:idx]    = grf[:idx]
        grf_mirr[idx+1:]  = grf[:idx][::-1]
        grf = grf_mirr
    grf *= mask_img.get_fdata()
    return nib.Nifti1Image(grf.astype(np.float32), mask_img.affine)


# settings
alphas = [0.0, 1.0, 2.0, 3.0]
n_grf  = 1000   # equal count for all alpha levels


# %% Phase 1 — generate and save GRF maps to external drive

print("=== Phase 1: generating GRF maps ===")
grf_data_path.mkdir(parents=True, exist_ok=True)


def _gen_and_save(alpha, seed):
    out = grf_data_path / f"alpha-{alpha:.1f}_seed-{seed:04d}.nii.gz"
    if out.exists():
        return
    img = generate_grf(alpha=alpha, seed=seed)
    math_img("img.astype(np.float32)", img=img).to_filename(out)


for alpha in alphas:
    print(f"Alpha {alpha} ...")
    Parallel(n_jobs=-1)(
        delayed(_gen_and_save)(alpha, seed)
        for seed in tqdm(range(n_grf), desc=f"alpha={alpha}")
    )


# %% Phase 2 — parcellate from disk using Parcellater

print("=== Phase 2: parcellating GRF maps ===")

# ordered list of all map paths and their labels
map_paths  = [grf_data_path / f"alpha-{a:.1f}_seed-{s:04d}.nii.gz"
              for a in alphas for s in range(n_grf)]
map_labels = [f"alpha-{a:.1f}_seed-{s:04d}" for a in alphas for s in range(n_grf)]

parc_space = "MNI152NLin2009cAsym"

for parc_name in PARCS:
    print(parc_name)

    out_path = nispace_source_data_path / "reference" / "grf" / "tab" / f"dset-grf_parc-{parc_name}.csv.gz"
    if out_path.exists():
        print("  Already exists — skipping")
        continue

    parc_img = load_parc(wd, parc_name, parc_space)
    labels   = load_parc_labels(wd, parc_name, parc_space)

    parcellater = Parcellater(
        parcellation=parc_img,
        space=parc_space,
        resampling_target="data",
    ).fit()

    def _parcellate(map_path):
        return parcellater.transform(
            data=map_path,
            space=parc_space,
            **DATASET_PARCELLATE_KWARGS["grf"],
        ).astype(np.float16)

    data = Parallel(n_jobs=-1)(
        delayed(_parcellate)(mp)
        for mp in tqdm(map_paths, desc=parc_name)
    )

    df = pd.DataFrame(
        np.stack(data).astype(np.float16),
        index=pd.Index(map_labels, name="map"),
        columns=labels,
        dtype=np.float16,
    )
    df.to_csv(out_path)
    print(f"  Saved {df.shape} → {out_path.name}")


# %% Collections

from nispace.io import write_json

ref_dir = nispace_source_data_path / "reference" / "grf"

# All
pd.Series(map_labels, name="map").to_csv(ref_dir / "collection-All.collect", index=False)

# Alpha0 (no spatial autocorrelation)
all_grf = pd.Series(map_labels, name="map")
all_grf.loc[all_grf.str.contains("alpha-0.0")].to_csv(ref_dir / "collection-Alpha0.collect", index=False)

# ByAlpha
write_json(
    {f"alpha-{a}": [idx for idx in map_labels if idx.startswith(f"alpha-{a:.01f}")] for a in alphas},
    ref_dir / "collection-ByAlpha.collect",
)

# %%
