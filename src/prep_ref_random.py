# %% Init

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import nibabel as nib
from scipy import fftpack
from joblib import Parallel, delayed
from tqdm import tqdm

# add nispace to path
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_parcellation, fetch_template, parcellation_lib
from nispace.io import parcellate_data

# nispace data path 
nispace_source_data_path = wd

# all parcellations (original, not aliases)
PARCS = [k for k in parcellation_lib.keys() if "alias" not in parcellation_lib[k]]
print("PARCS:", PARCS)
# %% Functions to generate random voxel-wise maps
# Copied from: https://github.com/netneurolab/markello_spatialnulls/blob/master/parspin/parspin/spatial.py
# Cite: https://doi.org/10.1016/j.neuroimage.2021.118052

def _fftind(x, y, z):
    """
    Return 3D shifted Fourier coordinates

    Returned coordinates are shifted such that zero-frequency component of the
    square grid with shape (x, y, z) is at the center of the spectrum

    Parameters
    ----------
    {x,y,z} : int
        Size of array to be generated

    Returns
    -------
    k_ind : (3, x, y, z) np.ndarray
        Shifted Fourier coordinates, where:
            k_ind[0] : k_x components
            k_ind[1] : k_y components
            k_ind[2] : k_z components

    Notes
    -----
    See scipy.fftpack.fftshift

    References
    ----------
    Based on code from Burt et al., 2020, NeuroImage
    """

    k_ind = np.mgrid[:x, :y, :z]
    zero = np.array([int((n + 1) / 2) for n in [x, y, z]])
    while zero.ndim < k_ind.ndim:
        zero = np.expand_dims(zero, -1)
    k_ind = fftpack.fftshift(k_ind - zero)

    return k_ind


def gaussian_random_field(x, y, z, noise=None, alpha=3.0, normalize=True,
                          seed=None):
    """
    Generate a Gaussian random field with k-space power law |k|^(-alpha/2).

    Parameters
    ----------
    {x,y,z} : int
        Grid size of generated field
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed `noise` generation. Default: None

    Returns
    -------
    gfield : (x, y, z) np.ndarray
        Realization of Gaussian random field

    References
    ----------
    Based on code from Burt et al., 2020, NeuroImage
    """

    rs = np.random.default_rng(seed)

    if not alpha:
        return rs.normal(size=(x, y, z))

    assert alpha > 0

    # k-space indices
    k_idx = _fftind(x, y, z)

    # define k-space amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(np.sum([k ** 2 for k in k_idx], axis=0) + 1e-10,
                         -alpha / 2.0)
    amplitude[0, 0, 0] = 0  # remove zero-freq mean shit

    # generate a complex gaussian random field where phi = phi_1 + i*phi_2
    if noise is None:
        noise = rs.normal(size=(x, y, z))
    elif noise.shape != (x, y, z):
        try:
            noise = noise.reshape(x, y, z)
        except ValueError:
            raise ValueError('Provided noise cannot be reshape to target: '
                             f'({x}, {y}, {z})')

    # transform back to real space
    gfield = np.fft.ifftn(np.fft.fftn(noise) * amplitude).real

    if normalize:
        return (gfield - gfield.mean()) / gfield.std()

    return gfield

# %% Function to generate random field maps

# reference space: 2mm MNI152NLin2009cAsym
mask_img = nib.load(fetch_template("MNI152NLin2009cAsym", res="2mm", desc="mask"))

# function to generate spatially-autocorrelated random field maps
def generate_grf(alpha, seed, mirrored=True, mask=mask_img, dtype=np.float32):
    
    # seed
    rng = np.random.default_rng(seed)
    
    # generate 3d gaussian random field
    grf = gaussian_random_field(
        *mask_img.shape, # generate 3d random field with dimensions of template image
        alpha=alpha, # smoothness of the field
        noise=None, # no starting data, function will generate random vector to add gaussian random fields to
        normalize=True, # normalize to unit variance
        seed=rng
    )
    
    if mirrored:
        idx = grf.shape[0] // 2
        grf_mirr = np.zeros(grf.shape, dtype=grf.dtype)
        grf_mirr[:idx, :, :] = grf[:idx, :, :]
        grf_mirr[idx+1:, :, :] = grf[:idx, :, :][::-1, :, :]
        grf = grf_mirr
        
    if mask:
        grf *= mask_img.get_fdata()
    
    # to nifti image
    grf_mni = nib.Nifti1Image(grf, mask_img.affine, dtype=dtype)
    
    # return
    return grf_mni


# %% Run

# alphas
alphas = [0.0, 1.0, 2.0, 3.0]

# number of random field maps per alpha
n_grf = {0.0: 10000, 1.0: 1000, 2.0: 1000, 3.0: 1000}

# batch size
batch_size = {0.0: 1000, 1.0: 1000, 2.0: 1000, 3.0: 1000}

# dict to save data
data_grf = {parc: {alpha: [] for alpha in alphas} for parc in PARCS}

# Iterate over alphas
for alpha in alphas:
    
    # iterate over batches
    for i in range(0, n_grf[alpha], batch_size[alpha]):
        
        # Generate random field maps
        grf_volumes = Parallel(n_jobs=-1)(
            delayed(generate_grf)
            (alpha=alpha, seed=i_seed) 
            for i_seed in tqdm(range(i, i+batch_size[alpha]), desc=f"Alpha {alpha} - Batch {i//batch_size[alpha]+1}")
        )
        
        # Parcellate random field maps
        for parc in PARCS:
            print(parc)
            
            # fetch parcellation
            parc_space = "MNI152NLin2009cAsym"
            resampling_target = "data"
            parc_img, parc_labels = fetch_parcellation(parc, space=parc_space, return_loaded=True)
            
            # parcellate
            grf_parc = parcellate_data(
                data=grf_volumes, 
                data_labels=[f"alpha-{alpha}_seed-{i_seed}" for i_seed in range(i, i+batch_size[alpha])], 
                data_space="mni152", 
                parcellation=parc_img, 
                parc_labels=parc_labels, 
                parc_space=parc_space, 
                resampling_target=resampling_target, 
                drop_background_parcels=False, 
                min_num_valid_datapoints=None, 
                min_fraction_valid_datapoints=None, 
                return_parc=False, 
                dtype=np.float16, 
                n_proc=-1, 
                verbose=False
            )
            
            # save
            data_grf[parc][alpha].append(grf_parc)
            
            
# to final dfs
for parc in PARCS:
    df = pd.concat(
        [
            pd.concat(data_grf[parc][alpha])
            for alpha in alphas
        ]
    )
    df = df.astype(np.float16)
    df.index.name = "map"
    df.to_csv(nispace_source_data_path / "reference" / "grf" / "tab" / f"dset-grf_parc-{parc}.csv.gz")

# %%
