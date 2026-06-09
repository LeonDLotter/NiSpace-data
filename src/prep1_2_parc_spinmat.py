# %% Init

from pathlib import Path
import tempfile
import numpy as np
from neuromaps import images

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

from nispace.nulls import generate_cornblath_mat, _img_density_for_neuromaps

# local utils
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists

# nispace-data root
nispace_source_data_path = wd

# settings
N_PERM = 10000
SEED = 42
N_PROC = -1  # -1 = all available cores; gen_spinsamples is GIL-free (cKDTree), threads scale well

# all parcellations (aliases excluded)
PARCS, _, _ = load_parc_lists(wd)
print("PARCS:", PARCS)


# %% Generate Cornblath transition matrices

for parc in PARCS:
    print(f"\n{'='*60}\nParcellation: {parc}")
    spaces = sorted(
        [s.name for s in (nispace_source_data_path / "parcellation" / parc).glob("*") if s.is_dir()]
    )

    for space in spaces:
        # spin tests are surface-only
        if "mni" in space.lower():
            continue
        print(f"\n  Space: {space}")

        # load parcellation label GIFTIs
        parc_lh = images.load_gifti(
            nispace_source_data_path / "parcellation" / parc / space /
            f"parc-{parc}_space-{space}_hemi-L.label.gii.gz"
        )
        parc_rh = images.load_gifti(
            nispace_source_data_path / "parcellation" / parc / space /
            f"parc-{parc}_space-{space}_hemi-R.label.gii.gz"
        )
        n_lh = len(np.trim_zeros(np.unique(parc_lh.darrays[0].data)))
        n_rh = len(np.trim_zeros(np.unique(parc_rh.darrays[0].data)))
        print(f"  Parcels: {n_lh} LH + {n_rh} RH = {n_lh + n_rh} total")

        density = _img_density_for_neuromaps((parc_lh, parc_rh))
        print(f"  Density: {density}")

        # generate Cornblath fractional transition matrices
        # T_lh: (N_PERM, n_lh, n_lh), T_rh: (N_PERM, n_rh, n_rh), dtype float16
        print(f"  Generating {N_PERM} Cornblath transition matrices ...", end=" ", flush=True)
        with tempfile.TemporaryDirectory() as memmap_dir:
            T_lh, T_rh = generate_cornblath_mat(
                parc=(parc_lh, parc_rh),
                parc_space=space,
                n_perm=N_PERM,
                seed=SEED,
                n_proc=N_PROC,
                dtype=np.float16,
                memmap_dir=memmap_dir,
            )
            assert T_lh.shape == (N_PERM, n_lh, n_lh), f"T_lh shape mismatch: {T_lh.shape}"
            assert T_rh.shape == (N_PERM, n_rh, n_rh), f"T_rh shape mismatch: {T_rh.shape}"
            assert T_lh.dtype == np.float16
            assert T_rh.dtype == np.float16

            for mat, hemi in zip([T_lh, T_rh], ["L", "R"]):
                out_path = (
                    nispace_source_data_path / "parcellation" / parc / space /
                    f"parc-{parc}_space-{space}_hemi-{hemi}.spin.npz"
                )
                np.savez_compressed(str(out_path), data=mat)

        print(f"saved hemi-L {T_lh.shape}, hemi-R {T_rh.shape}")

print("\nDone.")

# %%
