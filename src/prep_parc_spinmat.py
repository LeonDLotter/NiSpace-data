# %% Init

import sys
from pathlib import Path
import numpy as np
from neuromaps import images
from neuromaps.datasets import fetch_fsaverage, fetch_fslr
from neuromaps.nulls.spins import gen_spinsamples, get_parcel_centroids

wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

from nispace.nulls import _img_density_for_neuromaps

# nispace-data root
nispace_source_data_path = wd

# settings
N_PERM = 10000
SEED = 42

# all parcellations
PARCS = sorted(
    [p.name for p in (nispace_source_data_path / "parcellation").glob("*") if p.is_dir()]
)
print("PARCS:", PARCS)


# %% Generate spin matrices

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

        # detect density and atlas
        density = _img_density_for_neuromaps((parc_lh, parc_rh))
        atlas = "fsaverage" if "fsa" in space.lower() else "fslr"
        print(f"  Atlas: {atlas}, density: {density}")

        # fetch sphere surfaces for centroid extraction
        if atlas == "fsaverage":
            surfaces = fetch_fsaverage(density=density)["sphere"]
        else:
            surfaces = fetch_fslr(density=density)["sphere"]

        # get parcel centroids on sphere
        coords, hemiid = get_parcel_centroids(
            surfaces,
            parcellation=(parc_lh, parc_rh),
            method="surface"
        )
        assert coords.shape[0] == n_lh + n_rh, f"Wrong number of parcels in centroid calculation: {n_lh + n_rh} != {coords.shape[0]}"
        assert int((hemiid == 0).sum()) == n_lh, f"LH parcel count mismatch: {n_lh} != {int((hemiid == 0).sum())}"

        # generate alexander_bloch spins ("original" = nearest-neighbor, fast)
        print(f"  Generating {N_PERM} spins (alexander_bloch) ...", end=" ", flush=True)
        spins = gen_spinsamples(
            coords, hemiid,
            n_rotate=N_PERM,
            method="original",
            seed=SEED,
            verbose=False,
        )  # shape: (n_lh + n_rh, N_PERM), dtype int

        # split per hemisphere; make RH indices local to [0, n_rh)
        spins_lh = spins[:n_lh, :].astype(np.int32)
        spins_rh = (spins[n_lh:, :] - n_lh).astype(np.int32)
        assert spins_lh.min() >= 0 and spins_lh.max() < n_lh
        assert spins_rh.min() >= 0 and spins_rh.max() < n_rh

        for mat, hemi in zip([spins_lh, spins_rh], ["L", "R"]):
            out_path = (
                nispace_source_data_path / "parcellation" / parc / space /
                f"parc-{parc}_space-{space}_hemi-{hemi}.spin.npy"
            )
            np.save(out_path, mat)

        print(f"saved {spins_lh.shape}, {spins_rh.shape}")

print("\nDone.")

# %%
