# %% Init

import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

nispace_source_data_path = wd

# Import first — nispace.datasets resets NISPACE_DATA_DIR at import time (line 20)
from nispace.datasets import fetch_reference, fetch_example
from nispace.utils.utils import parc_vect_to_vol
from nispace.io import parcellate_data

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, load_parc, load_parc_labels, save_csv_gz

# Override AFTER import so local source data is used throughout
os.environ["NISPACE_DATA_DIR"] = str(nispace_source_data_path)

# parcellations
PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
print("PARCS:", PARCS)

# %% Build ENIGMA acAN signal volume (MNI152NLin2009cAsym)
# enigmathick/surf: estimated cortical volume (DesikanKilliany) + SubVol (Aseg) for dx-an_subtype-acAN_pub-walton2022

SPACE = "MNI152NLin2009cAsym"
ENIGMA_AN = "dx-an_subtype-acAN_pub-walton2022"
parc_base_dir = nispace_source_data_path / "parcellation"
ref_base_path = f"{wd}/reference/%s/tab/dset-%s_parc-%s.csv.gz"

enigma_dk = (
    pd.read_csv(ref_base_path % ("enigmathick", "enigmathick", "DesikanKilliany"), index_col=0).loc[[ENIGMA_AN]]
    * pd.read_csv(ref_base_path % ("enigmaarea", "enigmaarea", "DesikanKilliany"), index_col=0).loc[[ENIGMA_AN]]
)
enigma_aseg = pd.read_csv(ref_base_path % ("enigmathick", "enigmathick", "Aseg"), index_col=0).loc[[ENIGMA_AN]]


def get_mni_labels_and_img(parc_name, space=SPACE):
    d = parc_base_dir / parc_name / space
    labels = (d / f"parc-{parc_name}_space-{space}.label.txt").read_text().splitlines()
    img_path = str(d / f"parc-{parc_name}_space-{space}.label.nii.gz")
    return labels, img_path


dk_labels, dk_img = get_mni_labels_and_img("DesikanKilliany")
aseg_labels, aseg_img = get_mni_labels_and_img("Aseg")

# Reorder enigma values to match parcel integer-index order (as in the parcellation NIfTI)
dk_vals = enigma_dk.reindex(columns=dk_labels).fillna(0).values.squeeze().astype(float)
aseg_vals = enigma_aseg.reindex(columns=aseg_labels).fillna(0).values.squeeze().astype(float)

# Convert to MNI152 volumes and combine (DK = cortex, Aseg = subcortex, non-overlapping)
vol_dk = parc_vect_to_vol(dk_vals, dk_img)
vol_aseg = parc_vect_to_vol(aseg_vals, aseg_img)
an_signal_vol = nib.Nifti1Image(
    (vol_dk.get_fdata() + vol_aseg.get_fdata()).astype(np.float32),
    vol_dk.affine, vol_dk.header,
)


# %% EXAMPLE: ANOREXIA NERVOSA ---------------------------------------------------------------------
# Simulated acAN patients vs healthy controls.
# AN cases are based on GM probability + ENIGMA acAN effect (Cohen's d, negative = depletion) + noise.
# HC controls are GM probability + noise only.

name = "anorexianervosa"
for parc_name in PARCS:
    print(f"Simulating anorexia nervosa data for parcellation: {parc_name}")

    n_subs = 50
    rng = np.random.default_rng(42)

    # GM probability map as shared brain base
    tab_gm = pd.read_csv(ref_base_path % ("tpm", "tpm", parc_name), index_col=0).loc[["tissue-gm_pub-spm"]]
    gm_base = np.nan_to_num(tab_gm.values.squeeze(), nan=0.0)

    # Reparcellate ENIGMA acAN signal volume into this parcellation
    parc_space = SPACE if (parc_base_dir / parc_name / SPACE).exists() else "fsLR"
    parc = load_parc(wd, parc_name, parc_space)
    target_labels = load_parc_labels(wd, parc_name, parc_space)
    signal_df = parcellate_data(
        data=[an_signal_vol],
        data_space=SPACE,
        parcellation=parc,
        parc_labels=target_labels,
        parc_space=parc_space,
        ignore_background_data=False,
        drop_background_parcels=False,
        verbose=False,
    )

    # Align signal with tab_gm column order
    an_signal = signal_df.iloc[0].reindex(tab_gm.columns).fillna(0).values

    noise_scale_an = gm_base.std() * 1.0
    noise_scale_hc = gm_base.std() * 1.0

    data_an = pd.DataFrame(
        columns=tab_gm.columns,
        index=[f"sub-{i:03d}AN" for i in range(1, n_subs + 1)]
              + [f"sub-{i:03d}HC" for i in range(n_subs + 1, n_subs * 2 + 1)],
    )

    for i in range(n_subs):
        signal_strength = rng.uniform(0.1, 0.3)
        # AN: GM base + ENIGMA acAN Cohen's d pattern (negative = depletion) + more noise
        data_an.iloc[i] = (
            gm_base + signal_strength * an_signal
            + rng.standard_normal(len(gm_base)) * noise_scale_an
        )
        # HC: GM base + less noise
        data_an.iloc[i + n_subs] = (
            gm_base + rng.standard_normal(len(gm_base)) * noise_scale_hc
        )

    save_csv_gz(data_an, nispace_source_data_path / "example" / name / f"example-{name}_parc-{parc_name}.csv.gz")


# %% Test Anorexia Nervosa Data

# from nispace import NiSpace

# coloc = "spearman"
# group_comparison = "cohen(a,b)"
# parc_name = "Desikan"
# n_subs = 50

# nsp = NiSpace(
#     x=fetch_reference("pet", collection="UniqueTracers", parcellation=parc_name),
#     y=fetch_example("anorexianervosa", parcellation=parc_name, check_file_hash=False),
#     parcellation=parc_name,
#     n_proc=8,
# ).fit()

# nsp.transform_y(group_comparison, groups=[0] * n_subs + [1] * n_subs)
# nsp.colocalize(coloc, Y_transform=group_comparison)
# nsp.permute("groups", coloc, Y_transform=group_comparison, n_perm=1000)
# nsp.correct_p()
# nsp.plot(
#     method=coloc,
#     Y_transform=group_comparison,
#     permute_what="groups",
# )


# %%
