# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, load_parc, load_parc_labels, save_csv_gz
from nispace.utils.utils_datasets import download

nispace_source_data_path = wd
dataset = "celltypes-zhang2025"
ref_dir = nispace_source_data_path / "reference" / dataset


# %% Source data
# source: Zhang et al., 2025, Nat Neurosci — https://doi.org/10.1038/s41593-024-01812-2
# data: https://github.com/XihanZhang/human-cellular-func-con (no license file; used with the
# authors' permission: _archive/usage_permission/Zhang2025_permission.pdf)
# AHBA cortical bulk samples, cell-type fractions imputed with CIBERSORTx using the Jorstad et al.
# (2023) snRNA-seq reference (24 cell types). Samples are mapped to fsLR 32k vertices.
# We recompute the parcel tables from the sample-level data with our own fsLR parcellations,
# following the authors' approach: samples > 4 mm from the surface are excluded (1,676 remaining
# samples, as in the paper), then all samples within a parcel are averaged (pooled across donors).
# With these rules, the authors' published Schaefer tables are reproduced exactly (Schaefer400:
# 339/339 parcels using the CBIG 2017 fsLR file; Schaefer100: 99/99 using the CBIG 2020 file).
# In addition, samples are mirrored across hemispheres (fsLR vertex i in L <-> vertex i in R), as
# done for the mrna dataset via abagen (lr_mirror="bidirectional"): only 2/6 AHBA donors have
# right-hemisphere samples.

GH_BASE_URL = ("https://raw.githubusercontent.com/XihanZhang/human-cellular-func-con/"
               "3ba46aa0b1b031d171d331ab185c475792bdda57/cell_maps/vertex_level")
N_VERTICES_HEMI = 32492
MAX_MM_TO_SURF = 4

# original Jorstad subclass : map id cell name (subclass name without spaces/slashes)
CELL_NAMES = {
    # excitatory
    "L2/3 IT":    "L23IT",
    "L4 IT":      "L4IT",
    "L5 IT":      "L5IT",
    "L6 IT":      "L6IT",
    "L6 IT Car3": "L6ITCar3",
    "L5 ET":      "L5ET",
    "L5/6 NP":    "L56NP",
    "L6 CT":      "L6CT",
    "L6b":        "L6b",
    # inhibitory
    "Lamp5":      "Lamp5",
    "Lamp5 Lhx6": "Lamp5Lhx6",
    "Sncg":       "Sncg",
    "Vip":        "Vip",
    "Pax6":       "Pax6",
    "Chandelier": "Chandelier",
    "Pvalb":      "Pvalb",
    "Sst":        "Sst",
    "Sst Chodl":  "SstChodl",
    # non-neuronal
    "Astro":      "Astro",
    "Oligo":      "Oligo",
    "OPC":        "OPC",
    "Micro/PVM":  "MicroPVM",
    "Endo":       "Endo",
    "VLMC":       "VLMC",
}
# Jorstad class of each subclass, in the order of CELL_NAMES
CELL_CLASSES = ["excitatory"] * 9 + ["inhibitory"] * 9 + ["nonneuronal"] * 6
assert len(CELL_CLASSES) == len(CELL_NAMES)
MAP_IDS = {
    cell: f"cell-{name}_class-{cls}_pub-zhang2025"
    for (cell, name), cls in zip(CELL_NAMES.items(), CELL_CLASSES)
}
COLLECTIONS = {"Excitatory": "excitatory", "Inhibitory": "inhibitory", "NonNeuronal": "nonneuronal"}


# %% Load sample-level data

fractions = pd.read_csv(download(f"{GH_BASE_URL}/Jorstad_cell_type_fractions_with_vertexfsLR32k_MNI.csv"))
sample_info = pd.read_csv(download(f"{GH_BASE_URL}/sample_info_vertex_reannot_mapped_0.3.csv"))

cells = list(CELL_NAMES)
assert set(cells) <= set(fractions.columns), set(cells) - set(fractions.columns)

# vertex: full 64984-vertex fsLR index (L: 0..32491, R: 32492..64983)
samples = fractions[["well_id", "vertex"] + cells].merge(
    sample_info[["well_id", "brain", "mm_to_surf"]], on="well_id", how="left", validate="1:1"
)
assert samples.brain.notna().all()
print(f"Samples: {len(samples)}")

# exclude samples > 4 mm from the surface (authors' criterion)
samples = samples[samples.mm_to_surf.abs() < MAX_MM_TO_SURF].reset_index(drop=True)
print(f"Samples within {MAX_MM_TO_SURF} mm of surface: {len(samples)}")
assert len(samples) == 1676

is_rh = samples.vertex >= N_VERTICES_HEMI
print(f"  L: {(~is_rh).sum()} | R: {is_rh.sum()} | donors: {samples.brain.nunique()}")

# mirror: copy every sample to the corresponding vertex of the contralateral hemisphere
samples_mirrored = samples.assign(
    vertex=np.where(is_rh, samples.vertex - N_VERTICES_HEMI, samples.vertex + N_VERTICES_HEMI)
)
samples = pd.concat([samples, samples_mirrored], ignore_index=True)
print(f"Samples after L/R mirroring: {len(samples)}")


# %% Parcellate

_, PARCS_CX, _ = load_parc_lists(wd)
# drop the 1000-parcel versions: too few samples (median 1/parcel, >14% empty parcels after mirroring)
parc_names = [p for p in PARCS_CX if "1000" not in p]

tab_dir = ref_dir / "tab"
tab_dir.mkdir(parents=True, exist_ok=True)

coverage = []
for parc_name in parc_names:
    # fsLR parcel index i corresponds to the i-th label (L then R), identical across spaces
    labels = load_parc_labels(wd, parc_name, "MNI152NLin6Asym")
    assert np.array_equal(labels, load_parc_labels(wd, parc_name, "fsLR")), f"{parc_name}: MNI and fsLR labels differ"

    parc_l, parc_r = load_parc(wd, parc_name, "fsLR")
    parc_l, parc_r = parc_l.agg_data().astype(int), parc_r.agg_data().astype(int)
    n_l = int((np.char.find(labels, "hemi-L") == 0).sum())
    assert len(parc_l) == len(parc_r) == N_VERTICES_HEMI
    assert (np.unique(parc_l[parc_l > 0]) == np.arange(1, n_l + 1)).all(), parc_name
    assert (np.unique(parc_r[parc_r > 0]) == np.arange(n_l + 1, len(labels) + 1)).all(), parc_name
    parc = np.concatenate([parc_l, parc_r])

    # average all samples within each parcel
    tab = (
        samples
        .assign(parcel=parc[samples.vertex.values])
        .query("parcel > 0")
        .groupby("parcel")[cells].mean()
        .reindex(range(1, len(labels) + 1))
        .T
    )
    tab.index = pd.Index([MAP_IDS[c] for c in tab.index], name="map")
    tab.columns = labels
    tab = tab.astype(np.float32)

    n_nan = int(tab.isna().all(axis=0).sum())
    coverage.append(dict(parc=parc_name, n_parcels=len(labels), n_nan=n_nan,
                         pct_nan=round(100 * n_nan / len(labels), 1)))
    print(f"  [{parc_name}] shape {tab.shape} | {n_nan} all-NaN parcels")

    save_csv_gz(tab, tab_dir / f"dset-{dataset}_parc-{parc_name}.csv.gz")

print(pd.DataFrame(coverage).to_string(index=False))


# %% Collections

pd.Series(list(MAP_IDS.values()), name="map").to_csv(ref_dir / "collection-All.collect", index=False)
for collection, cls in COLLECTIONS.items():
    pd.Series([m for m in MAP_IDS.values() if f"_class-{cls}_" in m], name="map") \
        .to_csv(ref_dir / f"collection-{collection}.collect", index=False)

# %%
