# %% Init

import numpy as np
import pandas as pd
from pathlib import Path
import sys

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# import NiSpace functions
from nispace.datasets import reference_lib
from nispace.io import parcellate_data

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, load_parc, load_parc_labels, DATASET_PARCELLATE_KWARGS

# nispace data path
nispace_source_data_path = wd

# datasets with maps
DSETS_WITH_MAPS = [k for k, v in reference_lib.items() if "map" in v]
print("DSETS_WITH_MAPS: ", DSETS_WITH_MAPS)

# parcellations
PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
print("PARCS:", PARCS)

# parc_space / ref_space pairs per dataset
DATASET_SPACE_PAIRS = {
    "pet":            [("MNI152NLin6Asym", "MNI152NLin6Asym"), ("fsaverage", "fsaverageOriginal")],
    "rsn":            [("MNI152NLin6Asym", "MNI152NLin6Asym")],
    "cortexfeatures": [("fsLR", "fsLROriginal"), ("fsaverage", "fsaverageOriginal")],
    "bigbrain":       [("fsaverage", "fsaverageOriginal")],
    "tpm":            [("MNI152NLin6Asym", "MNI152NLin6Asym")],
}

# function to fetch reference map paths from local source
def fetch_reference(dataset, maps, space):
    map_paths = []
    map_dir = nispace_source_data_path / "reference" / dataset / "map"
    for m in maps:
        fp = sorted((map_dir / m).glob(f"{m}_space-{space}_*"))
        if len(fp) == 0:
            fp = sorted((map_dir / m).glob(f"{m}_space-{space}.*"))
        if len(fp) == 0:
            raise ValueError(f"No map found for {m} in {space}")
        elif len(fp) == 1:
            fp = fp[0]
        elif len(fp) == 2:
            fp = tuple(fp)
        elif len(fp) > 2:
            raise ValueError(f"More than two maps found for {m} in {space}: {fp}")
        map_paths.append(fp)
    return map_paths


# %% Parcellate map-based image data ---------------------------------------------------------------

for dataset in DSETS_WITH_MAPS:
    print("-------- " + dataset.upper() + " --------")

    if dataset not in DATASET_SPACE_PAIRS:
        print(f"No space pairs defined for dataset: {dataset} — skipping")
        continue

    # collect available maps per (original) space
    ref_maps = {}
    for space in ["MNI152", "MNI152NLin6Asym", "MNI152NLin2009cAsym",
                  "fsaverageOriginal", "fsaverage",
                  "fsLROriginal", "fsLR"]:
        maps_in_space = []
        for m in reference_lib[dataset]["map"].keys():
            if space not in reference_lib[dataset]["map"][m]:
                continue
            # skip private maps
            entry = reference_lib[dataset]["map"][m][space]
            private = False
            if "host" in entry:
                private = "private" in entry["host"]
            elif "L" in entry:
                private = "private" in entry["L"]["host"]
            elif "R" in entry:
                private = "private" in entry["R"]["host"]
            if not private:
                maps_in_space.append(m)
        if maps_in_space:
            ref_maps[space] = maps_in_space
            print(f"  {space}: {len(maps_in_space)} maps")
        else:
            print(f"  {space}: no maps")

    # unique maps across all spaces (preserve reference_lib order)
    ref_maps_avail_all = [m for m in reference_lib[dataset]["map"]
                          if any(m in v for v in ref_maps.values())]
    print(f"-> {len(ref_maps_avail_all)} unique maps")

    # iterate parcellations
    print("Parcellating...")
    for parc_name in PARCS:
        print(parc_name)

        # labels — all parcs have MNI6
        labels = load_parc_labels(wd, parc_name, "MNI152NLin6Asym")

        # initialise output dataframe
        ref_maps_df = pd.DataFrame(
            index=pd.Index(list(ref_maps_avail_all), name="map"),
            columns=labels,
        )
        print("  Pre-parcellation shape:", ref_maps_df.shape)

        for parc_space, ref_space in DATASET_SPACE_PAIRS[dataset]:
            # skip if this space is not present for this parcellation
            if not (nispace_source_data_path / "parcellation" / parc_name / parc_space).exists():
                continue
            # skip if no ref maps exist in this space
            if ref_space not in ref_maps:
                continue
            ref_maps_avail = ref_maps[ref_space]

            # load parcellation
            parc = load_parc(wd, parc_name, parc_space)
            print(f"  Parcellation loaded for space {parc_space}")

            # load reference map paths (local fetch)
            # space=parc_space is CORRECT here:
            # ref_space serves to filter maps, but they are loaded from disc in parc_space
            ref_paths = fetch_reference(dataset, maps=ref_maps_avail, space=parc_space)
            print(f"  -> {len(ref_paths)} maps for space {parc_space}, filtered by {ref_space}")

            # parcellate
            # bilateral case: MNI or all surface maps have two hemispheres
            if ("MNI152" in parc_space) or (
                "MNI152" not in parc_space and all([len(fp) == 2 for fp in ref_paths])
            ):
                parc_list   = [parc]
                labels_list = [labels]
                hemi_list   = [None]
                paths_list  = [ref_paths]
                avail_list  = [ref_maps_avail]
            # mixed: some surface maps are single-hemisphere
            else:
                parc_list   = [parc, parc[0], parc[1]]
                labels_list = [
                    labels,
                    [l for l in labels if "hemi-L" in l],
                    [l for l in labels if "hemi-R" in l],
                ]
                hemi_list  = [None, "L", "R"]
                paths_list = [
                    [fp for fp in ref_paths if len(fp) == 2],
                    [fp for fp in ref_paths if len(fp) == 1 and "hemi-L" in fp[0].name],
                    [fp for fp in ref_paths if len(fp) == 1 and "hemi-R" in fp[0].name],
                ]
                avail_list = [
                    [fp[0].parent.name for fp in paths_list[0]],
                    [fp[0].parent.name for fp in paths_list[1]],
                    [fp[0].parent.name for fp in paths_list[2]],
                ]

            for p, p_l, p_h, r_p, r_m_a in zip(
                parc_list, labels_list, hemi_list, paths_list, avail_list
            ):
                if len(r_p) == 0:
                    continue

                tab = parcellate_data(
                    parcellation=p,
                    parc_hemi=p_h,
                    parc_labels=p_l,
                    parc_space=parc_space,
                    data=r_p,
                    data_labels=r_m_a,
                    data_space=parc_space, # this is correct, see comment above
                    n_proc=-1,
                    dtype=np.float32,
                    **DATASET_PARCELLATE_KWARGS[dataset],
                )

                print(f"  Parcellated: {tab.shape}")
                ref_maps_df.loc[r_m_a, tab.columns] = tab

        # save
        print(ref_maps_df.head())
        save_dir = nispace_source_data_path / "reference" / dataset / "tab"
        save_dir.mkdir(parents=True, exist_ok=True)
        ref_maps_df.to_csv(save_dir / f"dset-{dataset}_parc-{parc_name}.csv.gz")
        print("-" * 80)

# %%
