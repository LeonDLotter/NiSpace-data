"""Shared helpers for nispace-data prep scripts."""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from neuromaps import images as neuro_images
import templateflow.api as tflow


def tflow_get(*args, name_filter=None, **kwargs):
    """Call tflow.get() and enforce exactly one result.

    `name_filter` is an optional substring matched against the filename for
    entities (e.g. 'seg', 'scale') that pybids does not index as BIDS entities.
    """
    result = tflow.get(*args, **kwargs)
    if result is None:
        raise FileNotFoundError(f"tflow.get({args}, {kwargs}) returned nothing")
    if isinstance(result, list):
        if name_filter:
            result = [f for f in result if name_filter in f.name]
        if len(result) != 1:
            raise ValueError(
                f"tflow.get({args}, {kwargs}) returned {len(result)} files"
                + (f" after filtering for '{name_filter}'" if name_filter else "")
                + ", expected 1:\n" + "\n".join(f"  {p}" for p in result)
            )
        return result[0]
    return result


def load_parc_lists(wd):
    """Return (PARCS, PARCS_CX, PARCS_SC) from per-entity parc.yaml files.

    PARCS   : all real parcellation names (alias entries excluded), sorted.
    PARCS_CX: parcellations with level=='cortex'.
    PARCS_SC: parcellations with level=='subcortex'.
    """
    wd = Path(wd)
    PARCS, PARCS_CX, PARCS_SC = [], [], []
    for d in sorted((wd / "parcellation").glob("*")):
        if not d.is_dir():
            continue
        yml = d / "parc.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        if "alias" in cfg:
            continue
        PARCS.append(cfg["name"])
        if cfg.get("level") == "cortex":
            PARCS_CX.append(cfg["name"])
        elif cfg.get("level") == "subcortex":
            PARCS_SC.append(cfg["name"])
    return PARCS, PARCS_CX, PARCS_SC


def load_ref_lists(wd):
    """Return (REFS, REFS_CX, REFS_MAP, REFS_TAB) from per-entity ref.yaml files.

    REFS    : all reference dataset names, sorted.
    REFS_CX : datasets with cortex_only: true.
    REFS_MAP: datasets that have a maps: section (map+tab datasets).
    REFS_TAB: datasets without maps: (tab-only datasets).
    """
    wd = Path(wd)
    REFS, REFS_CX, REFS_MAP, REFS_TAB = [], [], [], []
    for d in sorted((wd / "reference").glob("*")):
        if not d.is_dir():
            continue
        yml = d / "ref.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        name = cfg["name"]
        REFS.append(name)
        if cfg.get("cortex_only", False):
            REFS_CX.append(name)
        if "maps" in cfg:
            REFS_MAP.append(name)
        else:
            REFS_TAB.append(name)
    return REFS, REFS_CX, REFS_MAP, REFS_TAB


def load_parc(wd, parc_name, space):
    """Load a parcellation from disk.

    Returns a NIfTI image for MNI spaces or a (L, R) gifti tuple for surface spaces.
    """
    wd = Path(wd)
    base = wd / "parcellation" / parc_name / space
    if "mni" in space.lower():
        return neuro_images.load_nifti(base / f"parc-{parc_name}_space-{space}.label.nii.gz")
    return (
        neuro_images.load_gifti(base / f"parc-{parc_name}_space-{space}_hemi-L.label.gii.gz"),
        neuro_images.load_gifti(base / f"parc-{parc_name}_space-{space}_hemi-R.label.gii.gz"),
    )


def load_parc_labels(wd, parc_name, space):
    """Load parcellation labels as a numpy str array.

    For surface spaces concatenates L then R labels.
    """
    wd = Path(wd)
    base = wd / "parcellation" / parc_name / space
    if "mni" in space.lower():
        return np.loadtxt(base / f"parc-{parc_name}_space-{space}.label.txt", dtype=str)
    return np.concatenate([
        np.loadtxt(base / f"parc-{parc_name}_space-{space}_hemi-L.label.txt", dtype=str),
        np.loadtxt(base / f"parc-{parc_name}_space-{space}_hemi-R.label.txt", dtype=str),
    ])


def save_csv_gz(df, path, **kwargs):
    """Save DataFrame to .csv.gz with a fixed gzip mtime for git-stable output."""
    df.to_csv(path, compression={"method": "gzip", "mtime": 1}, **kwargs)


def parcellate_mapref(wd, dataset, spaces):
    """Parcellate a map-based reference dataset across all parcellations.

    Args:
        wd      : repo root (Path)
        dataset : name matching a reference/<dataset>/ref.yaml (e.g. "pet")
        spaces  : ordered list of space keys. An "Original"-suffixed key means
                  "filter maps by this tag but load/parcellate in the base space."
                  Later entries overwrite earlier ones for maps present in both.

    Saves: reference/<dataset>/tab/dset-<dataset>_parc-<PARC>.csv.gz
    Raises: ValueError on missing maps, wrong output shape, or all-NaN map rows.
    """
    from nispace.io import parcellate_data

    wd = Path(wd)
    ref_yaml_path = wd / "reference" / dataset / "ref.yaml"
    with open(ref_yaml_path) as f:
        ref_cfg = yaml.safe_load(f)
    yaml_maps = ref_cfg.get("maps", {})
    cx_only = ref_cfg.get("cortex_only", False)

    PARCS, PARCS_CX, _ = load_parc_lists(wd)
    parc_names = PARCS_CX if cx_only else PARCS

    def _is_private(entry):
        if not isinstance(entry, dict):
            return False  # "auto"
        if "host" in entry:
            return "private" in str(entry["host"])
        if "L" in entry:
            return "private" in str(entry["L"].get("host", ""))
        if "R" in entry:
            return "private" in str(entry["R"].get("host", ""))
        return False

    def _fetch_paths(maps, space):
        map_dir = wd / "reference" / dataset / "map"
        paths = []
        for m in maps:
            fp = sorted((map_dir / m).glob(f"{m}_space-{space}_*"))
            if not fp:
                fp = sorted((map_dir / m).glob(f"{m}_space-{space}.*"))
            if not fp:
                raise ValueError(f"No map found for {m} in space {space}")
            elif len(fp) == 1:
                fp = fp[0]
            elif len(fp) == 2:
                fp = tuple(fp)
            else:
                raise ValueError(f"More than two maps found for {m} in {space}: {fp}")
            paths.append(fp)
        return paths

    # collect available (non-private) maps per filter_space
    ref_maps = {}
    for space in spaces:
        maps_in_space = []
        for m, m_spaces in yaml_maps.items():
            if not isinstance(m_spaces, dict) or space not in m_spaces:
                continue
            if not _is_private(m_spaces[space]):
                maps_in_space.append(m)
        if maps_in_space:
            ref_maps[space] = maps_in_space

    # unique maps across all filter_spaces (preserve yaml order)
    ref_maps_avail_all = [
        m for m in yaml_maps
        if any(m in v for v in ref_maps.values())
    ]
    print(f"[{dataset}] cx_only={cx_only} | {len(parc_names)} parcellations")
    for space, maps in ref_maps.items():
        print(f"  {space}: {len(maps)} maps")
    print(f"  → {len(ref_maps_avail_all)} unique maps total")
    if not ref_maps_avail_all:
        raise ValueError(
            f"[{dataset}] No non-private maps found for spaces {spaces}. "
            "Check ref.yaml and that map files exist."
        )

    from tqdm.auto import tqdm
    for parc_name in tqdm(parc_names, desc=f"parcellate [{dataset}]"):
        labels = load_parc_labels(wd, parc_name, "MNI152NLin6Asym")
        ref_maps_df = pd.DataFrame(
            index=pd.Index(list(ref_maps_avail_all), name="map"),
            columns=labels,
        )

        passes_run = 0
        for space in spaces:
            filter_space = space
            parc_space = space.replace("Original", "")
            if filter_space not in ref_maps:
                print(f"  [{parc_name}] skip {filter_space}: no non-private maps")
                continue
            parc_dir = wd / "parcellation" / parc_name / parc_space
            if not parc_dir.exists():
                print(f"  [{parc_name}] skip {parc_space}: parcellation dir not found")
                continue
            ref_maps_avail = ref_maps[filter_space]
            passes_run += 1

            parc = load_parc(wd, parc_name, parc_space)
            ref_paths = _fetch_paths(ref_maps_avail, parc_space)

            # bilateral: MNI space, or all surface maps have two hemispheres
            if ("MNI152" in parc_space) or (
                "MNI152" not in parc_space and all(len(fp) == 2 for fp in ref_paths)
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

            maps_assigned = 0
            for p, p_l, p_h, r_p, r_m_a in zip(
                parc_list, labels_list, hemi_list, paths_list, avail_list
            ):
                if not r_p:
                    continue
                tab = parcellate_data(
                    parcellation=p,
                    parc_hemi=p_h,
                    parc_labels=p_l,
                    parc_space=parc_space,
                    data=r_p,
                    data_labels=r_m_a,
                    data_space=parc_space,
                    n_proc=-1,
                    dtype=np.float32,
                    **DATASET_PARCELLATE_KWARGS[dataset],
                )
                # check parcellate_data did not silently drop maps
                if len(tab) != len(r_m_a):
                    raise ValueError(
                        f"[{dataset}/{parc_name}/{parc_space}"
                        f"{'/' + p_h if p_h else ''}] "
                        f"parcellate_data returned {len(tab)} rows, "
                        f"expected {len(r_m_a)}"
                    )
                ref_maps_df.loc[r_m_a, tab.columns] = tab
                maps_assigned += len(r_m_a)

            print(
                f"  [{parc_name}] {parc_space} ({filter_space}): "
                f"{maps_assigned} map(s) assigned"
            )

        # --- per-parcellation validation ---

        if passes_run == 0:
            raise ValueError(
                f"[{dataset}/{parc_name}] No parcellation pass ran. "
                f"Check that at least one of {spaces} has a matching parcellation dir."
            )

        if len(ref_maps_df) != len(ref_maps_avail_all):
            raise ValueError(
                f"[{dataset}/{parc_name}] row count mismatch: "
                f"expected {len(ref_maps_avail_all)}, got {len(ref_maps_df)}"
            )

        # all-NaN rows = map produced no data at all (always an error)
        nan_rows = ref_maps_df.index[ref_maps_df.isna().all(axis=1)].tolist()
        if nan_rows:
            raise ValueError(
                f"[{dataset}/{parc_name}] {len(nan_rows)} map(s) are entirely NaN "
                f"(parcellation produced no data): "
                + ", ".join(nan_rows[:5])
                + ("..." if len(nan_rows) > 5 else "")
            )

        # all-NaN columns = parcel has no data for any map (warning only — can be
        # legitimate for background parcels or cortex parcels in mixed-space datasets)
        nan_cols = ref_maps_df.columns[ref_maps_df.isna().all(axis=0)].tolist()
        if nan_cols:
            print(
                f"  WARNING [{parc_name}] {len(nan_cols)} parcel(s) all-NaN: "
                + str(nan_cols[:3])
                + ("..." if len(nan_cols) > 3 else "")
            )

        n_filled = int((~ref_maps_df.isna().all(axis=1)).sum())
        print(
            f"  [{parc_name}] {n_filled}/{len(ref_maps_avail_all)} maps filled | "
            f"{len(nan_cols)} all-NaN parcels | shape {ref_maps_df.shape}"
        )

        save_dir = wd / "reference" / dataset / "tab"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_csv_gz(ref_maps_df, save_dir / f"dset-{dataset}_parc-{parc_name}.csv.gz")


# Parcellate kwargs per dataset.
# Used with parcellate_data (prep3_0) or Parcellater.transform (prep3_3, prep3_5).
# neurosynth and grf use nispace.parcellate.Parcellater, which takes background_parcels_to_nan instead of drop_background_parcels
DATASET_PARCELLATE_KWARGS = {
    "pet":            dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "cortexfeatures": dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "bigbrain":       dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "rsn":            dict(ignore_background_data=False, drop_background_parcels=False,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "rsn17":          dict(ignore_background_data=False, drop_background_parcels=False,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "tpm":            dict(ignore_background_data=True, drop_background_parcels=True,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "mitobrain":      dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "neurosynth":     dict(ignore_background_data=False, background_parcels_to_nan=False, min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "grf":            dict(ignore_background_data=False, background_parcels_to_nan=False, min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
}
