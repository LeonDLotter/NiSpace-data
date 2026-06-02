"""Shared helpers for nispace-data prep scripts."""

import yaml
import numpy as np
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


# Parcellate kwargs per dataset.
# Used with parcellate_data (prep3_0) or Parcellater.transform (prep3_3, prep3_5).
# neurosynth and grf use nispace.parcellate.Parcellater, which takes background_parcels_to_nan instead of drop_background_parcels
DATASET_PARCELLATE_KWARGS = {
    "pet":            dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "cortexfeatures": dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "bigbrain":       dict(ignore_background_data=True,  drop_background_parcels=True,    min_num_valid_datapoints=5,    min_fraction_valid_datapoints=0.3),
    "rsn":            dict(ignore_background_data=False, drop_background_parcels=False,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "rsn17":          dict(ignore_background_data=False, drop_background_parcels=False,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "tpm":            dict(ignore_background_data=False, drop_background_parcels=False,   min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "neurosynth":     dict(ignore_background_data=False, background_parcels_to_nan=False, min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
    "grf":            dict(ignore_background_data=False, background_parcels_to_nan=False, min_num_valid_datapoints=None, min_fraction_valid_datapoints=None),
}
