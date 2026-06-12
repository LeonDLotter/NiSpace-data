"""
Generate overview plots for all parcellations and reference datasets.

Saves PNGs at:
  parcellation/{Name}/plot/parc-{Name}_plot-overview.png
  reference/{name}/plot/dset-{name}_plot-overview.png

Usage:
  python src/prep_plots.py [--parcs] [--refs] [--overwrite] [--name NAME]

  --parcs     Only generate parcellation plots
  --refs      Only generate reference plots
  --overwrite Regenerate even if PNG already exists
  --name NAME Only process this parcellation/dataset name
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

wd = Path(__file__).parent.parent
sys.path.insert(0, str(wd / "src"))

# Point nispace to the local repo BEFORE any nispace import caches the path.
os.environ["NISPACE_DATA_DIR"] = str(wd)

from utils import load_parc_lists, load_ref_lists
from nispace.datasets import fetch_parcellation
from nispace.plotting import brainplot


# ── Config ─────────────────────────────────────────────────────────────────

DPI = 250
EXT = "png"
N_MAPS = 6
SLICE_CUT_COORDS = [-30, -15, 0, 15, 30]
SLICE_CUT_COORDS_SC = [-20, -10, 0, 10, 20]
SUPTITLE_SIZE = 24
TITLE_SIZE = 20
SUBTITLE_SIZE = 20

PARC_SURFACE_SPACES = ["fsLR", "fsaverage"]
PARC_MNI_SPACES = ["MNI152NLin2009cAsym", "MNI152NLin6Asym"]
# Layout order: [top-left, bottom-left, top-right, bottom-right]
CORTICAL_SPACE_ORDER = ["fsLR", "fsaverage", "MNI152NLin6Asym", "MNI152NLin2009cAsym"]
SUBCORTICAL_SPACE_ORDER = ["MNI152NLin6Asym", "MNI152NLin2009cAsym"]
# Ordered space names shown in the availability caption for reference datasets
DISPLAY_SPACES = ["fsLR", "fsaverage", "MNI152NLin6Asym", "MNI152NLin2009cAsym"]


# ── Curated map/tab selections per reference dataset ───────────────────────
# If a dataset is absent here (or its curated IDs are unavailable), the script
# falls back to the first N_MAPS entries from the dataset's maps/tabs.

PLOT_MAP_SELECTION = {
    "pet": [
        "target-rCPS_tracer-leucine_n-42_dx-hc_pub-smith2023",
        "target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018",
        "target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019",
        "target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015",
        "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012",
        "target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017",
    ],
    "rsn": [
        "nw-Auditory_pub-dworetsky2021",
        "nw-SomatomotorDorsal_pub-dworetsky2021",
        "nw-DorsalAttention_pub-dworetsky2021",
        "nw-DefaultMode_pub-dworetsky2021",
        "nw-Frontoparietal_pub-dworetsky2021",
        "nw-Language_pub-dworetsky2021",
    ],
    "rsn17": [
        "nw-Auditory_pub-kong2022",
        "nw-SomatomotorA_pub-kong2022",
        "nw-DorsAttnA_pub-kong2022",
        "nw-DefaultA_pub-kong2022",
        "nw-ControlA_pub-kong2022",
        "nw-Language_pub-kong2022",
    ],
    "bigbrain": [
        "feature-histogradient1_pub-paquola2021",
        "feature-histogradient2_pub-paquola2021",
        "feature-microgradient1_pub-paquola2021",
        "feature-funcgradient1_pub-paquola2021",
        "feature-layer1_pub-wagstyl2020",
        "feature-layer4_pub-wagstyl2020",
    ],
    "cortexfeatures": [
        "feature-saaxis_pub-sydnor2021",
        "feature-evolexpansion_pub-xu2020",
        "feature-thickness_pub-hcps1200",
        "feature-fcgradient1_pub-margulies2016",
        "feature-megpoweralpha_pub-shafiei2022",
        "feature-cbf_pub-vaishnavi2010",
    ]
        
}

PLOT_TAB_SELECTION = {
    "mrna": ["PVALB", "SST", "VIP", "SLC17A7", "HTR2A", "MAOA"],
    "magicc": ["PVALB", "SST", "VIP", "SLC17A7", "HTR2A", "MAOA"],
    "neurosynth": ["attention", "memory", "language", "pain", "reward", "motion"],
    "grf": [
        "alpha-0.0_seed-0000",
        "alpha-1.0_seed-0000",
        "alpha-3.0_seed-0000",
        "alpha-0.0_seed-0500",
        "alpha-1.0_seed-0500",
        "alpha-3.0_seed-0500",
    ],
    "enigmaarea": [
        "dx-mdd_age-adult_pub-schmaal2017",
        "dx-adhd_age-allages_pub-hoogman2019",
        "dx-bd_age-adult_pub-hibar2018",
        "dx-scz_pub-vanerp2018",
        "dx-ocd_age-adult_pub-boedhoe2018",
        "dx-antisocial_pub-gao2024",
    ],
    "enigmathick": [
        "dx-mdd_age-adult_pub-schmaal2017",
        "dx-adhd_age-allages_pub-hoogman2019",
        "dx-bd_age-adult_pub-hibar2018",
        "dx-scz_pub-vanerp2018",
        "dx-ocd_age-adult_pub-boedhoe2018",
        "dx-antisocial_pub-gao2024",
    ],
}

# Colormap override per reference dataset (None → brainplot default).
PLOT_CMAP = {
    "pet": "inferno",
    "rsn": "mako",
    "rsn17": "mako",
    "bigbrain": "magma",
    "cortexfeatures": "magma",
    "mitobrain": "magma",
    "tpm": "magma",
    "mrna": "viridis",
    "magicc": "viridis",
    "neurosynth": "crest",
    "grf": "mako",
    "enigmaarea": "RdBu_r",
    "enigmathick": "RdBu_r",
}

# Parcellation to use when tab-based datasets need tabular → brain mapping.
# If parc is in PARCS_CX → kind="surface"/fsLR; else → kind="slice"/MNI.
TAB_PARC = {
    "mrna": "Yan200+TianS3",
    "neurosynth": "Yan1000+TianS3",
    "grf": "Yan1000+TianS3",
    "magicc": "Yan1000",
    "enigmaarea": "DesikanKilliany",
    "enigmathick": "DesikanKilliany",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _available_parc_spaces_ordered(parc_name, space_order):
    parc_dir = wd / "parcellation" / parc_name
    return [s for s in space_order if (parc_dir / s).is_dir()]


def _resolve_ids(curated, available, n=N_MAPS):
    """Return up to n IDs: curated first (filtered to available), padded from available."""
    selected = [i for i in curated if i in available]
    if len(selected) < n:
        extras = [i for i in available if i not in selected]
        selected += extras[: n - len(selected)]
    return selected[:n]


def _short_title(map_id):
    """Derive a compact display title from a BIDS-style map ID."""
    label = map_id.split("_")[0]
    for prefix in ("target-", "feature-", "nw-", "mito-", "tissue-",
                   "dx-", "alpha-", "gene-"):
        if label.startswith(prefix):
            key = prefix.rstrip("-")
            value = label[len(prefix):]
            return f"{key}: {value}"
    return label


def _find_map_file(name, map_id, space):
    """Return local path(s) to the requested map file.

    Returns a Path for NIfTI/MNI spaces or a (lh_Path, rh_Path) tuple for
    surface spaces. Returns None if the file is not found on disk.
    """
    map_dir = wd / "reference" / name / "map" / map_id
    if not map_dir.is_dir():
        return None

    if "mni" in space.lower():
        for pattern in (
            f"*_space-{space}_desc-proc.nii.gz",
            f"*_space-{space}.nii.gz",
            f"*_space-{space}_*.nii.gz",
        ):
            hits = sorted(map_dir.glob(pattern))
            if hits:
                return hits[0]
    else:
        # Use wildcard between space and hemi to handle intervening entities (e.g. _desc-proc)
        lh = sorted(map_dir.glob(f"*_space-{space}*_hemi-L*.gii.gz"))
        rh = sorted(map_dir.glob(f"*_space-{space}*_hemi-R*.gii.gz"))
        if lh and rh:
            return (lh[0], rh[0])
    return None


def _availability_text(name, ref_cfg, PARCS_CX, PARCS_SC):
    """Build the availability caption: spaces (for map datasets) + parcellation coverage."""
    tab_dir = wd / "reference" / name / "tab"
    tab_parcs = set()
    if tab_dir.is_dir():
        for f in tab_dir.glob(f"dset-{name}_parc-*.csv.gz"):
            parc_name = f.stem.replace(f"dset-{name}_parc-", "").replace(".csv", "")
            tab_parcs.add(parc_name)

    cx_set = set(PARCS_CX)
    sc_set = set(PARCS_SC)
    if tab_parcs >= (cx_set | sc_set):
        parc_text = "all parcellations"
    elif tab_parcs >= cx_set:
        parc_text = "all cortex parcellations"
    elif tab_parcs:
        parc_text = ", ".join(sorted(tab_parcs))
    else:
        parc_text = None

    if "maps" in ref_cfg:
        first_map = next(iter(ref_cfg["maps"].values()))
        spaces = [s for s in DISPLAY_SPACES if s in first_map]
        parts = spaces + ([parc_text] if parc_text else [])
    else:
        parts = [parc_text] if parc_text else []

    return "(" + ", ".join(parts) + ")" if parts else None


# ── Parcellation plots ───────────────────────────────────────────────────────

def plot_parcellation(name, level, overwrite=False):
    out_path = wd / "parcellation" / name / "plot" / f"parc-{name}_plot-overview.{EXT}"
    if out_path.exists() and not overwrite:
        print(f"    skip (exists): {out_path.relative_to(wd)}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if level == "cortex":
        space_order = CORTICAL_SPACE_ORDER
        spaces = _available_parc_spaces_ordered(name, space_order)
        fig, axes = plt.subplots(2, 2, figsize=(16, 4.5), gridspec_kw={"hspace": 0.2, "wspace": 0.1})
        axes_flat = axes.flatten()
        pos = {s: axes_flat[i] for i, s in enumerate(space_order)}
    else:
        space_order = SUBCORTICAL_SPACE_ORDER
        spaces = _available_parc_spaces_ordered(name, space_order)
        fig, axes = plt.subplots(1, 2, figsize=(16, 2.5), gridspec_kw={"wspace": 0.1})
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes[0], axes[1]]
        pos = {s: axes_flat[i] for i, s in enumerate(space_order)}

    # Hide all axes upfront; they'll be filled or left blank
    for ax in axes_flat:
        ax.set_axis_off()

    try:
        parc_obj = fetch_parcellation(name, check_file_hash=False, verbose=False)
    except Exception as e:
        print(f"    ERROR fetching parcellation '{name}': {e}")
        plt.close(fig)
        return

    for space in space_order:
        ax = pos[space]
        if space not in spaces:
            ax.set_visible(True)
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{space}\nn/a", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#aaaaaa")
            continue

        kind = "surface" if space in PARC_SURFACE_SPACES else "slice"
        kwargs = dict(title=space, colorbar=False, shuffle_cmap=42,
                      fig=fig, axes=[ax], verbose=False, alpha=0.7)
        if kind == "slice":
            kwargs["cut_coords"] = SLICE_CUT_COORDS if level=="cortex" else SLICE_CUT_COORDS_SC
        if level == "cortex":
            kwargs["title_kwargs"] = {"fontsize": TITLE_SIZE, "y": 1.05}
        else:
            kwargs["title_kwargs"] = {"fontsize": TITLE_SIZE, "y": 0.95}

        try:
            parc_obj.plot(space=space, kind=kind, **kwargs)
        except Exception as e:
            print(f"    WARNING: {name}/{space} plot failed: {e}")
            ax.set_axis_off()

    fig.suptitle(name, fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.0 if level=="cortex" else 1.2, x=0.512)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {out_path.relative_to(wd)}")


# ── Reference plots ──────────────────────────────────────────────────────────

def plot_reference(name, ref_cfg, overwrite=False):
    out_path = wd / "reference" / name / "plot" / f"dset-{name}_plot-overview.{EXT}"
    if out_path.exists() and not overwrite:
        print(f"    skip (exists): {out_path.relative_to(wd)}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, PARCS_CX, PARCS_SC = load_parc_lists(wd)
    avail_text = _availability_text(name, ref_cfg, PARCS_CX, PARCS_SC)

    has_maps = "maps" in ref_cfg

    if has_maps:
        first_map = next(iter(ref_cfg["maps"].values()))
        if "MNI152NLin6Asym" in first_map:
            kind, space = "slice", "MNI152NLin6Asym"
        else:
            kind, space = "surface", "fsLR"
        _plot_reference_maps(name, ref_cfg, kind, space, out_path, avail_text)
    else:
        # kind/space determined inside from TAB_PARC parcellation type
        _plot_reference_tabs(name, ref_cfg, out_path, PARCS_CX, avail_text)


def _plot_reference_maps(name, ref_cfg, kind, space, out_path, avail_text=None):
    all_map_ids = list(ref_cfg.get("maps", {}).keys())
    curated = PLOT_MAP_SELECTION.get(name, [])
    selected_ids = _resolve_ids(curated, all_map_ids)

    if kind == "surface":
        # Collect (lh, rh) GIfTI tuples for a single brainplot call
        pairs = []
        titles = []
        for mid in selected_ids:
            path = _find_map_file(name, mid, space)
            if path is None:
                print(f"    WARNING: no {space} file for {name}/{mid}")
                continue
            if isinstance(path, tuple):
                pairs.append(path)
                titles.append(_short_title(mid))

        if not pairs:
            print(f"    WARNING: no surface files found for {name}, skipping")
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(24, 4.5), gridspec_kw={"hspace": 0.1, "wspace": 0.05})
            for ax in axes.ravel(): ax.set_axis_off()
            fig, _ = brainplot(
                pairs, kind="surface", space=space,
                ncols=3, title=titles, title_kwargs={"fontsize": TITLE_SIZE, "y": 1.08},
                cmap=PLOT_CMAP.get(name),
                colorbar=False, verbose=False, axes=axes, fig=fig,
                shared_colorscale=False,
            )
            fig.suptitle(name, fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.05, x=0.512)
            if avail_text:
                fig.text(0.512, 0.1, avail_text, ha="center", va="top",
                         fontsize=SUBTITLE_SIZE, style="italic", color="#555555",
                         transform=fig.transFigure)
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {out_path.relative_to(wd)}")
        except Exception as e:
            print(f"    ERROR plotting {name} surface maps: {e}")
            plt.close("all")

    else:
        # NIfTI: collect paths for a single brainplot call (list input)
        niis = []
        titles = []
        for mid in selected_ids:
            path = _find_map_file(name, mid, space)
            if path is None:
                print(f"    WARNING: no {space} file for {name}/{mid}")
                continue
            if isinstance(path, Path):
                niis.append(path)
                titles.append(_short_title(mid))

        if not niis:
            print(f"    WARNING: no MNI files found for {name}, skipping")
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(24, 5), gridspec_kw={"hspace": 0.2, "wspace": 0.05})
            for ax in axes.ravel(): ax.set_axis_off()
            fig, _ = brainplot(
                niis, kind="slice", space=space, cut_coords=SLICE_CUT_COORDS,
                ncols=3, title=titles, title_kwargs={"fontsize": TITLE_SIZE, "y": 1.0},
                cmap=PLOT_CMAP.get(name),
                axes=axes, fig=fig, colorbar=False, verbose=False,
            )
            fig.suptitle(name, fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.05, x=0.512)
            if avail_text:
                fig.text(0.512, 0.07, avail_text, ha="center", va="top",
                         fontsize=SUBTITLE_SIZE, style="italic", color="#555555",
                         transform=fig.transFigure)
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved: {out_path.relative_to(wd)}")
        except Exception as e:
            print(f"    ERROR plotting {name} NIfTI maps: {e}")
            plt.close("all")


def _plot_reference_tabs(name, ref_cfg, out_path, PARCS_CX, avail_text=None):
    parc_name = TAB_PARC.get(name)
    if parc_name is None:
        print(f"    WARNING: no TAB_PARC entry for '{name}', skipping")
        return

    # Support combined parcellations written as "CompA+CompB"
    components = parc_name.split("+") if "+" in parc_name else [parc_name]
    tab_dir = wd / "reference" / name / "tab"
    parts = []
    for comp in components:
        p = tab_dir / f"dset-{name}_parc-{comp}.csv.gz"
        if not p.exists():
            print(f"    WARNING: tab not found: {p.relative_to(wd)}")
            return
        parts.append(pd.read_csv(p, index_col=0))
    tab_df = pd.concat(parts, axis=1, join="inner") if len(parts) > 1 else parts[0]

    # kind/space: surface if all components are cortex-only, else slice
    if all(c in PARCS_CX for c in components):
        kind = "surface"
        space = "fsLR"
    else:
        kind = "slice"
        space = "MNI152NLin6Asym"

    curated = PLOT_TAB_SELECTION.get(name, [])
    selected_ids = _resolve_ids(curated, list(tab_df.index))

    df = tab_df.loc[selected_ids]
    titles = [_short_title(i) for i in selected_ids]

    if kind == "surface":
        fig, axes = plt.subplots(2, 3, figsize=(24, 4.5), gridspec_kw={"hspace": 0.1, "wspace": 0.05})
        for ax in axes.ravel(): ax.set_axis_off()
        plot_kwargs = dict(
            parcellation=parc_name, kind=kind, space=space,
            ncols=3, title=titles, title_kwargs={"fontsize": TITLE_SIZE, "y": 1.08},
            cmap=PLOT_CMAP.get(name),
            axes=axes, fig=fig, colorbar=False, verbose=False, shared_colorscale=False,
        )
        avail_y = 0.1
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 5), gridspec_kw={"hspace": 0.2, "wspace": 0.05})
        for ax in axes.ravel(): ax.set_axis_off()
        plot_kwargs = dict(
            parcellation=parc_name, kind=kind, space=space,
            cut_coords=SLICE_CUT_COORDS,
            ncols=3, title=titles, title_kwargs={"fontsize": TITLE_SIZE, "y": 1.0},
            cmap=PLOT_CMAP.get(name),
            axes=axes, fig=fig, colorbar=False, verbose=False,
        )
        avail_y = 0.07

    try:
        fig, _ = brainplot(df, **plot_kwargs)
        fig.suptitle(name, fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.05, x=0.512)
        if avail_text:
            fig.text(0.512, avail_y, avail_text, ha="center", va="top",
                     fontsize=SUBTITLE_SIZE, style="italic", color="#555555",
                     transform=fig.transFigure)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    saved: {out_path.relative_to(wd)}")
    except Exception as e:
        print(f"    ERROR plotting {name} tabs: {e}")
        plt.close("all")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parcs", action="store_true")
    parser.add_argument("--refs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--name", default=None, help="Only process this name")
    args = parser.parse_args()

    run_parcs = args.parcs or not (args.parcs or args.refs)
    run_refs = args.refs or not (args.parcs or args.refs)

    if run_parcs:
        print("=== Parcellation plots ===")
        PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
        for name in PARCS:
            if args.name and name != args.name:
                continue
            level = "cortex" if name in PARCS_CX else "subcortex"
            print(f"  [{name}] ({level})")
            try:
                plot_parcellation(name, level, overwrite=args.overwrite)
            except Exception as e:
                print(f"    ERROR: {e}")

    if run_refs:
        print("=== Reference plots ===")
        for ref_dir in sorted((wd / "reference").iterdir()):
            yml = ref_dir / "ref.yaml"
            if not yml.exists():
                continue
            ref_cfg = yaml.safe_load(yml.read_text())
            name = ref_cfg["name"]
            if args.name and name != args.name:
                continue
            print(f"  [{name}]")
            try:
                plot_reference(name, ref_cfg, overwrite=args.overwrite)
            except Exception as e:
                print(f"    ERROR: {e}")


if __name__ == "__main__":
    main()
