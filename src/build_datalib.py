"""
Aggregates per-entity YAML source files → toolbox JSON files.

Run manually or via pre-commit hook after any YAML change in nispace-data.

Config (in priority order):
  1. NISPACE_TOOLBOX_PATH environment variable
  2. nispace-data/config.json  {"nispace_toolbox_path": "/path/to/nispace"}  (gitignored)
"""

import json
import os
import sys
import yaml
from pathlib import Path

NISPACE_DATA_DIR = Path(__file__).parent.parent


def _get_toolbox_dir() -> Path:
    if path := os.environ.get("NISPACE_TOOLBOX_PATH"):
        return Path(path)
    cfg_file = NISPACE_DATA_DIR / "config.json"
    if cfg_file.exists():
        return Path(json.loads(cfg_file.read_text())["nispace_toolbox_path"])
    print("ERROR: NISPACE_TOOLBOX_PATH not set and config.json not found.", file=sys.stderr)
    sys.exit(1)


# ── Tab resolution ────────────────────────────────────────────────────────────

def _resolve_tabs(tabs_val, dset_name: str) -> dict:
    """
    Resolve the 'tabs' field from ref.yaml to a {parcname: {host, remote}} dict.

    Supported formats:
      tabs: auto                   → scan disk for all tab files
      tabs: [Parc1, Parc2, ...]   → explicit list; host/remote inferred
      tabs: {Parc1: {host, remote}, ...}  → legacy explicit dict (passed through)

    In all cases, each resolved tab file is checked for existence on disk.
    Missing files produce a WARNING (not an error — files may not yet be downloaded).
    """
    tab_dir = NISPACE_DATA_DIR / "reference" / dset_name / "tab"
    pattern = f"dset-{dset_name}_parc-"

    def _make_entry(parc_name):
        fname = f"dset-{dset_name}_parc-{parc_name}.csv.gz"
        fpath = tab_dir / fname
        if not fpath.exists():
            print(f"    WARNING: tab file not on disk: {fpath.relative_to(NISPACE_DATA_DIR)}")
        return {"host": "github-nispace", "remote": f"reference/{dset_name}/tab/{fname}"}

    if tabs_val == "auto":
        entries = {}
        for f in sorted(tab_dir.glob(f"{pattern}*.csv.gz")):
            parc_name = f.name.removeprefix(pattern).removesuffix(".csv.gz")
            entries[parc_name] = _make_entry(parc_name)
        print(f"    tabs [auto]: {len(entries)} files → {', '.join(entries)}")
        return entries

    if isinstance(tabs_val, list):
        entries = {p: _make_entry(p) for p in tabs_val}
        print(f"    tabs [list]: {len(entries)} entries → {', '.join(entries)}")
        return entries

    if isinstance(tabs_val, dict):
        # Legacy format: {parcname: {host, remote}} — check existence and pass through
        missing = []
        for parc_name, entry in tabs_val.items():
            remote = entry.get("remote", "")
            fpath = NISPACE_DATA_DIR / remote
            if not fpath.exists():
                missing.append(parc_name)
        if missing:
            print(f"    WARNING: tab files not on disk: {', '.join(missing)}")
        print(f"    tabs [dict]: {len(tabs_val)} explicit entries")
        return tabs_val

    return {}


# ── Map resolution ────────────────────────────────────────────────────────────

_SURFACE_SPACES = {"fsaverage", "fslr", "fsaverageoriginal", "fslroriginal"}


def _resolve_map_space(map_dir: Path, map_name: str, space: str, space_val):
    """
    Resolve a single map+space entry.

    space_val can be:
      "auto"                   → discover github-nispace file(s) from disk
      {host, remote}           → volumetric explicit entry
      {L: {host, remote}, R:}  → surface explicit entry
    """
    if space_val != "auto":
        return space_val  # explicit — pass through unchanged

    is_surface = space.lower() in _SURFACE_SPACES
    remote_prefix = str(map_dir.relative_to(NISPACE_DATA_DIR))

    def _pick_one(matches, label):
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # Multiple matches: prefer _desc-proc_ (processed) over plain original
        proc = [f for f in matches if "_desc-proc" in f.name]
        if len(proc) == 1:
            return proc[0]
        raise ValueError(f"[auto] {label}: multiple files, cannot resolve: {[f.name for f in matches]}")

    if is_surface:
        result = {}
        for hemi in ("L", "R"):
            # Allow any entities between _space-{space} and _hemi-{hemi} (e.g. _desc-proc_)
            candidates = sorted(
                list(map_dir.glob(f"*_space-{space}*_hemi-{hemi}*.gii.gz"))
                + list(map_dir.glob(f"*_space-{space}*_hemi-{hemi}*.gii"))
            )
            f = _pick_one(candidates, f"{map_name}/{space}/{hemi}")
            if f is None:
                print(f"      [auto] {map_name}/{space}/{hemi}: no file found — space skipped")
                return None
            result[hemi] = {"host": "github-nispace", "remote": f"{remote_prefix}/{f.name}"}
        return result
    else:
        # Use exact boundary: _space-{space}.nii.gz OR _space-{space}_{suffix}.nii.gz
        candidates = sorted(
            list(map_dir.glob(f"*_space-{space}.nii.gz"))
            + list(map_dir.glob(f"*_space-{space}_*.nii.gz"))
        )
        f = _pick_one(candidates, f"{map_name}/{space}")
        if f is None:
            print(f"      [auto] {map_name}/{space}: no file found — space skipped")
            return None
        return {"host": "github-nispace", "remote": f"{remote_prefix}/{f.name}"}


def _resolve_maps(maps_val: dict, dset_name: str) -> dict:
    """Resolve all map entries, handling per-space 'auto' values."""
    if not maps_val:
        return {}
    resolved = {}
    map_base = NISPACE_DATA_DIR / "reference" / dset_name / "map"
    for map_name, spaces in maps_val.items():
        resolved_spaces = {}
        map_dir = map_base / map_name
        for space, space_val in spaces.items():
            result = _resolve_map_space(map_dir, map_name, space, space_val)
            if result is not None:
                resolved_spaces[space] = result
        resolved[map_name] = resolved_spaces
    return resolved


# ── Transformation helpers ────────────────────────────────────────────────────

def _ref_yaml_to_json_entry(cfg: dict, dset_name: str) -> dict:
    """Convert a ref.yaml config dict to a reference.json dataset entry."""
    entry = {}

    for key in ("label", "description", "default_collection", "cortex_only", "citations"):
        if key in cfg:
            entry[key] = cfg[key]

    if "map_info" in cfg:
        entry["map_info"] = cfg["map_info"]
        entry["metadata"] = {
            "host": "github-nispace",
            "remote": f"reference/{dset_name}/map_info.csv",
        }

    # Collections: merge inferred file path with description/citations
    if "collections" in cfg:
        entry["collection"] = {}
        for coll_name, coll_meta in cfg["collections"].items():
            coll_entry = {
                "host": "github-nispace",
                "remote": f"reference/{dset_name}/collection-{coll_name}.collect",
            }
            if desc := (coll_meta or {}).get("description"):
                coll_entry["description"] = desc
            if cites := (coll_meta or {}).get("citations"):
                coll_entry["citations"] = cites
            entry["collection"][coll_name] = coll_entry

    # Tabs
    if "tabs" in cfg:
        entry["tab"] = _resolve_tabs(cfg["tabs"], dset_name)

    # Maps
    if "maps" in cfg:
        entry["map"] = _resolve_maps(cfg["maps"], dset_name)

    return entry


# ── Parcellation space resolution ─────────────────────────────────────────────

_PARC_SPACE_RESOLUTION = {
    "MNI152NLin6Asym":    "1mm",
    "MNI152NLin2009cAsym": "1mm",
    "fsaverage":           "41k",
    "fsLR":                "32k",
}
_PARC_SURFACE_SPACES = {"fsaverage", "fsLR"}


def _resolve_parc_spaces(parc_name: str, parc_dir: Path) -> dict:
    """
    Scan disk for all space subdirectories and known file types.

    Called when parc.yaml has `spaces: auto`.
    Resolution is inferred from the space name (see _PARC_SPACE_RESOLUTION).
    A space is only registered if its map file is present.
    """
    spaces = {}
    for space_dir in sorted(parc_dir.iterdir()):
        if not space_dir.is_dir():
            continue
        space = space_dir.name
        if space not in _PARC_SPACE_RESOLUTION:
            continue
        resolution = _PARC_SPACE_RESOLUTION[space]
        is_surface = space in _PARC_SURFACE_SPACES
        prefix = f"parc-{parc_name}_space-{space}"
        rel_base = f"parcellation/{parc_name}/{space}/{prefix}"
        entry = {"resolution": resolution}

        if is_surface:
            for ftype, ext in [
                ("map",     ".label.gii.gz"),
                ("label",   ".label.txt"),
                ("distmat", ".dist.csv.gz"),
                ("spinmat", ".spin.npy"),
                ("l2rmap",  ".l2rmap.csv.gz"),
            ]:
                hemi_entry = {}
                for hemi in ("L", "R"):
                    f = space_dir / f"{prefix}_hemi-{hemi}{ext}"
                    if f.exists():
                        hemi_entry[hemi] = {"host": "github-nispace", "remote": f"{rel_base}_hemi-{hemi}{ext}"}
                if len(hemi_entry) == 2:
                    entry[ftype] = hemi_entry
        else:
            for ftype, ext in [
                ("map",     ".label.nii.gz"),
                ("label",   ".label.txt"),
                ("distmat", ".dist.csv.gz"),
                ("l2rmap",  ".l2rmap.csv.gz"),
            ]:
                f = space_dir / f"{prefix}{ext}"
                if f.exists():
                    entry[ftype] = {"host": "github-nispace", "remote": f"{rel_base}{ext}"}

        if "map" in entry:
            spaces[space] = entry
        else:
            print(f"    WARNING: {space_dir.relative_to(NISPACE_DATA_DIR)} has no map file — skipped")

    return spaces


def _parc_yaml_to_json_entry(cfg: dict) -> dict:
    """Convert a parc.yaml config dict to a parcellation.json entry."""
    entry = {}
    for key in ("label", "level", "symmetric", "license"):
        if key in cfg:
            entry[key] = cfg[key]
    if "citation" in cfg:
        entry["citation"] = cfg["citation"]

    spaces_val = cfg.get("spaces", {})
    if spaces_val == "auto":
        parc_name = cfg["name"]
        parc_dir = NISPACE_DATA_DIR / "parcellation" / parc_name
        print(f"    spaces [auto]: scanning {parc_dir.relative_to(NISPACE_DATA_DIR)}")
        spaces = _resolve_parc_spaces(parc_name, parc_dir)
        print(f"      → {list(spaces)}")
    else:
        spaces = spaces_val

    for space, space_cfg in spaces.items():
        entry[space] = {k: v for k, v in space_cfg.items() if k != "resolution"}
    return entry


def _template_yaml_to_json_entry(cfg: dict) -> dict:
    return cfg.get("resolutions", {})


# ── Build functions ───────────────────────────────────────────────────────────

def build_reference_json() -> dict:
    ref = {}
    for dset_dir in sorted((NISPACE_DATA_DIR / "reference").iterdir()):
        yml = dset_dir / "ref.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        name = cfg["name"]
        print(f"  [{name}]")
        ref[name] = _ref_yaml_to_json_entry(cfg, name)
    return ref


def build_parcellation_json() -> dict:
    parc = {}
    for parc_dir in sorted((NISPACE_DATA_DIR / "parcellation").iterdir()):
        if not parc_dir.is_dir():
            continue
        yml = parc_dir / "parc.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        name = cfg["name"]
        parc[name] = {"alias": cfg["alias"]} if "alias" in cfg else _parc_yaml_to_json_entry(cfg)
    return parc


def _resolve_example_tabs(tabs_val, ex_name: str) -> dict:
    """
    Resolve 'tabs' from example.yaml.

    tabs: auto  → scan example/{name}/ for example-{name}_parc-*.csv.gz
    tabs: dict  → pass through unchanged (legacy/explicit)
    """
    ex_dir = NISPACE_DATA_DIR / "example" / ex_name
    pattern = f"example-{ex_name}_parc-"

    def _make_entry(parc_name):
        fname = f"example-{ex_name}_parc-{parc_name}.csv.gz"
        fpath = ex_dir / fname
        if not fpath.exists():
            print(f"    WARNING: example tab not on disk: {fpath.relative_to(NISPACE_DATA_DIR)}")
        return {"host": "github-nispace", "remote": f"example/{ex_name}/{fname}"}

    if tabs_val == "auto":
        entries = {}
        for f in sorted(ex_dir.glob(f"{pattern}*.csv.gz")):
            parc_name = f.name.removeprefix(pattern).removesuffix(".csv.gz")
            entries[parc_name] = _make_entry(parc_name)
        print(f"    tabs [auto]: {len(entries)} files")
        return entries

    if isinstance(tabs_val, dict):
        return tabs_val

    return {}


def build_example_json() -> dict:
    ex = {}
    for ex_dir in sorted((NISPACE_DATA_DIR / "example").iterdir()):
        if not ex_dir.is_dir():
            continue
        yml = ex_dir / "example.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        name = cfg["name"]
        print(f"  [{name}]")
        entry = {"tab": _resolve_example_tabs(cfg.get("tabs", {}), name)}
        for key in ("label", "description", "citations"):
            if key in cfg:
                entry[key] = cfg[key]
        if "info" in cfg:
            entry["info"] = cfg["info"]
        ex[name] = entry
    return ex


def build_template_json() -> dict:
    tpl = {}
    for space_dir in sorted((NISPACE_DATA_DIR / "template").iterdir()):
        if not space_dir.is_dir():
            continue
        yml = space_dir / "template.yaml"
        if not yml.exists():
            continue
        cfg = yaml.safe_load(yml.read_text())
        tpl[cfg["name"]] = _template_yaml_to_json_entry(cfg)
    return tpl


def build_affines_json() -> dict:
    yml = NISPACE_DATA_DIR / "template" / "affines.yaml"
    return yaml.safe_load(yml.read_text()) if yml.exists() else {}


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    toolbox_dir = _get_toolbox_dir()
    out_dir = toolbox_dir / "nispace" / "datalib"
    if not out_dir.exists():
        print(f"ERROR: output dir not found: {out_dir}", file=sys.stderr)
        sys.exit(1)

    print("Building reference.json ...")
    ref = build_reference_json()
    print("Building parcellation.json ...")
    parc = build_parcellation_json()
    print("Building example.json ...")
    ex = build_example_json()
    print("Building template.json ...")
    tpl = build_template_json()
    print("Building affines.json ...")
    aff = build_affines_json()

    outputs = {
        "reference.json":    ref,
        "parcellation.json": parc,
        "example.json":      ex,
        "template.json":     tpl,
        "affines.json":      aff,
    }
    for filename, data in outputs.items():
        path = out_dir / filename
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False))
        print(f"  wrote {path}")

    print("Done.")
