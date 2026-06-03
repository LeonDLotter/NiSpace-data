# `NiSpace` – data library

[![DOI: 10.5281/zenodo.12514622](https://img.shields.io/badge/DOI-10.5281/zenodo.12514622-1082C3)](https://zenodo.org/doi/10.5281/zenodo.12514622)
[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

This is the repository that contains source data for the [`NiSpace`](https://github.com/leondlotter/nispace) toolbox.  
Every part of the data in here could be licensed under different licenses, so please read carefully the notes associated with the individual files. 

Data should be fetched via `nispace.datasets` functions. Please refer to the [NiSpace documentation](https://nispace.readthedocs.io/) for more information.

---

## Developer Guide

### Overview

All data registrations live in per-entity YAML source files in this repo.
A build script (`src/build_datalib.py`) aggregates them to JSON files that ship with the nispace toolbox (`nispace/datalib/`). The toolbox never reads YAML at runtime — only the aggregated JSONs.

The build script runs automatically via a git pre-commit hook on every `nispace-data` commit. It can also be run manually:

```bash
python3 src/build_datalib.py
```

**Config**: set `NISPACE_TOOLBOX_PATH` env var or create `config.json` at the repo root (gitignored):
```json
{"nispace_toolbox_path": "/path/to/nispace"}
```

---

### YAML Source Files

#### Reference datasets — `reference/{dataset}/ref.yaml`

One file per dataset. Fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Internal key, must match directory name |
| `label` | string | Human-readable display name |
| `description` | multiline string | Replaces the old `reference.txt` section |
| `cortex_only` | bool | Whether dataset has cortex-only parcellations |
| `citations` | list of `{ref, doi}` | Dataset-level citations, always shown |
| `map_info` | dict | OPTIONAL: config for per-map citation table (only for datasets with `map_info.csv`) |
| `collections` | dict | At minimum `All`; each entry has `description` and optional `citations` |
| `tabs` | see below | Tab file registry |
| `maps` | see below | Map file registry (MAP+TAB datasets only) |

**`tabs` field** — three supported formats:

```yaml
tabs: auto                   # scan disk; all dset-{name}_parc-*.csv.gz files auto-registered
tabs:                        # explicit list; host/remote inferred from naming convention
  - DesikanKilliany
  - Aseg
tabs:                        # legacy: explicit dict with host/remote (still accepted)
  Glasser:
    host: github-nispace
    remote: reference/pet/tab/dset-pet_parc-Glasser.csv.gz
```

All three paths check file existence on disk and warn for missing files.

**`maps` field** (MAP+TAB datasets only) — per-map, per-space entries:

```yaml
maps:
  map_name:
    MNIOriginal:             # external host — always explicit
      host: neuromaps
      remote: "..."
    MNI152NLin6Asym: auto    # github-nispace file — detected from disk at build time
    fsaverage: auto          # L + R hemisphere files detected from disk
    fsLR: auto
```

For `auto` spaces: volumetric → looks for `*_space-{space}[_.].nii.gz` (prefers `_desc-proc_` if multiple matches); surface → looks for `*_space-{space}*_hemi-{L|R}*.gii[.gz]` (any BIDS entities between `_space-` and `_hemi-` are allowed, e.g. `_desc-proc_`). Raises if >1 file matches after disambiguation.

External hosts (`neuromaps`, `url`, `osfprivate`) must always be listed explicitly.

**`map_info.csv`** — present only for: pet, enigmathick, enigmaarea, cortexfeatures, tpm, bigbrain. Contains per-map citation info. Registered automatically in the JSON if `map_info:` section exists in `ref.yaml`.

---

#### Parcellations — `parcellation/{ParcName}/parc.yaml`

One file per parcellation (including alias entries). Written by `prep1_0_parc.py` for new parcellations.

```yaml
name: Glasser
label: "HCP multimodal parcellation (Glasser atlas)"
level: cortex           # cortex | subcortex
symmetric: true
license: free
citation:
  ref: "Glasser et al., 2016, Nature"
  doi: "10.1038/nature18933"
spaces: auto            # scan disk; all file types (map, label, distmat, spinmat, l2rmap) auto-detected
```

**`spaces: auto`** scans `parcellation/{Name}/{space}/` subdirectories for known file types using the
naming convention `parc-{Name}_space-{space}[_hemi-{L|R}].{ext}`. Resolution is inferred from the
space name (`MNI152NLin6Asym/MNI152NLin2009cAsym → 1mm`, `fsaverage → 41k`, `fsLR → 32k`). A space
is only registered if its map file is present.

Explicit `spaces:` dict is also accepted (legacy/override):
```yaml
spaces:
  MNI152NLin6Asym:
    resolution: 1mm
    map: {host: github-nispace, remote: "..."}
    distmat: {host: github-nispace, remote: "..."}
```

Alias entries (short name → real name):
```yaml
name: Schaefer100
alias: Schaefer100Parcels7Networks
```

The global `parcellation/metadata.csv` is **deleted** — replaced by per-entity `parc.yaml`.

---

#### Templates — `template/{space}/template.yaml`

One file per template space. Written by `prep0_1_template.py`.

```yaml
name: MNI152NLin6Asym
resolutions:
  1mm:
    T1w: {host: github-nispace, remote: "..."}
    brain: {host: github-nispace, remote: "..."}
    mask: {host: github-nispace, remote: "..."}
    mask_gm: {host: github-nispace, remote: "..."}
  2mm:
    # ...
```

#### Template affines — `template/affines.yaml`

Single global file. Written by `prep0_0_affines.py`. Contains affine matrices and voxel shapes per space and resolution, used internally for NIfTI image creation.

Note: `prep0_1_template.py` reads `template/affines.yaml` directly (not the JSON) so it always uses the latest affines without needing to run `build_datalib.py` in between.

---

#### MNI space transforms — `transform/transform.yaml`

Single global file. Written by `prep0_2_transforms.py`. Registers EasyReg deformation fields for 11 bidirectional MNI space pairs.

**File layout:**
```
transform/
  transform.yaml
  MNI152NLin2009cAsym/
    from-{flo}_to-MNI152NLin2009cAsym_dir-backward_field.nii.gz
    from-{flo}_to-MNI152NLin2009cAsym_dir-forward_field.nii.gz
    ...
  MNI152NLin6Asym/
    from-{flo}_to-MNI152NLin6Asym_dir-backward_field.nii.gz
    ...
```

Organized by target (hub) space. The `dir-backward` field resamples source → target (primary direction); `dir-forward` is the inverse.

**YAML schema:**
```yaml
MNI152NLin2009cAsym:        # target space
  MNI152NLin6Asym:          # source space
    backward: {host: github-nispace, remote: "transform/MNI152NLin2009cAsym/from-MNI152NLin6Asym_to-MNI152NLin2009cAsym_dir-backward_field.nii.gz"}
    forward:  {host: github-nispace, remote: "transform/MNI152NLin2009cAsym/from-MNI152NLin6Asym_to-MNI152NLin2009cAsym_dir-forward_field.nii.gz"}
  # ... 6 more source spaces
MNI152NLin6Asym:
  # ... 4 source spaces
```

**11 supported pairs** (target ← source):

| Target | Sources |
|--------|---------|
| `MNI152NLin2009cAsym` | `MNI152NLin6Asym`, `MNI152Lin`, `MNI305`, `MNIColin27`, `MNI152NLin2009cSym`, `MNI152NLin6Sym` |
| `MNI152NLin6Asym` | `MNI152Lin`, `MNI305`, `MNIColin27`, `MNI152NLin6Sym`, `MNI152NLin2009cSym` |

Fields are generated on a compute cluster using [EasyReg](https://surfer.nmr.mgh.harvard.edu/fswiki/EasyReg) (FreeSurfer 8.2) with pre-computed [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) segmentations and stored in the `llotter/mni_freesurfer` GIN repository.

---

#### Examples — `example/{name}/example.yaml`

One file per example dataset. Data CSVs live alongside the yaml in `example/{name}/`.

```yaml
name: anorexianervosa
label: "Toy dataset simulated from ENIGMA AN maps."
description: "..."
tabs: auto              # scan example/{name}/ for example-{name}_parc-*.csv.gz
info: {host: github-nispace, remote: "example/{name}/example-{name}_info.csv"}  # OPTIONAL
```

**`tabs: auto`** scans `example/{name}/` for files matching `example-{name}_parc-*.csv.gz` and
registers them all. An explicit dict is also accepted (legacy).

---

### Adding New Data

#### New reference dataset

1. Run prep scripts to generate `reference/{dataset}/map/` and `reference/{dataset}/tab/`
2. Write `reference/{dataset}/ref.yaml`
3. If per-map citations needed: create `reference/{dataset}/map_info.csv`
4. Commit → pre-commit hook runs `build_datalib.py` → JSONs updated automatically
5. Copy new commit hash → update `DATA_REPO_COMMIT` in `nispace/config.py`

#### New parcellation

1. Run `prep1_0_parc.py` → generates files + writes `parcellation/{ParcName}/parc.yaml`
2. Run `prep1_1_parc_distmat.py` and `prep1_2_parc_spinmat.py` as needed
3. Commit → JSONs updated automatically
4. Update `DATA_REPO_COMMIT`

#### New template space or affine update

1. Run `prep0_0_affines.py` → writes `template/affines.yaml`
2. Run `prep0_1_template.py` → generates files + writes `template/{space}/template.yaml`
3. Commit → JSONs updated automatically
4. Update `DATA_REPO_COMMIT`

#### New or updated MNI space transform fields

1. Update `gin_commit` in `prep0_2_transforms.py` to the relevant `llotter/mni_freesurfer` GIN commit
2. Run `prep0_2_transforms.py` → downloads fields, writes `transform/transform.yaml`
3. Commit → `build_datalib.py` compiles `transform.json` automatically
4. Update `DATA_REPO_COMMIT`

#### New example dataset

1. Generate parcellated CSVs → place in `example/{name}/` (subdirectory, not repo root)
2. Write `example/{name}/example.yaml` with `tabs: auto`
3. Commit → JSONs updated automatically
4. Update `DATA_REPO_COMMIT`