# %% Init

import json
import numpy as np
import pandas as pd
from pathlib import Path

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

ENIGMA_DIR = Path("/Applications/miniforge3/envs/nsp309/lib/python3.9/site-packages"
                  "/enigmatoolbox/datasets/summary_statistics")


# %% Structure mappings

SUBVOL_MAP = {
    "Laccumb": "hemi-L_lab-Accumbens+area",
    "Lamyg":   "hemi-L_lab-Amygdala",
    "Lcaud":   "hemi-L_lab-Caudate",
    "Lhippo":  "hemi-L_lab-Hippocampus",
    "Lpal":    "hemi-L_lab-Pallidum",
    "Lput":    "hemi-L_lab-Putamen",
    "Lthal":   "hemi-L_lab-Thalamus",
    "Raccumb": "hemi-R_lab-Accumbens+area",
    "Ramyg":   "hemi-R_lab-Amygdala",
    "Rcaud":   "hemi-R_lab-Caudate",
    "Rhippo":  "hemi-R_lab-Hippocampus",
    "Rpal":    "hemi-R_lab-Pallidum",
    "Rput":    "hemi-R_lab-Putamen",
    "Rthal":   "hemi-R_lab-Thalamus",
    # LLatVent, RLatVent: excluded (not in NiSpace Aseg)
    # hemi-{L|R}_lab-VentralDC: not in ENIGMA -> stays NaN
}

# Get standard column order from existing reference dataset (ensures consistency)
_ref_dk = pd.read_csv(
    wd / "reference/neurosynth/tab/dset-neurosynth_parc-DesikanKilliany.csv.gz",
    index_col=0, nrows=1
)
DK_COLS = _ref_dk.columns.tolist()

_ref_aseg = pd.read_csv(
    wd / "reference/neurosynth/tab/dset-neurosynth_parc-Aseg.csv.gz",
    index_col=0, nrows=1
)
ASEG_COLS = _ref_aseg.columns.tolist()

print(f"DK cols: {len(DK_COLS)}, Aseg cols: {len(ASEG_COLS)}")


# %% Helper functions

def load_enigma(file_stem):
    """Load an ENIGMA summary stats CSV file by its stem name."""
    path = ENIGMA_DIR / f"{file_stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return pd.read_csv(path, on_bad_lines="skip")


def cortical_to_nispace(structure):
    """Convert 'L_bankssts' -> 'hemi-L_lab-bankssts'."""
    hemi, region = structure.split("_", 1)
    return f"hemi-{hemi}_lab-{region}"


def get_n(df):
    """Return (n_cases, n_controls) as ints or NaN from ENIGMA DataFrame."""
    def _parse(col):
        if col not in df.columns:
            return np.nan
        s = df[col].astype(str).str.replace(",", "")
        v = pd.to_numeric(s, errors="coerce").max()
        return int(v) if pd.notna(v) else np.nan
    return _parse("n_patients"), _parse("n_controls")


def extract_cortical(df):
    """Return dict {nispace_col: d_icv} from CortThick/CortSurf DataFrame."""
    df = df.set_index("Structure")
    return {cortical_to_nispace(s): row["d_icv"] for s, row in df.iterrows()}


def extract_subvol(df):
    """Return dict {nispace_col: d_icv} from SubVol DataFrame, mapped via SUBVOL_MAP."""
    df = df.set_index("Structure")
    return {SUBVOL_MAP[s]: row["d_icv"] for s, row in df.iterrows() if s in SUBVOL_MAP}


# %% Map definitions
#
# Each dict:
#   map_id        row ID (no _metric_ field)
#   disorder      human-readable disorder label
#   age           age group string or None
#   subtype       subtype string or None
#   adult         True = included in Adult collection (all maps except age-adolescent/pediatric)
#   thick_file    CSV stem for CortThick (or None)
#   surf_file     CSV stem for CortSurf (or None)
#   subvol_file   CSV stem for SubVol (or None if no strict age match)
#   doi_cortical  DOI for cortical publication
#   doi_subvol    DOI for SubVol publication (may differ)

MAPS = [
    # --- MDD ---
    dict(
        map_id="dx-mdd_age-adult_pub-schmaal2017",
        disorder="MDD", age="adult", subtype=None, adult=True,
        thick_file="mddadult_case-controls_CortThick",
        surf_file="mddadult_case-controls_CortSurf",
        subvol_file="mdd_case-controls_SubVol",
        doi_cortical="10.1038/mp.2016.60",
        doi_subvol="10.1038/mp.2015.69",
    ),
    dict(
        map_id="dx-mdd_age-adolescent_pub-schmaal2017",
        disorder="MDD", age="adolescent", subtype=None, adult=False,
        thick_file="mddadolescent_case-controls_CortThick",
        surf_file="mddadolescent_case-controls_CortSurf",
        subvol_file=None,
        doi_cortical="10.1038/mp.2016.60",
        doi_subvol=None,
    ),
    # --- ADHD ---
    dict(
        map_id="dx-adhd_age-allages_pub-hoogman2019",
        disorder="ADHD", age="allages", subtype=None, adult=True,
        thick_file="adhdallages_case-controls_CortThick",
        surf_file="adhdallages_case-controls_CortSurf",
        subvol_file="adhdallages_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2018.18091016",
        doi_subvol="10.1016/S2215-0366(16)30386-4",
    ),
    dict(
        map_id="dx-adhd_age-adult_pub-hoogman2019",
        disorder="ADHD", age="adult", subtype=None, adult=True,
        thick_file="adhdadult_case-controls_CortThick",
        surf_file="adhdadult_case-controls_CortSurf",
        subvol_file="adhdadult_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2018.18091016",
        doi_subvol="10.1016/S2215-0366(16)30386-4",
    ),
    dict(
        map_id="dx-adhd_age-adolescent_pub-hoogman2019",
        disorder="ADHD", age="adolescent", subtype=None, adult=False,
        thick_file="adhdadolescent_case-controls_CortThick",
        surf_file="adhdadolescent_case-controls_CortSurf",
        subvol_file="adhdadolescent_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2018.18091016",
        doi_subvol="10.1016/S2215-0366(16)30386-4",
    ),
    dict(
        map_id="dx-adhd_age-pediatric_pub-hoogman2019",
        disorder="ADHD", age="pediatric", subtype=None, adult=False,
        thick_file="adhdpediatric_case-controls_CortThick",
        surf_file="adhdpediatric_case-controls_CortSurf",
        subvol_file="adhdpediatric_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2018.18091016",
        doi_subvol="10.1016/S2215-0366(16)30386-4",
    ),
    # --- ASD (CortThick only; no CortSurf in ENIGMA) ---
    dict(
        map_id="dx-asd_pub-vanrooij2018",
        disorder="ASD", age=None, subtype=None, adult=True,
        thick_file="asd_mega-analysis_case-controls_CortThick",
        surf_file=None,
        subvol_file="asd_meta-analysis_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2017.17091017",
        doi_subvol="10.1176/appi.ajp.2017.17091017",
    ),
    # --- BD ---
    dict(
        map_id="dx-bd_age-adult_pub-hibar2018",
        disorder="BD", age="adult", subtype=None, adult=True,
        thick_file="bd_case-controls_CortThick_adult",
        surf_file="bd_case-controls_CortSurf_adult",
        subvol_file="bd_case-controls_SubVol_typeI",
        doi_cortical="10.1038/s41380-018-0033-x",
        doi_subvol="10.1038/mp.2015.154",
    ),
    dict(
        map_id="dx-bd_age-adolescent_pub-hibar2018",
        disorder="BD", age="adolescent", subtype=None, adult=False,
        thick_file="bd_case-controls_CortThick_adolescent",
        surf_file="bd_case-controls_CortSurf_adolescent",
        subvol_file=None,
        doi_cortical="10.1038/s41380-018-0033-x",
        doi_subvol=None,
    ),
    # --- SCZ ---
    dict(
        map_id="dx-scz_pub-vanerp2018",
        disorder="SCZ", age=None, subtype=None, adult=True,
        thick_file="scz_case-controls_CortThick",
        surf_file="scz_case-controls_CortSurf",
        subvol_file="scz_case-controls_SubVol",
        doi_cortical="10.1016/j.biopsych.2017.08.017",
        doi_subvol="10.1038/mp.2015.61",
    ),
    # --- OCD ---
    dict(
        map_id="dx-ocd_age-adult_pub-boedhoe2018",
        disorder="OCD", age="adult", subtype=None, adult=True,
        thick_file="ocdadults_case-controls_CortThick",
        surf_file="ocdadults_case-controls_CortSurf",
        subvol_file="ocdadult_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2017.17030297",
        doi_subvol="10.1176/appi.ajp.2016.16020201",
    ),
    dict(
        map_id="dx-ocd_age-pediatric_pub-boedhoe2018",
        disorder="OCD", age="pediatric", subtype=None, adult=False,
        thick_file="ocdpediatric_case-controls_CortThick",
        surf_file="ocdpediatric_case-controls_CortSurf",
        subvol_file="ocdpediatric_case-controls_SubVol",
        doi_cortical="10.1176/appi.ajp.2017.17030297",
        doi_subvol="10.1176/appi.ajp.2016.16020201",
    ),
    # --- Epilepsy (CortThick only; no CortSurf in ENIGMA) ---
    dict(
        map_id="dx-epilepsy_pub-whelan2018",
        disorder="Epilepsy", age=None, subtype=None, adult=True,
        thick_file="allepi_case-controls_CortThick",
        surf_file=None,
        subvol_file="allepi_case-controls_SubVol",
        doi_cortical="10.1093/brain/awx341",
        doi_subvol="10.1093/brain/awx341",
    ),
    dict(
        map_id="dx-epilepsy_subtype-gge_pub-whelan2018",
        disorder="Epilepsy", age=None, subtype="gge", adult=True,
        thick_file="gge_case-controls_CortThick",
        surf_file=None,
        subvol_file="gge_case-controls_SubVol",
        doi_cortical="10.1093/brain/awx341",
        doi_subvol="10.1093/brain/awx341",
    ),
    dict(
        map_id="dx-epilepsy_subtype-ltle_pub-whelan2018",
        disorder="Epilepsy", age=None, subtype="ltle", adult=True,
        thick_file="tlemtsl_case-controls_CortThick",
        surf_file=None,
        subvol_file="tlemtsl_case-controls_SubVol",
        doi_cortical="10.1093/brain/awx341",
        doi_subvol="10.1093/brain/awx341",
    ),
    dict(
        map_id="dx-epilepsy_subtype-rtle_pub-whelan2018",
        disorder="Epilepsy", age=None, subtype="rtle", adult=True,
        thick_file="tlemtsr_case-controls_CortThick",
        surf_file=None,
        subvol_file="tlemtsr_case-controls_SubVol",
        doi_cortical="10.1093/brain/awx341",
        doi_subvol="10.1093/brain/awx341",
    ),
    # --- 22q ---
    dict(
        map_id="dx-22q_pub-sun2020",
        disorder="22q11.2DS", age=None, subtype=None, adult=True,
        thick_file="22q_case-controls_CortThick",
        surf_file="22q_case-controls_CortSurf",
        subvol_file="22q_case-controls_SubVol",
        doi_cortical="10.1038/s41380-020-0717-9",
        doi_subvol="10.1176/appi.ajp.2019.19060638",
    ),
    # --- AN ---
    dict(
        map_id="dx-an_pub-walton2022",
        disorder="AN", age=None, subtype=None, adult=True,
        thick_file="anorexia_case-controls_CortThick",
        surf_file="anorexia_case-controls_CortSurf",
        subvol_file="anorexia_case-controls_SubVol",
        doi_cortical="10.1016/j.biopsych.2022.02.006",
        doi_subvol="10.1016/j.biopsych.2022.02.006",
    ),
    dict(
        map_id="dx-an_subtype-acAN_pub-walton2022",
        disorder="AN", age=None, subtype="acAN", adult=True,
        thick_file="anorexia_acAN-controls_CortThick",
        surf_file="anorexia_acAN-controls_CortSurf",
        subvol_file="anorexia_acAN-controls_SubVol",
        doi_cortical="10.1016/j.biopsych.2022.02.006",
        doi_subvol="10.1016/j.biopsych.2022.02.006",
    ),
    dict(
        map_id="dx-an_subtype-pwrAN_pub-walton2022",
        disorder="AN", age=None, subtype="pwrAN", adult=True,
        thick_file="anorexia_pwrAN-controls_CortThick",
        surf_file="anorexia_pwrAN-controls_CortSurf",
        subvol_file="anorexia_pwrAN-controls_SubVol",
        doi_cortical="10.1016/j.biopsych.2022.02.006",
        doi_subvol="10.1016/j.biopsych.2022.02.006",
    ),
    # --- Antisocial ---
    dict(
        map_id="dx-antisocial_pub-gao2024",
        disorder="AsPD", age=None, subtype=None, adult=True,
        thick_file="Antisocial_case-controls_CortThick",
        surf_file="Antisocial_case-controls_CortSurf",
        subvol_file="Antisocial_case-controls_SubVol",
        doi_cortical="10.1016/S2215-0366(24)00187-1",
        doi_subvol="10.1016/S2215-0366(24)00187-1",
    ),
    # --- PD ---
    dict(
        map_id="dx-pd_pub-laansma2021",
        disorder="PD", age=None, subtype=None, adult=True,
        thick_file="parkinsons_case-controls_CortThick_PDvsCN",
        surf_file="parkinsons_case-controls_CortSurf_PDvsCN",
        subvol_file="parkinsons_case-controls_Subvol_PDvsCN",
        doi_cortical="10.1002/mds.28706",
        doi_subvol="10.1002/mds.28706",
    ),
]


# %% Load all ENIGMA data

print("\nLoading ENIGMA data...")

for m in MAPS:
    data = {}
    for metric in ("thick", "surf", "subvol"):
        file_key = f"{metric}_file"
        if m[file_key] is None:
            data[metric] = None
            continue
        df = load_enigma(m[file_key])
        data[metric] = df
        n_cases, n_ctrl = get_n(df)
        m[f"n_cases_{metric}"] = n_cases
        m[f"n_ctrl_{metric}"] = n_ctrl

    m["_data"] = data
    print(f"  {m['map_id']}: thick={'ok' if data['thick'] is not None else 'none'}"
          f", surf={'ok' if data['surf'] is not None else 'none'}"
          f", subvol={'ok' if data['subvol'] is not None else 'none'}")


# %% Build tables

print("\nBuilding tables...")

# enigmathick: all maps with thick_file (22 maps)
thick_maps = [m for m in MAPS if m["_data"]["thick"] is not None]

# enigmaarea: maps with surf_file only (17 maps; no ASD, no Epilepsy)
surf_maps = [m for m in MAPS if m["_data"]["surf"] is not None]

print(f"enigmathick: {len(thick_maps)} maps")
print(f"enigmaarea:  {len(surf_maps)} maps")


def build_tabs(maps, cortical_key):
    """Build DesikanKilliany and Aseg DataFrames for a given metric."""
    dk_rows = {}
    aseg_rows = {}

    for m in maps:
        mid = m["map_id"]
        cort_df = m["_data"][cortical_key]
        subvol_df = m["_data"]["subvol"]

        # cortical
        dk_rows[mid] = extract_cortical(cort_df)

        # subcortical (NaN row if no SubVol match)
        if subvol_df is not None:
            aseg_rows[mid] = extract_subvol(subvol_df)
        else:
            aseg_rows[mid] = {}

    # Build DataFrames with standard column order; NaN for missing values
    dk_df = pd.DataFrame(dk_rows).T.reindex(columns=DK_COLS)
    aseg_df = pd.DataFrame(aseg_rows).T.reindex(columns=ASEG_COLS)

    return dk_df, aseg_df


thick_dk, thick_aseg = build_tabs(thick_maps, "thick")
surf_dk, surf_aseg = build_tabs(surf_maps, "surf")

print(f"\nenigmathick DK:   {thick_dk.shape}, NaN cols: {thick_dk.isna().all().sum()}")
print(f"enigmathick Aseg: {thick_aseg.shape}, NaN rows: {thick_aseg.isna().all(axis=1).sum()}")
print(f"enigmaarea DK:    {surf_dk.shape}, NaN cols: {surf_dk.isna().all().sum()}")
print(f"enigmaarea Aseg:  {surf_aseg.shape}, NaN rows: {surf_aseg.isna().all(axis=1).sum()}")


# %% Verify pwrAN CortThick anomaly

pwran_thick = next(m for m in MAPS if "pwrAN" in m["map_id"] and m["_data"]["thick"] is not None)
nc = pwran_thick.get("n_ctrl_thick")
np_ = pwran_thick.get("n_cases_thick")
print(f"\npwrAN CortThick: n_cases={np_}, n_ctrl={nc}")
if nc == np_:
    print("  NOTE: n_cases == n_controls in pwrAN CortThick — possible data issue in ENIGMA source")


# %% Build metadata CSVs

def build_metadata(maps, cortical_metric):
    rows = []
    for m in maps:
        rows.append({
            "disorder": m["disorder"],
            "age_group": m.get("age"),
            "subtype": m.get("subtype"),
            "n_cases": m.get(f"n_cases_{cortical_metric}"),
            "n_controls": m.get(f"n_ctrl_{cortical_metric}"),
            "doi": m["doi_cortical"],
            "n_cases_subvol": m.get("n_cases_subvol"),
            "n_controls_subvol": m.get("n_ctrl_subvol"),
            "doi_subvol": m["doi_subvol"],
            "enigma_file": m.get(f"{cortical_metric}_file"),
            "enigma_file_subvol": m.get("subvol_file"),
        })
    return pd.DataFrame(rows, index=[m["map_id"] for m in maps])


thick_meta = build_metadata(thick_maps, "thick")
surf_meta = build_metadata(surf_maps, "surf")


# %% Build collection files

def build_collections(maps):
    all_ids = [m["map_id"] for m in maps]
    adult_ids = [m["map_id"] for m in maps if m["adult"]]
    return all_ids, adult_ids


thick_all, thick_adult = build_collections(thick_maps)
surf_all, surf_adult = build_collections(surf_maps)

print(f"\nenigmathick collections: All={len(thick_all)}, Adult={len(thick_adult)}")
print(f"enigmaarea  collections: All={len(surf_all)}, Adult={len(surf_adult)}")


# %% Save outputs

for dset, dk_df, aseg_df, meta_df, all_ids, adult_ids in [
    ("enigmathick", thick_dk, thick_aseg, thick_meta, thick_all, thick_adult),
    ("enigmaarea",  surf_dk,  surf_aseg,  surf_meta,  surf_all,  surf_adult),
]:
    out_dir = wd / "reference" / dset
    tab_dir = out_dir / "tab"
    tab_dir.mkdir(parents=True, exist_ok=True)

    # tabs
    dk_path = tab_dir / f"dset-{dset}_parc-DesikanKilliany.csv.gz"
    aseg_path = tab_dir / f"dset-{dset}_parc-Aseg.csv.gz"
    dk_df.to_csv(dk_path)
    aseg_df.to_csv(aseg_path)

    # metadata
    meta_df.to_csv(out_dir / "metadata.csv")

    # collections
    pd.DataFrame({"map": all_ids}).to_csv(out_dir / "collection-All.collect", index=False)
    pd.DataFrame({"map": adult_ids}).to_csv(out_dir / "collection-Adult.collect", index=False)

    print(f"\nSaved {dset}:")
    print(f"  {dk_path.name}: {dk_df.shape}")
    print(f"  {aseg_path.name}: {aseg_df.shape}")
    print(f"  collection-All.collect: {len(all_ids)} maps")
    print(f"  collection-Adult.collect: {len(adult_ids)} maps")

print("\nDone.")
