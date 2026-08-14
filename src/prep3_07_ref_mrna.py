# %% Init

import sys
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import nibabel as nib
from nilearn.image import resample_img, resample_to_img

# wd
wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# Abagen
from abagen import get_expression_data, keep_stable_genes

# Nispace
from nispace.io import load_labels, write_json, read_msigdb_json
from nispace.utils.utils_datasets import download

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, save_csv_gz

# nispace data path
nispace_source_data_path = wd

# parcellations
PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
print("PARCS:", PARCS)
print("PARCS_SC:", PARCS_SC)

# %% mRNA tabulated data ---------------------------------------------------------------------------

# settings
overwrite = True  # set True to recompute stability atlas and gene filter
stab_voxel_size = 8
corr_threshold = 0.2
n_jobs_extraction = 2


# Function to build "parcellation" for stability analysis
def build_stability_atlas(wd, voxel_size=stab_voxel_size):
    space = "MNI152NLin6Asym"
    offset = 100_000

    def _load_hemi_mask(parc, hemi):
        base = wd / "parcellation" / parc / space
        img = nib.load(base / f"parc-{parc}_space-{space}.label.nii.gz")
        labels = load_labels(base / f"parc-{parc}_space-{space}.label.txt")
        ids = [i+1 for i, lbl in enumerate(labels)
               if lbl.split("hemi-")[1].split("_")[0] == hemi]
        mask = np.isin(np.round(img.get_fdata()).astype(int), ids)
        return nib.Nifti1Image(mask.astype(np.uint8), img.affine)

    structures = [
        (_load_hemi_mask("Schaefer100Parcels7Networks", "L"), "L", "cortex"),
        (_load_hemi_mask("Schaefer100Parcels7Networks", "R"), "R", "cortex"),
        (_load_hemi_mask("TianS1",                      "L"), "L", "subcortex/brainstem"),
        (_load_hemi_mask("TianS1",                      "R"), "R", "subcortex/brainstem"),
    ]

    ref_img = resample_img(structures[0][0], target_affine=np.diag([voxel_size]*3),
                           interpolation="nearest", copy_header=True, force_resample=True)

    def _resample(img):
        return np.round(
            resample_to_img(img, ref_img, interpolation="nearest",
                            copy_header=True, force_resample=True).get_fdata()
        ).astype(bool)

    combined = np.zeros(ref_img.shape, dtype=np.int32)
    rows = []
    for k, (mask_img, hemi, structure) in enumerate(structures):
        mask = _resample(mask_img)
        ids = np.arange(1, mask.sum() + 1) + k * offset
        combined[mask] = ids
        rows.extend(
            {"id": int(vid), "label": f"stab_{vid}", "hemisphere": hemi, "structure": structure}
            for vid in ids
        )

    print(f"Stability atlas built with voxel size {voxel_size}, resulting in {len(rows)} voxels/'parcels'")
    return nib.Nifti1Image(combined, ref_img.affine), pd.DataFrame(rows)


# Function to extract mRNA data for a given parcellation
def par_fun(parc):
    
    # Monkey patch pandas append for abagen compatibility
    import pandas._libs.lib as lib
    if not hasattr(pd.DataFrame, 'append'):
        def _append(self, other, axis=0, **kwargs):
            return pd.concat([self, other], axis=axis, **kwargs)
        pd.DataFrame.append = _append
        # Also patch the C-extension module
        if hasattr(lib, 'DataFrame'):
            lib.DataFrame.append = _append

    # all parcellations are available in MNI152NLin6Asym
    space = "MNI152NLin6Asym"
    parc_path = str(
        nispace_source_data_path / "parcellation" / parc / space /
        f"parc-{parc}_space-{space}.label.nii.gz"
    )
    parc_labels = load_labels(
        nispace_source_data_path / "parcellation" / parc / space /
        f"parc-{parc}_space-{space}.label.txt"
    )

    # parc info
    parc_info = pd.DataFrame({
        "id": np.arange(1, len(parc_labels) + 1),
        "label": parc_labels,
        "hemisphere": [l.split("hemi-")[1].split("_")[0] for l in parc_labels],
        "structure": ["cortex" if parc not in PARCS_SC else "subcortex/brainstem"] * len(parc_labels)
    })

    # get combined data for all donors
    mRNA_tab = get_expression_data(
        atlas=parc_path,
        atlas_info=parc_info,
        lr_mirror="bidirectional",
        norm_matched=False, # required to ensure that cortex and subcortex data can be combined post-hoc
        n_proc=1,
        verbose=False, #0
    )

    # process dataset        
    mRNA_tab.index = parc_info.label
    mRNA_tab = mRNA_tab.T
    mRNA_tab.index.name = "map"
    mRNA_tab = mRNA_tab.astype(np.float32)

    # subset dataset
    n_genes_prior = mRNA_tab.shape[0]
    mRNA_tab = mRNA_tab.loc[genes_to_extract]
    print(f"Parcellation: {parc}. Originally {n_genes_prior} genes.\n"
          f"After correlation threshold of >= {corr_threshold}, {mRNA_tab.shape[0]} genes remain.")

    # save
    save_csv_gz(mRNA_tab, nispace_source_data_path / "reference" / "mrna" / "tab" / f"dset-mrna_parc-{parc}.csv.gz")

#%% Run stability

# --- global stability filter (parcellation-independent) ---
stab_atlas_path = nispace_source_data_path / "reference" / "mrna" / "tab" / "_stability_atlas.nii.gz"
stab_repro_path = nispace_source_data_path / "reference" / "mrna" / "tab" / "dset-mrna_reproducibility.csv.gz"

if not stab_repro_path.exists() or overwrite:
    stab_img, stab_info = build_stability_atlas(wd)
    nib.save(stab_img, stab_atlas_path)

    stab_donors = get_expression_data(
        atlas=str(stab_atlas_path),
        atlas_info=stab_info,
        lr_mirror="bidirectional",
        norm_matched=False,
        tolerance=stab_voxel_size, # * 2,
        missing=None, # "centroids",
        n_proc=1,
        verbose=False,
        return_donors=True,
    )
    stab_list, stability_arr = keep_stable_genes(
        list(stab_donors.values()),
        threshold=corr_threshold,
        percentile=False,
        rank=True,
        return_stability=True,
    )
    stability = pd.Series(
        stability_arr,
        index=stab_donors[next(iter(stab_donors))].columns,
        dtype=np.float32,
    )
    stability.name = "reproducibility"
    stability.index.name = "map"
    save_csv_gz(stability, stab_repro_path)
else:
    stability = pd.read_csv(stab_repro_path, index_col=0).squeeze()

# get genes
genes_to_extract = stability[stability >= corr_threshold].index.tolist()
print(f"Global stable genes: {len(genes_to_extract)}")


# %% Run par_fun for each parcellation

# parcellations
print(f"{len(PARCS)} parcellations: {PARCS}")

# Run in parallel
Parallel(n_jobs=n_jobs_extraction)(
    delayed(par_fun)(parc)
    for parc in tqdm(PARCS)
)


# %% Collections

import zipfile
import tarfile
import shutil
from scipy import stats
from statsmodels.stats.multitest import multipletests

ref_dir = nispace_source_data_path / "reference" / "mrna"

# All genes (union across all parcellations)
all_genes = set()
for f in (ref_dir / "tab").glob("dset-mrna_parc-*.csv.gz"):
    if "_reproducibility" not in f.name:
        all_genes.update(pd.read_csv(f, index_col=0).index.unique())
pd.Series(sorted(all_genes), name="map").to_csv(ref_dir / "collection-All.collect", index=False)

# Siletti et al. 2023 — cell types
df_siletti = (
    pd.read_excel("https://raw.githubusercontent.com/linnarsson-lab/adult-human-brain/3832d54f6ecffd001b55fd80d1d8b32ceca25bfa/tables/cluster_annotation.xlsx")
    .dropna(how="all")
    .loc[:, ["Supercluster", "Cluster name", "Top Enriched Genes"]]
)
siletti_sets = {
    "Upper-layer intratelencephalic": "Upper-layer IT",
    "Deep-layer intratelencephalic": "Deep-layer IT",
    "Deep-layer near-projecting": "Deep-layer NP",
    "Deep-layer corticothalamic and 6b": "Deep-layer CT/6b",
    "MGE interneuron": "MGE interneuron",
    "CGE interneuron": "CGE interneuron",
    "LAMP5-LHX6 and Chandelier": "LAMP5-LHX6/Chandelier",
    "Hippocampal CA1-3": "Hippocampus CA1-3",
    "Hippocampal CA4": "Hippocampus CA4",
    "Hippocampal dentate gyrus": "Hippocampus DG",
    "Amygdala excitatory": "Amygdala excitatory",
    "Medium spiny neuron": "MSN",
    "Eccentric medium spiny neuron": "Eccentric MSN",
    "Splatter": "Splatter",
    "Mammillary body": "Mammillary body",
    "Thalamic excitatory": "Thalamic excitatory",
    "Midbrain-derived inhibitory": "Midbrain-derived inhibitory",
    "Upper rhombic lip": "Upper rhombic lip",
    "Cerebellar inhibitory": "Cerebellar inhibitory",
    "Lower rhombic lip": "Lower rhombic lip",
    "Astrocyte": "Astrocyte",
    "Oligodendrocyte": "Oligodendrocyte",
    "Oligodendrocyte precursor": "OPC",
    "Committed oligodendrocyte precursor": "Committed OPC",
    "Microglia": "Microglia",
    "Bergmann glia": "Bergmann glia",
    "Vascular": "Vascular",
    "Choroid plexus": "Choroid plexus",
    "Fibroblast": "Fibroblast",
    "Ependymal": "Ependymal",
    "Miscellaneous": "Miscellaneous",
}
# clusters
write_json(
    {f"{siletti_sets[s]} - {x}": df_siletti.query("`Cluster name`==@x")["Top Enriched Genes"].str.split(", ").explode().unique().tolist()
     for s in siletti_sets for x in df_siletti.query("Supercluster==@s")["Cluster name"].unique()},
    ref_dir / "collection-CellTypesSilettiClusters.collect",
)
# superclusters
write_json(
    {siletti_sets[s]: df_siletti.query("Supercluster==@s")["Top Enriched Genes"].str.split(", ").explode().unique().tolist()
     for s in siletti_sets if s != "Miscellaneous"},
    ref_dir / "collection-CellTypesSilettiSuperclusters.collect",
)

# PsychEncode cell types
for url, save_name in [
    ("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/DER-19_Single_cell_markergenes_TPM.xlsx",  "CellTypesPsychEncodeTPM"),
    ("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/DER-21_Single_cell_markergenes_UMI.xlsx", "CellTypesPsychEncodeUMI"),
]:
    df_pe = pd.read_excel(url) if "TPM" in save_name else pd.read_excel(url, header=1)
    df_pe = df_pe.rename(columns={"GeneName": "gene", "CellType": "set"} if "TPM" in save_name else {"Gene": "gene", "Cluster": "set"}).astype(str)
    collection_pe = {k: sorted(df_pe.query("set==@k").gene.unique()) for k in df_pe.set.unique()}
    if "TPM" in save_name:
        tpm_names = {
            "Adult-Ex1": "Ex1 CortProject (L2/3)", "Adult-Ex2": "Ex2 Granule (L3/4)",
            "Adult-Ex3": "Ex3 Granule (L4)", "Adult-Ex4": "Ex4 SubcortProject (L4)",
            "Adult-Ex5": "Ex5 SubcortProject (L4-6)", "Adult-Ex6": "Ex6 SubcortProject (L5-6)",
            "Adult-Ex7": "Ex7 Corticothalamic", "Adult-Ex8": "Ex8 Corticothalamic (L6)",
            "Adult-In1": "In1 VIP+RELN+NDNF+ (L1/2)", "Adult-In2": "In2 VIP+RELN-NDNF- (L6)",
            "Adult-In3": "In3 VIP+RELN+NDNF- (L6)", "Adult-In4": "In4 VIP-RELN+NDNF+ (L1-3)",
            "Adult-In5": "In5 CCK+NOS1+CALB2+ (L2/3)", "Adult-In6": "In6 PVALB+CRHBP+ (L4/5)",
            "Adult-In7": "In7 SST+CALB1+NPY+ (L5/6)", "Adult-In8": "In8 SST+NOS1+ (L6)",
            "Adult-OtherNeuron": "Other Neurons", "Dev-quiescent": "Developing-quiescent",
            "Dev-replicating": "Developing-replicating", "Adult-Astro": "Astrocyte",
            "Adult-Endo": "Endothelial", "Adult-Micro": "Microglia",
            "Adult-OPC": "OPC", "Adult-Oligo": "Oligodendrocyte",
        }
        collection_pe = {tpm_names[k]: collection_pe[k] for k in collection_pe}
    write_json(collection_pe, ref_dir / f"collection-{save_name}.collect")

# SynGO
syngo_path = download("https://syngoportal.org/data/syngo1.3_complete_data.zip")
with zipfile.ZipFile(syngo_path).open("ontologies.xlsx") as f:
    df_syngo = pd.read_excel(f)
write_json(
    {name: genes.split(", ") for _, name, genes in zip(df_syngo["id"], df_syngo["name"], df_syngo["hgnc_symbol"])},
    ref_dir / "collection-SynGO.collect",
)

# GO gene sets
for name, short_name in [
    ("GOBiologicalProcess", "bp"),
    ("GOCellularComponent", "cc"),
    ("GOMolecularFunction", "mf"),
]:
    raw = read_msigdb_json(download(
        f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c5.go.{short_name}.v2025.1.Hs.json"
    ))
    write_json(
        {k.replace(f"GO{short_name.upper()}_", ""): v for k, v in raw.items()},
        ref_dir / f"collection-{name}.collect",
    )

# Chromosome locations
raw_chr = read_msigdb_json(download(
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c1.all.v2025.1.Hs.json"
))
collection_chr = {}
for k, v in raw_chr.items():
    if "chr" in k:
        chr_id = k.replace("chr", "").split("p")[0].split("q")[0]
        if chr_id not in ["X", "Y"]:
            chr_id = chr_id.zfill(2)
        pq = ("p" if "p" in k else "q") + k.split("p")[-1].split("q")[-1].zfill(2)
        k = f"chr{chr_id}{pq}"
    collection_chr[k] = v
write_json(
    {k: collection_chr[k] for k in sorted(collection_chr.keys())},
    ref_dir / "collection-Chromosome.collect",
)

# Cortical layers (Wagstyl 2024)
df_wagstyl = pd.read_excel("https://cdn.elifesciences.org/articles/86933/elife-86933-supp2-v1.xlsx")
write_json(
    {s.lower().replace(" ", ""): df_wagstyl[df_wagstyl[s] == True]["gene.symbol"].unique().tolist()
     for s in ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"]},
    ref_dir / "collection-CorticalLayers.collect",
)

# Protein Atlas — brain expression categories (HPA v25.1, fetched 2026-08-14; no immutable version
# pin available - v25.proteinatlas.org still redirects to the live site since v25 hasn't been
# superseded yet, unlike e.g. v24.proteinatlas.org which is a frozen prior-version snapshot)
collection_pa = {}
for s, url in [
    # order is ordinal by brain-relevance - used below to resolve the handful of gene symbols
    # that HPA's search-export returns in two adjacent categories at once (duplicate Ensembl-level
    # records sharing one HGNC symbol, with differing category calls)
    ("ElevatedInBrain",       "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("NotElevatedInBrain",    "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BNot+detected+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("LowSpecificityInBrain", "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BLow+tissue+specificity+AND+NOT+tissue_category_rna%3Abrain%3BNot+detected?format=tsv&download=yes"),
    ("NotInBrain",            "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BNot+detected+AND+NOT+tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes"),
    ("NotInTissue",           "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes"),
]:
    collection_pa[s] = set(pd.read_table(url)["Gene"].unique().tolist())
seen_pa = set()
for s in collection_pa:
    collection_pa[s] -= seen_pa
    seen_pa |= collection_pa[s]
collection_pa = {s: sorted(genes) for s, genes in collection_pa.items()}
# derived rollups - the practical "is this gene brain-expressed" filter/background
collection_pa["ExpressedInBrain"] = sorted(
    set(collection_pa["ElevatedInBrain"]) | set(collection_pa["NotElevatedInBrain"]) | set(collection_pa["LowSpecificityInBrain"])
)
collection_pa["NotExpressedInBrain"] = sorted(set(collection_pa["NotInBrain"]) | set(collection_pa["NotInTissue"]))
write_json(collection_pa, ref_dir / "collection-ProteinAtlas.collect")

# BrainSpan — developmental expression marker genes (ABAEnrichment)
aba_fp = nispace_source_data_path / "_archive" / "ABAData_1.0.0.tar.gz"
if not aba_fp.exists():
    download("https://mghp.osn.xsede.org/bir190004-bucket01/archive.bioconductor.org/packages/3.2/data/experiment/src/contrib/ABAData_1.0.0.tar.gz", aba_fp)
with tarfile.open(aba_fp, "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith("dataset_5_stages.rda"):
            tar.extract(member, path=nispace_source_data_path / "_archive")
import pyreadr
expression = list(pyreadr.read_r(nispace_source_data_path / "_archive" / "ABAData" / "data" / "dataset_5_stages.rda").values())[0]
aba_stages  = {1: "prenatal", 2: "infant", 3: "child", 4: "adolescent", 5: "adult"}
aba_regions = {
    10194: "OFC", 10173: "dlPFC", 10185: "vlPFC", 10278: "ACC", 10163: "M1C",
    10209: "S1C", 10225: "IPC", 10236: "A1C", 10243: "STC", 10252: "ITC",
    10269: "V1C", 10294: "HIP", 10361: "AMY", 10333: "STR", 10398: "mdTHA", 10657: "CBC",
}
expression = (
    expression
    .rename(columns={"hgnc_symbol": "gene_symbol", "structure": "region", "signal": "expression", "age_category": "stage"})
    [["gene_symbol", "region", "stage", "expression"]]
)
expression["stage"] = expression["stage"].replace(aba_stages)
expression["region"] = expression["region"].replace(aba_regions)
expression["region_stage"] = [f"{s1}-{s2}" for s1, s2 in zip(expression.stage, expression.region)]
expr_matrix = expression[["gene_symbol", "region_stage", "expression"]].pivot_table(
    columns="region_stage", index="gene_symbol"
).droplevel(0, axis=1)

# Restricted to Protein-Atlas-confirmed brain-expressed genes before stage-enrichment testing:
# unlike every other collection in this script, BrainSpan derives set membership from its own
# t-test/fold-change on raw expression magnitude, which floor-noise from non-brain-expressed genes
# can spuriously trip.
expr_matrix = expr_matrix.loc[expr_matrix.index.isin(collection_pa["ExpressedInBrain"])]
expr_matrix = expr_matrix[np.not_equal(expr_matrix.sum(axis=1), 0)]
expr_matrix = np.log2(expr_matrix + 1)

n_genes, n_cond = expr_matrix.shape
p_mat  = np.full((n_genes, n_cond), np.nan)
fc_mat = np.full((n_genes, n_cond), np.nan)
pc_mat = np.full((n_genes, n_cond), np.nan)
arr = expr_matrix.values
for i in tqdm(range(n_cond), desc="BrainSpan conditions"):
    target = arr[:, i]
    other  = arr[:, np.arange(n_cond) != i]
    for j in range(n_genes):
        _, p_mat[j, i] = stats.ttest_1samp(other[j], target[j], alternative="less")
        with np.errstate(divide="ignore"):
            fc = np.log2(target[j] / other[j].mean())
            fc_mat[j, i] = 0.0 if np.isinf(fc) else fc
    pc_mat[:, i] = multipletests(p_mat[:, i], method="fdr_bh")[1]
p_df  = pd.DataFrame(p_mat,  index=expr_matrix.index, columns=expr_matrix.columns)
fc_df = pd.DataFrame(fc_mat, index=expr_matrix.index, columns=expr_matrix.columns)
pc_df = pd.DataFrame(pc_mat, index=expr_matrix.index, columns=expr_matrix.columns)
pbonf_df = p_df * n_genes * n_cond
aba_sets = [f"{s}-{r}" for s in aba_stages.values() for r in aba_regions.values()]

write_json(
    {k: p_df[k][(pbonf_df[k] < 0.05) & (fc_df[k] > 1)].index.tolist() for k in aba_sets},
    ref_dir / "collection-BrainSpan.collect",
)
pd.concat(
    {k: fc_df[k][fc_df[k] > 0].to_frame(name="weight") for k in aba_sets},
    axis=0, names=["set", "map"],
).astype(np.float16).to_csv(ref_dir / "collection-BrainSpanWeights.collect", index=True)

# ASD-dysregulated WGCNA modules (Gandal et al., 2022)
#
# Source: Supplementary Data 5 (MOESM7, gene->WGCNA-module membership) and Supplementary
# Data 6 (MOESM8, module characterization incl. Whole.Cortex_ASD_Beta/FDR and per-region
# ASD_{region}_Beta/FDR, both from linear mixed models of module eigengene ~ ASD status).
#
# The paper states (Results, "Cortex-wide modules..."): "In total, 38 modules were up- or
# downregulated in at least one region in ASD. Most of these fell into two broad groups:
# (1) dysregulated cortex-wide with comparable magnitude across regions (18 modules); or
# (2) exhibiting variable changes across regions (13 modules)." That 38/18/13 pools combine
# the 35 gene modules AND 39 non-overlapping transcript(isoform) modules (e.g. IsoformM37
# is explicitly one of the 18 "cortex-wide" pool). This collection is gene-level only (the
# nispace mrna/magicc reference datasets have no isoform-level tabs to match against), so it
# reproduces the full GENE-only portion of that 38/18/13 breakdown - 24 of the 35 gene
# modules - rather than the combined 31 (18+13). One of these 24, GeneM9, is further excluded
# below for a nispace-specific data-quality reason (see "excluded_modules" below), leaving 23.
#
# Core set (18 modules): Whole.Cortex_ASD_FDR < 0.05 -> 18 of 35 gene modules (M0_grey, the
# WGCNA "unassigned" bin, is never significant and excluded by convention regardless). This
# threshold independently reproduces the paper's headline "18" cortex-wide count, and,
# using a looser "FDR<0.05 in >=1 of 11 individual regions" criterion on the same 35 modules,
# exactly 24 modules split into exactly 9 down / 15 up (paper: "Nine modules were
# downregulated and 15 were upregulated in ASD" - the pre-region-classification headline
# figure for genes alone). Module gene counts also match Fig. 3 exactly (GeneM5: 398,
# GeneM9: 243, GeneM24: 102, GeneM32: 65), confirming correct extraction.
#
# Regionally variable extension (+6 modules): GeneM4, GeneM6, GeneM13, GeneM16, GeneM17,
# GeneM30 are the remaining 6 of the 24 gene modules significant in >=1 region but NOT
# whole-cortex (this is the complete gene-only "regionally variable" pool; the paper's 13
# also includes 7 isoform-only modules not represented here). Four of these six -
# GeneM4, GeneM6, GeneM16, GeneM30 - are exactly the "four modules exhibiting significant
# associations with ASD that were only detectable in [BA17]" named in the paper's "Regional
# variation" section (GeneM30: OPC module w/ hub genes SOX4/SOX11; GeneM4: inhibitory
# neuron module w/ SCN9A). GeneM13 (BA7-only) and GeneM17 (BA17 + BA41/42/22) are not
# individually named/discussed in the paper, so they get no label below. Direction for all
# 6 is taken from the beta of their one (or, for GeneM17, both concordant-sign) significant
# region rather than Whole.Cortex_ASD_Beta: the whole-cortex beta is not just non-significant
# for these but can even have the opposite sign (GeneM30: whole-cortex beta -0.0019 [ns] vs
# BA17 beta +0.0120 [FDR=0.0385]), so it would give the wrong direction if used here.
#
# Direction: sign of Whole.Cortex_ASD_Beta for the 18 cortex-wide modules; sign of each
# module's own significant-region beta for the 6 regionally variable ones (see dict below).
# Naming: "_cortex" vs "_region" suffix makes the scope explicit in the set name itself, e.g.
# GeneM3_up_cortex (significant across the whole cortex) vs GeneM4_down_region (significant
# in at least one region, not whole-cortex).
#
# Labels: an author-given descriptive label exists for 10 of these modules, sourced from
# Fig. 3 (GeneM5, GeneM24, GeneM32) and Fig. 4a / main text "Regional variation" section
# (GeneM3, GeneM4, GeneM7, GeneM8, GeneM14, GeneM23, GeneM30) of the paper - not present in
# any supplementary data column, manually transcribed here. No per-set metadata mechanism
# exists in the nispace-data/.collect schema (ref.yaml collections only support
# whole-collection description/citations), so direction and label are encoded directly into
# the set name, as already done above for CellTypesSilettiClusters.
#
# The remaining 13 modules have no author-given label, so a label was INFERRED here (not from
# the paper) from each module's top Neural_Cell_Type enrichment + top GeneModule_Ontology GO
# term (both from Supp Data 6 / MOESM8). This heuristic was validated against the 10 known
# paper labels first: cell-type top-hit alone reconstructs the 5 purely cell-type-defined
# labels (microglia/oligo/endothelial/OPC/astrocyte) but is unreliable alone (gets GeneM7
# "immune response" wrong, predicting Endothelial); adding the top GO term fixes that case and
# reconstructs/supports 9 of 10 labels. Applying both signals together to the 13 unlabeled
# modules: where BOTH the cell-type FDR and the GO FDR are significant and thematically
# coherent, a specific inferred label is used (8 modules); where either signal is
# non-significant or the top GO terms don't cohere, the label is prefixed "uncertain (...)"
# to flag it as a weak/low-confidence guess rather than a confident functional call (5 modules).
asd_module_labels = {
    "M2": "astrocyte metabolism",
    "M3": "neuronal energy processes",
    "M4": "neuronal signal transduction",
    "M5": "synaptic plasticity",
    "M6": "oligodendrocyte RNA processing",
    "M7": "immune response",
    "M8": "reactive microglia",
    "M12": "uncertain (cell cycle / chromatin)",
    "M13": "uncertain (immune / vascular)",
    "M14": "neurite morphogenesis",
    "M15": "pericyte immune signaling",
    "M16": "synapse assembly",
    "M17": "neuronal protein regulation",
    "M19": "synaptic signaling",
    "M21": "OPC signaling",
    "M23": "oligo organelle regulation",
    "M24": "blood-brain barrier transport",
    "M25": "uncertain (RNA metabolism)",
    "M27": "ribosome biogenesis",
    "M30": "oligodendrocyte progenitor",
    "M32": "reactive astrocyte",
    "M33": "uncertain (chromatin regulation)",
    "M34": "uncertain (inflammatory signaling)",
}
# Modules significant in exactly one region (or, for M17, two concordant-sign regions) but
# NOT whole-cortex - see "Regionally variable extension" above. Value = region used for direction.
region_specific_modules = {"M4": "BA17", "M6": "BA17", "M13": "BA7", "M16": "BA17", "M17": "BA17", "M30": "BA17"}

# GeneM9 ("neural noncoding") is EXCLUDED despite being ASD-significant (whole-cortex FDR=0.0014,
# one of the strongest of the 24) because its defining "noncoding" property is essentially
# eliminated by nispace's mrna/magicc reproducibility-based gene dropout: of GeneM9's 241
# biotype-annotated genes, 97 are non-coding (lincRNA/pseudogene/antisense/etc.), of which only
# 3 pseudogenes survive dropout (0% survival for every other non-coding biotype) - vs. 47%
# survival for GeneM9's 144 protein-coding genes. What nispace actually delivers for "GeneM9"
# is therefore a 71-gene remnant that is ~96% protein-coding (68/71) - i.e., a set that no
# longer represents what the label describes, only an incidental leftover. Verified this is not
# a general problem: the other 10 labeled modules' paper-cited emblematic/hub genes (GRIN2A,
# MYO5A, BTRC for GeneM5; SOX4, SOX11 for GeneM30; SCN9A for GeneM4) all survive dropout in
# both mrna and magicc, and their aggregate dropout rates (30-48%) are unremarkable relative to
# the collection's 17-71% range - GeneM9's 71%/68% (mrna/magicc) dropout is a genuine outlier.
excluded_modules = {"M9"}

def _fix_excel_date_mangled_gene_symbol(v):
    # Excel autocorrects gene symbols like "MARCH4"/"SEPT9" into dates on file creation. Readers
    # then return either a datetime object (openpyxl) or a raw Excel serial date number - a plain
    # int/float counting days since 1899-12-30 (pandas' read_excel, seen with the Li et al. 2020
    # GEO file) - instead of the original string. Only the MARCH1-11 and SEPT1-15 gene families
    # are affected in the files used here (verified: all mangled cells resolve to month 3 or 9).
    # Reconstruct the original symbol from month/day in either representation.
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        v = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=v)
    month_prefix = {3: "MARCH", 9: "SEPT"}
    return f"{month_prefix[v.month]}{v.day}"

df_gandal_genes = pd.read_excel(
    download("https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-022-05377-7/MediaObjects/41586_2022_5377_MOESM7_ESM.xlsx"),
    sheet_name="Gene_Level",
)
kme_cols = [c for c in df_gandal_genes.columns if c.startswith("kME")]
df_gandal_genes = df_gandal_genes[["WGCNA_module", "external_gene_name"] + kme_cols]
df_gandal_genes["external_gene_name"] = df_gandal_genes["external_gene_name"].apply(_fix_excel_date_mangled_gene_symbol)
df_gandal_genes["module_n"] = df_gandal_genes["WGCNA_module"].str.split("_").str[0]  # "M5_green" -> "M5"
# each gene's kME to its OWN assigned module (kME{n}_{color} column <-> "{n}_{color}" module) -
# a continuous, per-gene module-membership-strength score, used below as an enrichment weight.
kme_col_by_module_n = {f"M{c.split('_')[0].replace('kME', '')}": c for c in kme_cols}
df_gandal_genes["own_kme"] = df_gandal_genes.apply(
    lambda r: r[kme_col_by_module_n[r["module_n"]]] if r["module_n"] in kme_col_by_module_n else np.nan,
    axis=1
)

region_beta_cols = sorted({f"ASD_{region}_Beta" for region in region_specific_modules.values()})
df_gandal_stats = pd.read_excel(
    download("https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-022-05377-7/MediaObjects/41586_2022_5377_MOESM8_ESM.xlsx"),
    sheet_name="GeneModules", header=1,
)[["Module", "Whole.Cortex_ASD_Beta", "Whole.Cortex_ASD_FDR"] + region_beta_cols]
df_gandal_stats["module_n"] = df_gandal_stats["Module"].str.replace("Gene", "").str.split("_").str[0]  # "GeneM5_green" -> "M5"

# resolve the set name for each of the 24 significant modules once, shared by both the
# unweighted (plain gene list) and weighted (kME-weighted) collections built below.
selected_modules = {}  # module_n -> set_name
for _, row in df_gandal_stats.iterrows():
    n = row["module_n"]
    if n in excluded_modules:
        continue
    if n in region_specific_modules:
        beta = row[f"ASD_{region_specific_modules[n]}_Beta"]
    elif row["Whole.Cortex_ASD_FDR"] < 0.05:
        beta = row["Whole.Cortex_ASD_Beta"]
    else:
        continue
    direction = "up" if beta > 0 else "down"
    scope = "region" if n in region_specific_modules else "cortex"
    set_name = f"Gene{n}_{direction}_{scope}"
    if n in asd_module_labels:
        set_name += f": {asd_module_labels[n]}"
    selected_modules[n] = set_name

collection_asd = {
    set_name: df_gandal_genes.loc[df_gandal_genes["module_n"] == n, "external_gene_name"].tolist()
    for n, set_name in selected_modules.items()
}
print(f"ASD modules (Gandal et al., 2022): {len(collection_asd)} modules "
      f"(expected 23: 18 whole-cortex significant + 6 regionally variable "
      f"[GeneM4, GeneM6, GeneM13, GeneM16, GeneM17, GeneM30], minus 1 excluded [GeneM9])")
write_json(collection_asd, ref_dir / "collection-ASDModulesGandal2022.collect")

# Weighted companion: same 24 modules/genes as above, but every member gene keeps its own_kme
# as a weight instead of being reduced to a binary in/out membership - avoids having to pick a
# hub/marker-gene cutoff (a fixed top-N or percentile threshold was considered but rejected:
# module "tightness" varies enough that any single threshold either empties out weakly-coherent
# modules or barely filters strongly-coherent ones - see conversation/commit history).
collection_asd_weighted = pd.concat(
    {
        set_name: df_gandal_genes.loc[df_gandal_genes["module_n"] == n]
            .set_index("external_gene_name")["own_kme"]
        for n, set_name in selected_modules.items()
    },
    axis=0, names=["set", "map"],
).rename("weight").astype(np.float32)
collection_asd_weighted.to_csv(ref_dir / "collection-ASDModulesGandal2022Weights.collect", index=True)

# FMR1/FMRP-related gene sets: CLIP-seq binding targets (Li et al., 2020, human iPSC-derived
# cells, plus 3 earlier groups' target lists bundled in that paper's own supplementary data)
# and RNA-editing-based gene sets (Tran et al., 2019) linking ASD and Fragile X syndrome.
#
# Source (CLIP targets): GEO GSE128860, file GSE128860_CLIP.xlsx, sheet "gene lists". Four
# columns are Li et al.'s own human CLIP-seq targets per cell type/lineage (dNPC/vNPC = dorsal/
# ventral forebrain neural progenitor cells; dNeuron/vNeuron = dorsal excitatory/ventral
# inhibitory neurons, all differentiated from human ESC/iPSC lines). Three further columns are
# OTHER groups' previously published FMRP target lists, reproduced by Li et al. for comparison
# (Ascano et al. 2012, human HEK293 - non-neuronal; Darnell et al. 2011, mouse brain; Maurin
# et al. 2018, mouse brain). Verified counts match the paper exactly: dNPC=1232, vNPC=1234,
# dNeuron=629, vNeuron=721 (union=1653); Darnell=844 (~842 commonly cited elsewhere). Excel
# gene-symbol date-mangling affects only the Ascano (9 cells) and Darnell (2 cells) columns.
#
# Source (RNA editing): Tran et al. 2019 Supplementary Tables 5 (module memberships of RNA-
# editing sites from WGCNA) and 7 (differential RNA-editing sites in Fragile X samples),
# obtained as the paper's own Nature-hosted supplementary files (local archive, not
# re-downloaded here, since Nature/PMC gate the direct download behind a JS/bot challenge).
# Table 5's "turquoise" module (per brain region) is the editing-site co-variation module
# significantly associated with ASD diagnosis - these genes are from idiopathic ASD brains,
# NOT Fragile X patients (their FXS relevance is only via the paper's own convergence finding
# with Table 7). Table 7 gives genes with direct differential RNA editing (Fisher's p<0.05,
# table is pre-filtered to significant sites) between Fragile X patients/carriers/controls, in
# two independent cohorts: NeuroBioBank (full-mutation FXS vs. carriers) and UC Davis (FXTAS -
# premutation carriers with tremor/ataxia syndrome vs. controls - a mechanistically distinct
# condition from full-mutation FXS, kept as its own set for that reason).
#
# Naming: {EvidenceType}_{Context}_{StudyTag} - no per-set metadata mechanism exists in the
# nispace-data/.collect schema (see ASDModulesGandal2022 above), so evidence type/species/
# tissue/cohort/study all have to be encoded in the set name itself. EvidenceType is one of
# FMR1Target (CLIP-seq binding target), ASDEditMod (RNA-editing co-module linked to ASD
# diagnosis - a module-membership call), or FXSEditDiff (direct differential RNA editing in
# Fragile-X-spectrum patients - a case-control significance call). EditMod and EditDiff are
# deliberately different suffixes: both are RNA-editing-based, but module co-membership and
# direct case-control difference are different statistical claims, not interchangeable. No
# weights are included for any set in this collection (unlike ASDModulesGandal2022's kME,
# there is no single quantity here that would mean the same thing across all 12 sets).
fmr1_target_labels = {
    "dNPC": "FMR1Target_dNPC_Li2020",
    "vNPC": "FMR1Target_vNPC_Li2020",
    "dNeuron": "FMR1Target_dNeuron_Li2020",
    "vNeuron": "FMR1Target_vNeuron_Li2020",
    "Ascano": "FMR1Target_NonNeuronal_Ascano2012",
    "Darnell": "FMR1Target_MouseBrain_Darnell2011",
    "Maurin": "FMR1Target_MouseBrain_Maurin2018",
}
df_fmr1 = pd.read_excel(
    download("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE128nnn/GSE128860/suppl/GSE128860_CLIP.xlsx"),
    sheet_name="gene lists",
)[list(fmr1_target_labels)]
collection_fmr1 = {
    set_name: df_fmr1[col].dropna().apply(_fix_excel_date_mangled_gene_symbol).tolist()
    for col, set_name in fmr1_target_labels.items()
}

tran_table5_path = nispace_source_data_path / "_archive" / "Tran2018" / "41593_2018_287_MOESM7_ESM.xlsx"
asd_editmod_sheets = {
    "5b": "ASDEditMod_FrontalCx_Tran2019",
    "5c": "ASDEditMod_TemporalCx_Tran2019",
    "5d": "ASDEditMod_Cerebellum_Tran2019",
}
for sheet, set_name in asd_editmod_sheets.items():
    df = pd.read_excel(tran_table5_path, sheet_name=sheet, header=2)
    collection_fmr1[set_name] = sorted(set(df.loc[df["moduleColor"] == "turquoise", "gene_name"].dropna()))

tran_table7_path = nispace_source_data_path / "_archive" / "Tran2018" / "41593_2018_287_MOESM9_ESM.xlsx"
fxs_editdiff_sheets = {
    "7b": "FXSEditDiff_NeuroBioBank_Tran2019",
    "7c": "FXSEditDiff_UCDavisFXTAS_Tran2019",
}
for sheet, set_name in fxs_editdiff_sheets.items():
    df = pd.read_excel(tran_table7_path, sheet_name=sheet, header=2)
    collection_fmr1[set_name] = sorted(set(df["gene_name"].dropna().apply(_fix_excel_date_mangled_gene_symbol)))

print(f"FMR1Targets: {len(collection_fmr1)} sets (expected 12: 7 FMR1Target + 3 ASDEditMod + 2 FXSEditDiff)")
write_json(collection_fmr1, ref_dir / "collection-FMR1Targets.collect")

# FXSEditDiffWeights: NOT a full weighted parallel of FMR1Targets (which has no weights at all -
# no single quantity applies across all 12 sets), just the 2 FXSEditDiff sets, weighted by
# editing_level_effect_size (Tran et al. 2019, Supplementary Table 7b/c) - an unsigned magnitude
# (0.06-0.9, verified no negative values) of how much a site's editing level differs between
# Fragile-X-spectrum patients and carriers/controls. ~1/3 of genes have multiple differentially
# edited sites (up to 32 in NeuroBioBank, 15 in UC Davis) - each gene's weight is the MAX
# effect size across its own sites (the single strongest piece of evidence for that gene,
# analogous to how ASDModulesGandal2022Weights uses one kME value per gene, not an aggregate
# across a network).
collection_fxs_weighted = {}
for sheet, set_name in fxs_editdiff_sheets.items():
    df = pd.read_excel(tran_table7_path, sheet_name=sheet, header=2).dropna(subset=["gene_name"])
    df["gene_name"] = df["gene_name"].apply(_fix_excel_date_mangled_gene_symbol)
    collection_fxs_weighted[set_name] = df.groupby("gene_name")["editing_level_effect_size"].max()

collection_fxs_weighted = pd.concat(collection_fxs_weighted, names=["set", "map"]).rename("weight").astype(np.float32)
collection_fxs_weighted.to_csv(ref_dir / "collection-FXSEditDiffWeights.collect", index=True)

# %% Psychiatric/neurological/neurodevelopmental disorder GWAS genes (Watanabe et al., 2019 GWAS ATLAS)
#
# Source: gwasATLAS_v20191115 bulk download (_archive/gwasATLAS_v20191115/), not the interactive
# atlas.ctglab.nl website. Two files used: the trait metadata table (id/PMID/N/Ncase/Ncontrol/...)
# and gwasATLAS_v20191115_magma_P.txt.gz - MAGMA gene-based association p-values (SNP-wise mean
# model, hg19, 1kb gene window both sides) for every gene x every one of 4,756 GWAS in the atlas.
# These are CTG Lab's own independently-computed output from public GWAS summary statistics
# (including several PGC studies) - not a redistribution of the underlying study's raw data, so
# PGC's "do not cross-post these data" clause (which restricts the summary-statistics file itself,
# not derived/computed output - see memory reference-gwas-data-licensing) does not apply here.
#
# Disorder + study selection: out of ~800 psychiatric/neurological/neurodevelopmental keyword or
# domain matches in the atlas, manually curated down to 21 actual diagnosed disorders (excluding
# UKB Mental Health Questionnaire symptom/severity items, medication/treatment-response traits,
# family-history-of-relative proxy phenotypes, and case-case contrasts like "Schizophrenia vs
# Bipolar disorder"), then one best-powered version selected per disorder (by effective sample
# size Neff = 4/(1/Ncase+1/Ncontrol), preferring genuine case-control designs over ancestry-arm
# duplicates of the same underlying study). 3 disorders (Migraine, Narcolepsy, NicotineDependence)
# have no independent source publication - Watanabe et al. themselves ran these directly on UK
# Biobank fields, so they cite only the GWAS ATLAS paper itself.
#
# Genes: MAGMA output uses Ensembl gene IDs, mapped to HGNC symbols via the HGNC complete gene
# set (single canonical, stable source), keeping only unambiguous 1:1 symbol<->Ensembl mappings
# (18,888 of 20,187 tested genes retained; the rest have no clean current HGNC symbol).
#
# Two collections:
# - GWASAtlas: binary, genome-wide-significant genes only (p < 0.05 / 20,187 genes tested
#   = 2.477e-6, the standard MAGMA/Bonferroni gene-wide threshold, computed BEFORE symbol mapping
#   to match MAGMA's actual multiple-testing burden). 4 of 21 disorders (AlcoholDependence,
#   AnxietyDisorder, FocalEpilepsy, PTSD) have zero genes at this threshold and are excluded
#   entirely from this collection (not included as empty sets).
# - GWASAtlasWeights: all 21 disorders, every tested gene, weighted by -log10(p). This is a
#   confidence/significance measure, not a direct effect-size estimate (MAGMA's gene-based test
#   already corrects for gene size/SNP density/LD via its null, but does not fully separate true
#   effect magnitude from per-gene tagging power - see memory reference-mrna-magicc-genesets for
#   the full reasoning). A threshold applied uniformly across all 21 sets (e.g. recovering
#   GWASAtlas from GWASAtlasWeights via weight > -log10(2.477e-6) = 5.606) is
#   statistically well-calibrated across disorders despite their very different sample sizes,
#   because MAGMA's null is the same test for every column; comparing raw weight *magnitude*
#   between disorders is not valid for the same reason (confounded by each disorder's own power).
#
# Sample size (n) in each set name is the plain total N as reported by the source study itself
# (not the Neff used for study selection) - e.g. MDD_n173005_Wray2018 is exactly Wray et al.
# 2018's public (non-23andMe) total: (135458-75607) + (344901-231747) = 59851+113154 = 173005,
# an exact match to the paper's stated 23andMe contribution subtracted from its headline number.
# Depression_n500199_Howard2019 is likewise the standard "PGC+UKB, 23andMe excluded" public
# release of Howard et al. 2019 (source file literally named PGC_UKB_depression_genome-wide.txt),
# smaller than the paper's 23andMe-inclusive headline number for the same reason.
gwasatlas_dir = nispace_source_data_path / "_archive" / "gwasATLAS_v20191115"

gwasatlas_disorders = {
    3982: "SCZ_n105318_Pardinas2018",
    4039: "BD_n74194_Ruderfer2018",
    4014: "MDD_n173005_Wray2018",
    4293: "Depression_n500199_Howard2019",
    3: "ADHD_n55374_Demontis2019",
    4037: "ASD_n46350_Grove2019",
    17: "AN_n14477_Duncan2017",
    4027: "AlcoholDependence_n52848_Walters2018",
    3689: "NicotineDependence_n244890_Watanabe2019",
    6: "AnxietyDisorder_n17310_Otowa2016",
    4042: "OCD_n9725_Arnold2018",
    16: "PTSD_n19884_Duncan2018",
    4094: "AD_n455258_Jansen2019",
    4167: "PD_n482730_Nalls2019",
    4010: "ALS_n80610_Nicolas2018",
    3819: "MS_n27148_Sawcer2011",
    4156: "Epilepsy_n44889_ILAE2018",
    1201: "FocalEpilepsy_n31467_ILAE2014",
    1200: "GGE_n28763_ILAE2014",
    3603: "Migraine_n289307_Watanabe2019",
    3234: "Narcolepsy_n384879_Watanabe2019",
}

df_magma = pd.read_csv(
    gwasatlas_dir / "gwasATLAS_v20191115_magma_P.txt.gz", sep="\t",
    usecols=["GENE"] + [str(i) for i in gwasatlas_disorders],
)
gwas_threshold = 0.05 / len(df_magma)

df_hgnc = pd.read_csv(
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    sep="\t", usecols=["symbol", "ensembl_gene_id"],
)
df_hgnc = df_hgnc.dropna(subset=["ensembl_gene_id"])
df_hgnc = df_hgnc[~df_hgnc["ensembl_gene_id"].duplicated(keep=False)]  # drop ambiguous 1:many
ensg_to_symbol = df_hgnc.set_index("ensembl_gene_id")["symbol"].to_dict()
df_magma["symbol"] = df_magma["GENE"].map(ensg_to_symbol)
df_magma = df_magma.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])

collection_gwasatlas_weighted = {}
for gwas_id, set_name in gwasatlas_disorders.items():
    collection_gwasatlas_weighted[set_name] = -np.log10(df_magma.set_index("symbol")[str(gwas_id)].dropna())
collection_gwasatlas_weighted = (
    pd.concat(collection_gwasatlas_weighted, names=["set", "map"]).rename("weight").astype(np.float32)
)
collection_gwasatlas_weighted.to_csv(ref_dir / "collection-GWASAtlasWeights.collect", index=True)

collection_gwasatlas = {}
for gwas_id, set_name in gwasatlas_disorders.items():
    s = df_magma.set_index("symbol")[str(gwas_id)].dropna()
    sig_genes = sorted(s[s < gwas_threshold].index.tolist())
    if len(sig_genes) > 0:
        collection_gwasatlas[set_name] = sig_genes
print(f"GWASAtlas: {len(collection_gwasatlas)} sets (of {len(gwasatlas_disorders)} disorders; "
      f"{len(gwasatlas_disorders) - len(collection_gwasatlas)} excluded for 0 genome-wide-significant genes)")
write_json(collection_gwasatlas, ref_dir / "collection-GWASAtlas.collect")

# %% Psychiatric-relevant genes from the Human Phenotype Ontology (HPO)
#
# Source: genes_to_phenotype.txt, HPO's gene<->phenotype annotation file (pinned to release
# v2026-06-23, a GitHub Releases asset - a release tag is the correct immutable anchor for an
# asset that is not part of the git tree, unlike a raw repo file pinned by commit hash).
# Unlike GWASAtlas, this reflects rare/monogenic disease-causing gene<->clinical-feature
# annotations (curated from OMIM/Orphanet case reports), not common-variant GWAS association -
# a complementary, not redundant, evidence type to GWASAtlas for the same disorder names.
#
# Term selection: started from the term HP:0025792 "Abnormal cognitive process", traced up to
# its parent HP:0011446 "Abnormality of mental function" as the working heading, then manually
# reviewed every one of its ~600 descendant terms across all 10 direct child branches
# (excluding Reduced consciousness, Micrographia and Synesthesia entirely at the branch level
# for being acute/critical-care neurology, a single Parkinsonian motor-sign leaf term, and 0
# genes respectively) - keeping 214 terms with >=1 annotated gene after excluding non-
# psychiatric noise found during review: physical/sensory symptoms like Photophobia/Polydipsia
# that happen to be filed under behavioral branches, motor speech/articulation disorders like
# Dysarthria, libido/sexual-physiology terms, generic neurologic-deterioration terms that do
# not specify cognitive involvement, and a false-cognate "Anorexia" term that in HPO means
# clinical loss-of-appetite, not the eating disorder - it is not even in this branch, its
# parent is "Abdominal symptom". Full per-term include/exclude/reason audit trail is in
# _archive/HPO/hpo_psych_term_tracker.csv.
#
# Two collections from the same 214-term review:
# - HPOAllPsychiatric: all 214 terms as individual sets (not flattened into one background
#   list) so users can pick their own combination downstream (set_size_range etc.).
# - HPOCorePsychiatric: 38 of the 214 that are actual diagnosis(-like) labels rather than
#   granular symptom terms (e.g. Schizophrenia, Autism, Bipolar affective disorder) - a subset
#   of HPOAllPsychiatric, not a separate/independent selection.
# "Psychiatric" is used in the DSM-5 nosological sense (which covers Neurodevelopmental
# Disorders/intellectual disability, Sleep-Wake Disorders and Neurocognitive Disorders/dementia
# under the same manual as Schizophrenia etc.), not narrowly "distinct from neurological" -
# genuinely neurological-disease-defining phenotypes (Parkinsonism, muscle weakness, aphasia
# from stroke) were never in scope, since HPO files their core symptoms under the separate
# "Abnormality of the nervous system" branch, not "Abnormality of mental function".
#
# No weights: HPO's only continuous-ish field (per-annotation "frequency", e.g. "15/15") mixes
# fraction strings, missing values and categorical HPO frequency-modifier term codes - usable
# in principle but not built here; binary set membership only for now.
#
# Gene identifiers are already HGNC symbols in this file (unlike GWASAtlas's Ensembl IDs) - no
# ID mapping needed. "-" is a missing-gene-symbol placeholder in the raw file, dropped.
# Dict values are (set_name, is_core_diagnosis) - set_name is the term name in CamelCase.

hpo_psychiatric_terms = {
    "HP:0000709": ("Psychosis", True),  # Psychosis
    "HP:0000710": ("Hyperorality",False),  # Hyperorality
    "HP:0000711": ("Restlessness",False),  # Restlessness
    "HP:0000712": ("EmotionalLability",False),  # Emotional lability
    "HP:0000713": ("Agitation",False),  # Agitation
    "HP:0000716": ("Depression", True),  # Depression
    "HP:0000717": ("Autism", True),  # Autism
    "HP:0000718": ("AggressiveBehavior", True),  # Aggressive behavior
    "HP:0000719": ("InappropriateBehavior",False),  # Inappropriate behavior
    "HP:0000721": ("LackOfSpontaneousPlay",False),  # Lack of spontaneous play
    "HP:0000722": ("CompulsiveBehaviors", True),  # Compulsive behaviors
    "HP:0000723": ("RestrictiveBehavior",False),  # Restrictive behavior
    "HP:0000725": ("PsychoticEpisodes",False),  # Psychotic episodes
    "HP:0000726": ("Dementia", True),  # Dementia
    "HP:0000727": ("FrontalLobeDementia",False),  # Frontal lobe dementia
    "HP:0000728": ("ReducedAbilityToFormPeerRelationships",False),  # Reduced ability to form peer relationships
    "HP:0000729": ("AutisticBehavior", True),  # Autistic behavior
    "HP:0000732": ("InflexibleAdherenceToRoutines",False),  # Inflexible adherence to routines
    "HP:0000733": ("MotorStereotypy", True),  # Motor stereotypy
    "HP:0000734": ("Disinhibition",False),  # Disinhibition
    "HP:0000736": ("ShortAttentionSpan",False),  # Short attention span
    "HP:0000737": ("Irritability",False),  # Irritability
    "HP:0000738": ("Hallucinations", True),  # Hallucinations
    "HP:0000739": ("Anxiety", True),  # Anxiety
    "HP:0000740": ("EpisodicParoxysmalAnxiety",False),  # Episodic paroxysmal anxiety
    "HP:0000741": ("Apathy",False),  # Apathy
    "HP:0000742": ("SelfMutilation",False),  # Self-mutilation
    "HP:0000744": ("LowFrustrationTolerance",False),  # Low frustration tolerance
    "HP:0000745": ("AbnormalDiminishedVolition",False),  # Abnormal diminished volition
    "HP:0000746": ("Delusion", True),  # Delusion
    "HP:0000748": ("InappropriateLaughter",False),  # Inappropriate laughter
    "HP:0000749": ("ParoxysmalBurstsOfLaughter",False),  # Paroxysmal bursts of laughter
    "HP:0000750": ("DelayedSpeechAndLanguageDevelopment",False),  # Delayed speech and language development
    "HP:0000751": ("PersonalityChanges",False),  # Personality changes
    "HP:0000752": ("Hyperactivity", True),  # Hyperactivity
    "HP:0000756": ("Agoraphobia", True),  # Agoraphobia
    "HP:0000757": ("LackOfInsight",False),  # Lack of insight
    "HP:0000758": ("AbnormalNonverbalCommunicativeBehavior",False),  # Abnormal nonverbal communicative behavior
    "HP:0000817": ("ReducedEyeContact",False),  # Reduced eye contact
    "HP:0001249": ("IntellectualDisability", True),  # Intellectual disability
    "HP:0001254": ("Lethargy",False),  # Lethargy
    "HP:0001256": ("MildIntellectualDisability", True),  # Mild intellectual disability
    "HP:0001262": ("ExcessiveDaytimeSomnolence",False),  # Excessive daytime somnolence
    "HP:0001268": ("MentalDeterioration",False),  # Mental deterioration
    "HP:0001289": ("Confusion",False),  # Confusion
    "HP:0001344": ("AbsentSpeech",False),  # Absent speech
    "HP:0001345": ("PsychoticMentation",False),  # Psychotic mentation
    "HP:0001483": ("EyePoking",False),  # Eye poking
    "HP:0002145": ("FrontotemporalDementia", True),  # Frontotemporal dementia
    "HP:0002167": ("AbnormalSpeechPattern",False),  # Abnormal speech pattern
    "HP:0002187": ("ProfoundIntellectualDisability", True),  # Profound intellectual disability
    "HP:0002300": ("Mutism",False),  # Mutism
    "HP:0002330": ("SleepAttack",False),  # Sleep attack
    "HP:0002332": ("LackOfPeerRelationships",False),  # Lack of peer relationships
    "HP:0002342": ("ModerateIntellectualDisability", True),  # Moderate intellectual disability
    "HP:0002354": ("MemoryImpairment",False),  # Memory impairment
    "HP:0002360": ("SleepDisturbance", True),  # Sleep disturbance
    "HP:0002367": ("VisualHallucination",False),  # Visual hallucination
    "HP:0002371": ("LossOfSpeech",False),  # Loss of speech
    "HP:0002381": ("Aphasia",False),  # Aphasia
    "HP:0002427": ("ExpressiveAphasia",False),  # Expressive aphasia
    "HP:0002439": ("FrontolimbicDementia",False),  # Frontolimbic dementia
    "HP:0002463": ("LanguageImpairment",False),  # Language impairment
    "HP:0002465": ("PoorSpeech",False),  # Poor speech
    "HP:0002474": ("ExpressiveLanguageDelay",False),  # Expressive language delay
    "HP:0002494": ("AbnormalRapidEyeMovementSleep",False),  # Abnormal rapid eye movement sleep
    "HP:0002519": ("HypnagogicHallucination",False),  # Hypnagogic hallucination
    "HP:0002549": ("DeficitInPhonologicShortTermMemory",False),  # Deficit in phonologic short-term memory
    "HP:0003763": ("Bruxism",False),  # Bruxism
    "HP:0006863": ("SevereExpressiveLanguageDelay",False),  # Severe expressive language delay
    "HP:0006889": ("BorderlineIntellectualDisability", True),  # Borderline intellectual disability
    "HP:0006896": ("HypnopompicHallucination",False),  # Hypnopompic hallucination
    "HP:0006977": ("DeficitInGrammar",False),  # Deficit in grammar
    "HP:0006979": ("SleepWakeCycleDisturbance",False),  # Sleep-wake cycle disturbance
    "HP:0007017": ("ProgressiveForgetfulness",False),  # Progressive forgetfulness
    "HP:0007018": ("AttentionDeficitHyperactivityDisorder", True),  # Attention deficit hyperactivity disorder
    "HP:0007064": ("ProgressiveLanguageDeterioration",False),  # Progressive language deterioration
    "HP:0007086": ("SocialAndOccupationalDeterioration",False),  # Social and occupational deterioration
    "HP:0007123": ("SubcorticalDementia",False),  # Subcortical dementia
    "HP:0007200": ("EpisodicHypersomnia",False),  # Episodic hypersomnia
    "HP:0007302": ("BipolarAffectiveDisorder", True),  # Bipolar affective disorder
    "HP:0008760": ("ViolentBehavior",False),  # Violent behavior
    "HP:0008762": ("RepetitiveCompulsiveBehavior",False),  # Repetitive compulsive behavior
    "HP:0008763": ("NoSocialInteraction",False),  # No social interaction
    "HP:0008765": ("AuditoryHallucination",False),  # Auditory hallucination
    "HP:0008767": ("SelfMutilationOfTongueAndLipsDueToInvoluntaryMovements",False),  # Self-mutilation of tongue and lips due to involuntary movements
    "HP:0008768": ("AbnormalSexualBehavior",False),  # Abnormal sexual behavior
    "HP:0008770": ("ObsessiveCompulsiveTrait", True),  # Obsessive-compulsive trait
    "HP:0009088": ("SpeechArticulationDifficulties",False),  # Speech articulation difficulties
    "HP:0010523": ("Alexia",False),  # Alexia
    "HP:0010524": ("DisturbedSensoryPerception",False),  # Disturbed sensory perception
    "HP:0010526": ("Dysgraphia",False),  # Dysgraphia
    "HP:0010534": ("TransientGlobalAmnesia",False),  # Transient global amnesia
    "HP:0010863": ("ReceptiveLanguageDelay",False),  # Receptive language delay
    "HP:0010864": ("SevereIntellectualDisability", True),  # Severe intellectual disability
    "HP:0010865": ("OppositionalDefiantDisorder", True),  # Oppositional defiant disorder
    "HP:0011346": ("MildExpressiveLanguageDelay",False),  # Mild expressive language delay
    "HP:0011351": ("ModerateReceptiveLanguageDelay",False),  # Moderate receptive language delay
    "HP:0011352": ("SevereReceptiveLanguageDelay",False),  # Severe receptive language delay
    "HP:0011856": ("Pica", True),  # Pica
    "HP:0011973": ("ParoxysmalLethargy",False),  # Paroxysmal lethargy
    "HP:0011999": ("Paranoia", True),  # Paranoia
    "HP:0012075": ("PersonalityDisorder", True),  # Personality disorder
    "HP:0012076": ("BorderlinePersonalityDisorder", True),  # Borderline personality disorder
    "HP:0012166": ("SkinPicking",False),  # Skin-picking
    "HP:0012167": ("HairPulling",False),  # Hair-pulling
    "HP:0012168": ("HeadBanging",False),  # Head-banging
    "HP:0012169": ("SelfBiting",False),  # Self-biting
    "HP:0012170": ("NailBiting",False),  # Nail-biting
    "HP:0012171": ("StereotypicalHandWringing",False),  # Stereotypical hand wringing
    "HP:0012172": ("StereotypicalBodyRocking",False),  # Stereotypical body rocking
    "HP:0012433": ("AbnormalSocialBehavior",False),  # Abnormal social behavior
    "HP:0012434": ("DelayedEarlyChildhoodSocialMilestoneDevelopment",False),  # Delayed early-childhood social milestone development
    "HP:0012523": ("OralAversion",False),  # Oral aversion
    "HP:0012671": ("Abulia",False),  # Abulia
    "HP:0012672": ("AkineticMutism",False),  # Akinetic mutism
    "HP:0012760": ("ReducedSocialResponsiveness",False),  # Reduced social responsiveness
    "HP:0025160": ("AbnormalTemperTantrums",False),  # Abnormal temper tantrums
    "HP:0025161": ("FrequentTemperTantrums",False),  # Frequent temper tantrums
    "HP:0025162": ("SevereTemperTantrums",False),  # Severe temper tantrums
    "HP:0025233": ("SleepParalysis",False),  # Sleep paralysis
    "HP:0025234": ("Parasomnia",False),  # Parasomnia
    "HP:0025235": ("NREMParasomnia",False),  # NREM parasomnia
    "HP:0025236": ("SleepWalking",False),  # Sleep walking
    "HP:0025268": ("Stuttering",False),  # Stuttering
    "HP:0025269": ("PanicAttack", True),  # Panic attack
    "HP:0025479": ("SelfNeglect",False),  # Self-neglect
    "HP:0025732": ("AbnormalSocialDevelopment",False),  # Abnormal social development
    "HP:0030082": ("AbnormalDrinkingBehavior",False),  # Abnormal drinking behavior
    "HP:0030212": ("Collectionism",False),  # Collectionism
    "HP:0030213": ("EmotionalDearth",False),  # Emotional dearth
    "HP:0030215": ("InappropriateCrying",False),  # Inappropriate crying
    "HP:0030216": ("Inertia",False),  # Inertia
    "HP:0030218": ("Punding",False),  # Punding
    "HP:0030219": ("SemanticDementia",False),  # Semantic dementia
    "HP:0030223": ("PerseverativeThought",False),  # Perseverative thought
    "HP:0030391": ("SpokenWordRecognitionDeficit",False),  # Spoken word recognition deficit
    "HP:0030765": ("SleepTerror",False),  # Sleep terror
    "HP:0030784": ("AnomicAphasia",False),  # Anomic aphasia
    "HP:0030955": ("AddictiveAlcoholUse",False),  # Addictive alcohol use
    "HP:0031258": ("Delirium",False),  # Delirium
    "HP:0031354": ("SleepOnsetInsomnia",False),  # Sleep onset insomnia
    "HP:0031355": ("MaintenanceInsomnia",False),  # Maintenance insomnia
    "HP:0031432": ("RestrictedOrRepetitiveBehaviorsOrInterests",False),  # Restricted or repetitive behaviors or interests
    "HP:0031434": ("AbnormalProsody",False),  # Abnormal prosody
    "HP:0031467": ("DysregulatedNegativeEmotionalState",False),  # Dysregulated negative emotional state
    "HP:0031468": ("SeparationInsecurity",False),  # Separation insecurity
    "HP:0031469": ("LowSelfEsteem",False),  # Low self-esteem
    "HP:0031473": ("Anger",False),  # Anger
    "HP:0031589": ("SuicidalIdeation", True),  # Suicidal ideation
    "HP:0031843": ("AbnormallySlowThoughtProcess",False),  # Abnormally slow thought process
    "HP:0031844": ("Euphoria",False),  # Euphoria
    "HP:0031849": ("SleepWakeInversion",False),  # Sleep-wake inversion
    "HP:0031873": ("EarlyChronotype",False),  # Early chronotype
    "HP:0031874": ("LateChronotype",False),  # Late chronotype
    "HP:0031943": ("Akathisia",False),  # Akathisia
    "HP:0031987": ("DiminishedAbilityToConcentrate",False),  # Diminished ability to concentrate
    "HP:0032508": ("Polyembolokoilamania",False),  # Polyembolokoilamania
    "HP:0032509": ("Onychotillomania",False),  # Onychotillomania
    "HP:0032521": ("SelfHugging",False),  # Self hugging
    "HP:0032936": ("IntrusionSymptom",False),  # Intrusion symptom
    "HP:0033051": ("ImpairedExecutiveFunctioning",False),  # Impaired executive functioning
    "HP:0033052": ("NonEpilepticSeizure",False),  # Non-epileptic seizure
    "HP:0033063": ("ShortenedSleepPhase",False),  # Shortened sleep phase
    "HP:0033511": ("AddictiveSubstanceUse",False),  # Addictive substance use
    "HP:0033687": ("ShortTermMemoryImpairment",False),  # Short term memory impairment
    "HP:0033689": ("AnterogradeMemoryImpairment",False),  # Anterograde memory impairment
    "HP:0033705": ("Tearfulness",False),  # Tearfulness
    "HP:0033838": ("Dysphoria",False),  # Dysphoria
    "HP:0034435": ("AbnormalEyeContact",False),  # Abnormal eye contact
    "HP:0040082": ("HappyDemeanor",False),  # Happy demeanor
    "HP:0100023": ("RecurrentHandFlapping",False),  # Recurrent hand flapping
    "HP:0100024": ("ConspicuouslyHappyDisposition",False),  # Conspicuously happy disposition
    "HP:0100025": ("Overfriendliness",False),  # Overfriendliness
    "HP:0100033": ("Tics", True),  # Tics
    "HP:0100034": ("MotorTics",False),  # Motor tics
    "HP:0100035": ("PhonicTics",False),  # Phonic tics
    "HP:0100543": ("CognitiveImpairment",False),  # Cognitive impairment
    "HP:0100703": ("TongueThrusting",False),  # Tongue thrusting
    "HP:0100710": ("Impulsivity", True),  # Impulsivity
    "HP:0100716": ("SelfInjuriousBehavior", True),  # Self-injurious behavior
    "HP:0100738": ("AbnormalEatingBehavior",False),  # Abnormal eating behavior
    "HP:0100739": ("Bulimia", True),  # Bulimia
    "HP:0100753": ("Schizophrenia", True),  # Schizophrenia
    "HP:0100754": ("Mania", True),  # Mania
    "HP:0100785": ("Insomnia", True),  # Insomnia
    "HP:0100851": ("AbnormalEmotionalState",False),  # Abnormal emotional state
    "HP:0100852": ("AbnormalFearInducedBehavior",False),  # Abnormal fear-induced behavior
    "HP:0100962": ("ExcessiveShyness",False),  # Excessive shyness
    "HP:0410291": ("Negativism",False),  # Negativism
    "HP:4000064": ("PoorSleep",False),  # Poor sleep
    "HP:4000070": ("FixatedInterests",False),  # Fixated interests
    "HP:4000073": ("PronounReversal",False),  # Pronoun reversal
    "HP:4000076": ("UseOfAnotherPerson'sBodyToCommunicate",False),  # Use of another person's body to communicate
    "HP:4000079": ("SensorySeeking",False),  # Sensory seeking
    "HP:4000081": ("ReducedProductionOfGestures",False),  # Reduced production of gestures
    "HP:5200001": ("SelectiveMutism",False),  # Selective mutism
    "HP:5200018": ("StereotypicUpperExtremityMovements",False),  # Stereotypic upper-extremity movements
    "HP:5200021": ("ReducedSocialInsight",False),  # Reduced social insight
    "HP:5200044": ("ReducedAttentionRegulation",False),  # Reduced attention regulation
    "HP:5200045": ("ReducedImpulseControl",False),  # Reduced impulse control
    "HP:5200057": ("LimitedPointing",False),  # Limited pointing
    "HP:5200069": ("Spinning",False),  # Spinning
    "HP:5200102": ("ReducedSocialSmiling",False),  # Reduced social smiling
    "HP:5200134": ("Jumping",False),  # Jumping
    "HP:5200136": ("ReducedSocialReciprocity",False),  # Reduced social reciprocity
    "HP:5200218": ("Derealization",False),  # Derealization
    "HP:5200291": ("REMSleepBehaviorDisorder",False),  # REM sleep behavior disorder
    "HP:5200310": ("DiminishmentOfSocialInteractions",False),  # Diminishment of social interactions
    "HP:5200321": ("AmplificationOfSexualBehavior",False),  # Amplification of sexual behavior
    "HP:5200330": ("SuicideBehaviors",False),  # Suicide behaviors
    "HP:5200356": ("SleepOnsetRapidEyeMovementPeriod",False),  # Sleep onset rapid eye movement period
    "HP:5200360": ("ShortREMSleep",False),  # Short REM sleep
    "HP:6000029": ("SocialAnxiety",False),  # Social anxiety
}

df_hpo = pd.read_csv(
    "https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2026-06-23/genes_to_phenotype.txt",
    sep="\t",
)
df_hpo = df_hpo[df_hpo["gene_symbol"] != "-"]
df_hpo = df_hpo[df_hpo["hpo_id"].isin(hpo_psychiatric_terms)]

collection_hpo_all = {}
collection_hpo_core = {}
for hpo_id, (set_name, is_core) in hpo_psychiatric_terms.items():
    genes = sorted(set(df_hpo.loc[df_hpo["hpo_id"] == hpo_id, "gene_symbol"]))
    collection_hpo_all[set_name] = genes
    if is_core:
        collection_hpo_core[set_name] = genes

print(f"HPOAllPsychiatric: {len(collection_hpo_all)} sets (expected 214)")
print(f"HPOCorePsychiatric: {len(collection_hpo_core)} sets (expected 38)")
write_json(collection_hpo_all, ref_dir / "collection-HPOAllPsychiatric.collect")
write_json(collection_hpo_core, ref_dir / "collection-HPOCorePsychiatric.collect")

# %%

