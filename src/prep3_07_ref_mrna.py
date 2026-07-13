# %% Init

import sys
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
from nispace.io import load_labels, write_json, read_msigdb_json, read_json
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

# Protein Atlas — brain expression categories
collection_pa = {}
for s, url in [
    ("ExpressedElevated",      "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("ExpressedNotElevated",   "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BNot+detected+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("ExpressedLowSpecificity","https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BLow+tissue+specificity+AND+NOT+tissue_category_rna%3Abrain%3BNot+detected?format=tsv&download=yes"),
    ("NotInBrain",             "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BNot+detected+AND+NOT+tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes"),
    ("NotInTissue",            "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes"),
]:
    collection_pa[s] = sorted(pd.read_table(url)["Gene"].unique().tolist())
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

genes_in_brain = read_json(ref_dir / "collection-ProteinAtlas.collect")
genes_in_brain = genes_in_brain["ExpressedElevated"] + genes_in_brain["ExpressedNotElevated"] + genes_in_brain["ExpressedLowSpecificity"]
expr_matrix = expr_matrix.loc[expr_matrix.index.isin(genes_in_brain)]
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
# modules - rather than the combined 31 (18+13).
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
# Labels: an author-given descriptive label exists for 11 of these 24 modules, sourced from
# Fig. 3 (GeneM5, GeneM9, GeneM24, GeneM32) and Fig. 4a / main text "Regional variation"
# section (GeneM3, GeneM4, GeneM7, GeneM8, GeneM14, GeneM23, GeneM30) of the paper - not
# present in any supplementary data column, manually transcribed here. No per-set metadata
# mechanism exists in the nispace-data/.collect schema (ref.yaml collections only support
# whole-collection description/citations), so direction and label are encoded directly into
# the set name, as already done above for CellTypesSilettiClusters.
asd_module_labels = {
    "M3": "neuronal energy processes",
    "M4": "neuronal signal transduction",
    "M5": "synaptic plasticity",
    "M7": "immune response",
    "M8": "reactive microglia",
    "M9": "neural noncoding",
    "M14": "neurite morphogenesis",
    "M23": "oligo organelle regulation",
    "M24": "blood-brain barrier transport",
    "M30": "oligodendrocyte progenitor",
    "M32": "reactive astrocyte",
}
# Modules significant in exactly one region (or, for M17, two concordant-sign regions) but
# NOT whole-cortex - see "Regionally variable extension" above. Value = region used for direction.
region_specific_modules = {"M4": "BA17", "M6": "BA17", "M13": "BA7", "M16": "BA17", "M17": "BA17", "M30": "BA17"}

def _fix_excel_date_mangled_gene_symbol(v):
    # Excel autocorrects gene symbols like "MARCH4"/"SEPT9" into dates on file creation;
    # openpyxl then reads these cells back as datetime objects instead of strings. Only
    # the MARCH1-11 and SEPT1-15 gene families are affected in this file (verified: all
    # mangled cells have month 3 or 9). Reconstruct the original symbol from month/day.
    if isinstance(v, str):
        return v
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
      f"(expected 24: 18 whole-cortex significant + 6 regionally variable "
      f"[GeneM4, GeneM6, GeneM13, GeneM16, GeneM17, GeneM30])")
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

# %%
