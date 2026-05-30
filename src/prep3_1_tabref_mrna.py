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
from utils import load_parc_lists

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
    mRNA_tab.to_csv(
        nispace_source_data_path / "reference" / "mrna" / "tab" / f"dset-mrna_parc-{parc}.csv.gz"
    )

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
    stability.to_csv(stab_repro_path)
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

# %%
