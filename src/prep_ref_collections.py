# %% imports ---------------------------------------------------------------------------------------

from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
import shutil
import tarfile
import pyreadr
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

# Nispace
wd = Path.cwd().parent
print(f"Working dir: {wd}")
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import fetch_reference
from nispace.io import write_json, read_msigdb_json, read_json
from nispace.utils.utils import _rm_ext
from nispace.utils.utils_datasets import download

# nispace data path 
nispace_source_data_path = wd


# %% PET collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

pet_files = fetch_reference("pet")
pet_files = [f.name.split("_space")[0] for f in pet_files]

# All
pd.Series(pet_files, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-All.collect", index=None)
    
# AllTargetSets
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in pet_files],
    "x": pet_files,
    "weight": [f.split("_")[2].split("-")[1] for f in pet_files],
}) \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-AllTargetSets.collect", index=None)
    
# UniqueTracers
collection = {
    "General": [
        "target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023",
        # "CBF"
        "target-SV2A_tracer-ucbj_n-76_dx-hc_pub-finnema2016",
        "target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016",
        "target-VMAT2_tracer-dtbz_n-76_dx-hc_pub-larsen2020"
    ],
    "Immunity": [
        'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018', 
        'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020'
    ],
    "Glutamate": [
        "target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019",
        "target-NMDA_tracer-ge179_n-29_dx-hc_pub-galovic2021"
    ],
    "GABA": [
        "target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022",
        "target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018"
    ],
    "Dopamine": [
        "target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018",
        "target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017",
        "target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015",
        "target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018",
    ],
    "Serotonin": [
        "target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012",
        "target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012",
        "target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012",
        "target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017",
        "target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018",
        "target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012",
    ],
    "Noradrenaline/Acetylcholine": [
        "target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017",
        "target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016",
        "target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020",
        "target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017",
    ],
    "Opiods/Endocannabinoids": [
        "target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020",
        "target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018",
        "target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015",
    ],
    "Histamine": [
        "target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017",
    ]
}
write_json(
    collection,
    nispace_source_data_path / "reference" / "pet" / "collection-UniqueTracers.collect"
)

# UniqueTracerSets
collection = [
    'target-5HT1a_tracer-way100635_n-35_dx-hc_pub-savli2012',
    'target-5HT1b_tracer-p943_n-23_dx-hc_pub-savli2012',
    'target-5HT1b_tracer-p943_n-65_dx-hc_pub-gallezot2010',
    'target-5HT2a_tracer-altanserin_n-19_dx-hc_pub-savli2012',
    'target-5HT4_tracer-sb207145_n-59_dx-hc_pub-beliveau2017',
    'target-5HT6_tracer-gsk215083_n-30_dx-hc_pub-radhakrishnan2018',
    'target-5HTT_tracer-dasb_n-100_dx-hc_pub-beliveau2017',
    'target-5HTT_tracer-dasb_n-18_dx-hc_pub-savli2012',
    'target-A4B2_tracer-flubatine_n-30_dx-hc_pub-hillmer2016',
    'target-CB1_tracer-omar_n-77_dx-hc_pub-normandin2015',
    'target-CMRglu_tracer-fdg_n-20_dx-hc_pub-castrillon2023',
    'target-COX1_tracer-ps13_n-11_dx-hc_pub-kim2020',
    'target-D1_tracer-sch23390_n-13_dx-hc_pub-kaller2017',
    'target-D23_tracer-flb457_n-37_dx-hc_pub-smith2017',
    'target-D23_tracer-flb457_n-55_dx-hc_pub-sandiego2015',
    'target-DAT_tracer-fpcit_n-174_dx-hc_pub-dukart2018',
    'target-DAT_tracer-fpcit_n-30_dx-hc_pub-garciagomez2013',
    'target-FDOPA_tracer-fluorodopa_n-12_dx-hc_pub-garciagomez2018',
    'target-GABAa_tracer-flumazenil_n-16_dx-hc_pub-norgaard2021',
    'target-GABAa_tracer-flumazenil_n-6_dx-hc_pub-dukart2018',
    'target-GABAa5_tracer-ro154513_n-10_dx-hc_pub-lukow2022',
    'target-H3_tracer-gsk189254_n-8_dx-hc_pub-gallezot2017',
    'target-HDAC_tracer-martinostat_n-8_dx-hc_pub-wey2016',
    'target-KOR_tracer-ly2795050_n-28_dx-hc_pub-vijay2018',
    'target-M1_tracer-lsn3172176_n-24_dx-hc_pub-naganawa2020',
    'target-mGluR5_tracer-abp688_n-22_dx-hc_pub-rosaneto',
    'target-mGluR5_tracer-abp688_n-28_dx-hc_pub-dubois2015',
    'target-mGluR5_tracer-abp688_n-73_dx-hc_pub-smart2019',
    'target-MOR_tracer-carfentanil_n-204_dx-hc_pub-kantonen2020',
    'target-MOR_tracer-carfentanil_n-39_dx-hc_pub-turtonen2021',
    'target-NET_tracer-mrb_n-10_dx-hc_pub-hesse2017',
    'target-NET_tracer-mrb_n-77_dx-hc_pub-ding2010',
    'target-SV2A_tracer-ucbj_n-76_dx-hc_pub-finnema2016',
    'target-TSPO_tracer-pbr28_n-6_dx-hc_pub-lois2018',
    'target-VAChT_tracer-feobv_n-18_dx-hc_pub-aghourian2017',
    'target-VAChT_tracer-feobv_n-4_dx-hc_pub-tuominen',
    'target-VAChT_tracer-feobv_n-5_dx-hc_pub-bedard2019',
    "target-VMAT2_tracer-dtbz_n-76_dx-hc_pub-larsen2020"
]
pd.DataFrame({
    "set": [f.split("_")[0].split("-")[1] for f in collection],
    "x": collection,
    "weight": [f.split("_")[2].split("-")[1] for f in collection],
}) \
    .to_csv(nispace_source_data_path / "reference" / "pet" / "collection-UniqueTracerSets.collect", index=None)


# %% mRNA collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# TODO: brainspan, add weighted cell types
# TODO: add GWAS from PGC after mapping to genes ("35 kb upstream and 10 kb downstream")

# All genes
mRNA_tab_files = (nispace_source_data_path / "reference" / "mrna" / "tab").glob("*.csv.gz")
all_genes = set()
for f in mRNA_tab_files:
    all_genes.update(pd.read_csv(f, index_col=0).index.unique())
all_genes = pd.Series(sorted(list(all_genes)), name="map")
all_genes.to_csv(nispace_source_data_path / "reference" / "mrna" / "collection-All.collect", index=False)

# Siletti et al cell types
df = (
    pd.read_excel("https://raw.githubusercontent.com/linnarsson-lab/adult-human-brain/refs/heads/main/tables/cluster_annotation.xlsx")
    .dropna(how="all")
    .loc[:, ["Supercluster", "Cluster name", "Top Enriched Genes"]]
)
sets = {
    'Upper-layer intratelencephalic': 'Upper-layer IT',
    'Deep-layer intratelencephalic': 'Deep-layer IT',
    'Deep-layer near-projecting': 'Deep-layer NP', 
    'Deep-layer corticothalamic and 6b': 'Deep-layer CT/6b',
    'MGE interneuron': 'MGE interneuron', 
    'CGE interneuron': 'CGE interneuron',
    'LAMP5-LHX6 and Chandelier': 'LAMP5-LHX6/Chandelier', 
    'Hippocampal CA1-3': "Hippocampus CA1-3",
    'Hippocampal CA4': "Hippocampus CA4", 
    'Hippocampal dentate gyrus': "Hippocampus DG",
    'Amygdala excitatory': "Amygdala excitatory",
    'Medium spiny neuron': "MSN", 
    'Eccentric medium spiny neuron': "Eccentric MSN", 
    'Splatter': "Splatter",
    'Mammillary body': "Mammillary body", 
    'Thalamic excitatory': "Thalamic excitatory",
    'Midbrain-derived inhibitory': "Midbrain-derived inhibitory",
    'Upper rhombic lip': "Upper rhombic lip", 
    'Cerebellar inhibitory': "Cerebellar inhibitory", 
    'Lower rhombic lip': "Lower rhombic lip",
    'Astrocyte': "Astrocyte", 
    'Oligodendrocyte': "Oligodendrocyte",
    'Oligodendrocyte precursor': "OPC",
    'Committed oligodendrocyte precursor': "Committed OPC", 
    'Microglia': "Microglia", 
    'Bergmann glia': "Bergmann glia", 
    'Vascular': "Vascular", 
    'Choroid plexus': "Choroid plexus",
    'Fibroblast': "Fibroblast", 
    'Ependymal': "Ependymal",  
    'Miscellaneous': "Miscellaneous",
}
# Clusters
collection = {}
for s in sets.keys():
    for x in df.query("Supercluster==@s")["Cluster name"].unique().tolist():
        collection[f"{sets[s]} - {x}"] = df.query("`Cluster name`==@x")["Top Enriched Genes"].str.split(", ").explode().unique().tolist()
write_json(
    collection,
    nispace_source_data_path / "reference" / "mrna" / "collection-CellTypesSilettiClusters.collect"
)
# Superclusters
collection = {}
for s in sets.keys():
    if s != "Miscellaneous":
        collection[sets[s]] = (
            df
            .query("Supercluster==@s")["Top Enriched Genes"]
            .str
            .split(", ").explode().unique().tolist()
        )
write_json(
    collection,
    nispace_source_data_path / "reference" / "mrna" / "collection-CellTypesSilettiSuperclusters.collect"
)

# PsychEncode cell types: Darmanis 2015 / Lake 2016 vs.  Lake 2018
for collection, save_name in zip(
    [pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/"
                   "DER-19_Single_cell_markergenes_TPM.xlsx") \
        .rename(columns=dict(GeneName="gene", CellType="set")),
     pd.read_excel("http://resource.psychencode.org/Datasets/Derived/SC_Decomp/"
                   "DER-21_Single_cell_markergenes_UMI.xlsx", header=1) \
        .rename(columns=dict(Gene="gene", Cluster="set"))],
    ["CellTypesPsychEncodeTPM", 
     "CellTypesPsychEncodeUMI"]
):
    collection = collection.astype(str)
    collection = {k: sorted(collection.query("set==@k").gene.unique()) for k in collection.set.unique()}
    all_genes = sum([collection[k] for k in collection], [])
    print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
    write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-{save_name}.collect")

# SynGO
url = "https://syngoportal.org/data/SynGO_bulk_download_release_20231201.zip"
path = download(url)
zip_file = zipfile.ZipFile(path)
with zip_file.open("syngo_ontologies.xlsx") as file:
    df = pd.read_excel(file)
collection = {name: genes.split(", ") 
              for id, name, genes in zip(df["id"], df["name"], df["hgnc_symbol"])}
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-SynGO.collect")

# GO
for name, short_name in [
    ("GOBiologicalProcess", "bp"),
    ("GOCellularComponent", "cc"),
    ("GOMolecularFunction", "mf"),
]:
    collection = {}
    for k, v in read_msigdb_json(download(
        f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c5.go.{short_name}.v2025.1.Hs.json"
    )).items():
        collection[k.replace(f"GO{short_name.upper()}_", "")] = v
    all_genes = sum([collection[k] for k in collection], [])
    print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
    write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-{name}.collect")

# Chromosome locations
collection = {}
for k, v in read_msigdb_json(download(
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c1.all.v2025.1.Hs.json"
)).items():
    
    # replace keys directly
    if "chr" not in k:
        k = k
    else:
        chr = k.replace("chr","").split("p")[0].split("q")[0]
        if chr not in ["X", "Y"]:
            chr = chr.zfill(2)
        pq = ('p' if 'p' in k else 'q') + k.split("p")[-1].split("q")[-1].zfill(2)
        k = f"chr{chr}{pq}"
    # save
    collection[k] = v
# sort by keys
collection = {k: collection[k] for k in sorted(collection.keys())}
# save
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-Chromosome.collect")

# Cortical layers
# https://elifesciences.org/reviewed-preprints/86933v2
wagstyl2024_tableS2 = pd.read_excel("https://cdn.elifesciences.org/articles/86933/elife-86933-supp2-v1.xlsx")
sets = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"]
collection = {
    s.lower().replace(" ",""): wagstyl2024_tableS2[wagstyl2024_tableS2[s]==True]["gene.symbol"].unique().tolist()
    for s in sets
}    
all_genes = sum([collection[k] for k in collection], [])
print(len(collection), "sets,", len(all_genes), "genes,", len(set(all_genes)), "unique.")
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-CorticalLayers.collect")

# Tissue-specific gene expression
collection = {}
for s, url in [
    ("ExpressedElevated", "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("ExpressedNotElevated", "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BTissue+enriched%2CGroup+enriched%2CTissue+enhanced+NOT+tissue_category_rna%3Abrain%3BNot+detected+AND+sort_by%3Atissue+specific+score?format=tsv&download=yes"),
    ("ExpressedLowSpecificity", "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BLow+tissue+specificity+AND+NOT+tissue_category_rna%3Abrain%3BNot+detected?format=tsv&download=yes"),
    ("NotInBrain", "https://www.proteinatlas.org/search/tissue_category_rna%3Abrain%3BNot+detected+AND+NOT+tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes"),
    ("NotInTissue", "https://www.proteinatlas.org/search/tissue_category_rna%3AAny%3BNot+detected?format=tsv&download=yes")
]:
    df = pd.read_table(url)
    print(s, len(df["Gene"].unique()))
    collection[s] = sorted(df["Gene"].unique().tolist())
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-ProteinAtlas.collect")

#%%
# load data from R ABAEnrichment package hosted at:
# https://bioconductor.org/packages/ABAData/

# get dev effect and expression data
url = "https://mghp.osn.xsede.org/bir190004-bucket01/archive.bioconductor.org/packages/3.2/data/experiment/src/contrib/ABAData_1.0.0.tar.gz"
fp = wd / "_archive" / "ABAData_1.0.0.tar.gz"
if not fp.exists():
    download(url, fp)
with tarfile.open(fp, "r:gz") as tar:
    for member in tar.getmembers():
        if member.name.endswith("dataset_dev_effect.rda") or member.name.endswith("dataset_5_stages.rda"):
            tar.extract(member, path=wd / "_archive")
            
# read rdata files
expression = pyreadr.read_r(wd / "_archive" / "ABAData" / "data" / "dataset_5_stages.rda")
expression = list(expression.values())[0]

# mappings for developmental stages
stages = {
    1: "prenatal", 
    2: "infant", 
    3: "child", 
    4: "adolescent", 
    5: "adult"
}

# mappings for regions
regions = {
    10194: "OFC", #'OFC_orbital frontal cortex',
    10173: "dlPFC", #'DFC_dorsolateral prefrontal cortex',
    10185: "vlPFC", #'VFC_ventrolateral prefrontal cortex',
    10278: 'ACC', #'MFC_anterior (rostral) cingulate (medial prefrontal) cortex',
    10163: "M1C", #'M1C_primary motor cortex (area M1, area 4)',
    10209: "S1C", #'S1C_primary somatosensory cortex (area S1, areas 3,1,2)',
    10225: "IPC", #'IPC_posteroventral (inferior) parietal cortex',
    10236: "A1C", #'A1C_primary auditory cortex (core)',
    10243: 'STC', #'STC_posterior (caudal) superior temporal cortex (area 22c)',
    10252: 'ITC', #'ITC_inferolateral temporal cortex (area TEv, area 20)',
    10269: 'V1C', #'V1C_primary visual cortex (striate cortex, area V1/17)',
    10294: 'HIP', #'HIP_hippocampus (hippocampal formation)',
    10361: "AMY", #'AMY_amygdaloid complex',
    10333: 'STR', #'STR_striatum',
    10398: "mdTHA", #'MD_mediodorsal nucleus of thalamus',
    10657: "CBC", #'CBC_cerebellar cortex'
 }

# sort into final matrix form
expression = expression.rename(columns={"hgnc_symbol": "gene_symbol", "structure": "region", "signal": "expression", "age_category": "stage"})
expression = expression[["gene_symbol", "region", "stage", "expression"]]
expression["stage"] = expression["stage"].replace(stages)
expression["region"] = expression["region"].replace(regions)
expression["region_stage"] = [f"{s1}-{s2}" for s1, s2 in zip(expression.stage, expression.region)]
expression_matrix = expression[["gene_symbol", "region_stage", "expression"]] \
    .pivot_table(columns="region_stage", index="gene_symbol").droplevel(0, axis=1)
    
# genes in brain according to protein atlas
genes_in_brain = read_json(wd / "reference" / "mrna" / "collection-ProteinAtlas.collect")
genes_in_brain = genes_in_brain["ExpressedElevated"] + genes_in_brain["ExpressedNotElevated"] + genes_in_brain["ExpressedLowSpecificity"]
genes_in_brain = sorted(list(set(genes_in_brain)))
expression_matrix = expression_matrix.loc[expression_matrix.index.isin(genes_in_brain)]
print(expression_matrix.shape)

# identify marker genes
def identify_marker_genes(expression_matrix, 
                          p_correction="fdr_bh",
                          min_expression=0.1):
    """
    Identify marker genes for each region-stage combination
    
    Parameters:
    -----------
    expression_matrix : pd.DataFrame
        Genes (rows) x Region-Stage combinations (columns)
    region_stage_labels : list
        Labels for each column
    p_threshold : float
        Significance threshold after correction
    fc_threshold : float
        Minimum log2 fold change
    min_expression : float
        Minimum expression threshold necessary to keep a gene in the analysis.
        Calculated for the mean expression of each gene across all region-stage combinations.
    
    Returns:
    --------
    dict : Marker genes for each combination
    """
    print(f"Expression matrix shape before removing genes with no expression: {expression_matrix.shape}")
    expression_matrix = expression_matrix[np.not_equal(expression_matrix.sum(axis=1), 0)]
    print(f"Expression matrix shape after removing genes with no expression: {expression_matrix.shape}")
    
    # convert expression matrix to pseudocounts and log2 transform, then save as array
    expression_matrix = np.log2(expression_matrix + 1)
    
    # apply min_expression threshold
    if min_expression is not None:
        print(f"Expression matrix shape before min_expression threshold: {expression_matrix.shape}")
        expression_matrix_mean = expression_matrix.mean(axis=1)
        expression_matrix = expression_matrix[expression_matrix_mean > expression_matrix_mean.quantile(min_expression)]
        print(f"Expression matrix shape after min_expression threshold: {expression_matrix.shape}")
        
    # to array
    expression_matrix_array = expression_matrix.values
    
    # initialize matrices
    p_matrix = np.full_like(expression_matrix, np.nan)
    fc_matrix = np.full_like(expression_matrix, np.nan)
    pc_matrix = np.full_like(expression_matrix, np.nan)
    n_genes = expression_matrix.shape[0]
    n_conditions = expression_matrix.shape[1]
    
    # iterate columns = conditions
    for i_condition in tqdm(range(n_conditions), desc="Conditions"):
        
        # Data for target condition
        target_expr = expression_matrix_array[:, i_condition]
        
        # Data for other conditions
        other_expr = expression_matrix_array[:, np.arange(n_conditions) != i_condition]
        
        # Iterate genes
        for i_gene in range(n_genes):
            target_val = target_expr[i_gene]
            other_vals = other_expr[i_gene, :]
            
            # t-test
            t_stat, p_val = stats.ttest_1samp(other_vals, target_val, alternative="less")
            
            # save p value and fold change
            p_matrix[i_gene, i_condition] = p_val
            with np.errstate(divide='ignore'):
                fc = np.log2(target_val / other_vals.mean())
                if np.isinf(fc):
                    fc = 0.0
                fc_matrix[i_gene, i_condition] = fc
                
        # multiple testing
        pc_matrix[:, i_condition] = multipletests(p_matrix[:, i_condition], method=p_correction)[1]
            
    # to dfs
    p_matrix = pd.DataFrame(p_matrix, index=expression_matrix.index, columns=expression_matrix.columns)
    fc_matrix = pd.DataFrame(fc_matrix, index=expression_matrix.index, columns=expression_matrix.columns)
    pc_matrix = pd.DataFrame(pc_matrix, index=expression_matrix.index, columns=expression_matrix.columns)
    return fc_matrix, p_matrix, pc_matrix

# get markers
fc_matrix, p_matrix, pc_matrix = identify_marker_genes(expression_matrix, min_expression=None)
pbonf_matrix = p_matrix * p_matrix.shape[0] * p_matrix.shape[1]

# set order
sets = [f"{s}-{r}" for s in stages.values() for r in regions.values()]

# collection with only bonferroni significant genes and fold change > 1
print("Only Bonferroni significant genes")
collection = {
    k: p_matrix[k][(pbonf_matrix[k] < 0.05) & (fc_matrix[k] > 1)].index.tolist()
    for k in sets
}
print(f"{len(collection)} sets, between {min(len(v) for v in collection.values())} and {max(len(v) for v in collection.values())} genes.")
write_json(collection, nispace_source_data_path / "reference" / "mrna" / f"collection-BrainSpan.collect")

# collection with all genes that show positive fold change and fold values
print("All genes and fold values")
df = {
    k: fc_matrix[k][fc_matrix[k] > 0].to_frame(name="weight")
    for k in sets
}
df = pd.concat(df, axis=0, names=["set", "map"]).astype(np.float16)
df.to_csv(nispace_source_data_path / "reference" / "mrna" / f"collection-BrainSpanWeights.collect", index=True)


# %% magicc collections ----------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# Copy from mRNA
for fp_src in (nispace_source_data_path / "reference" / "mrna").glob("*.c*"):
    fp_dst = nispace_source_data_path / "reference" / "magicc" / fp_src.name
    shutil.copy(fp_src, fp_dst)

# Replace collection-All.collect
fp = sorted((nispace_source_data_path / "reference" / "magicc" / "tab").glob("dset-magicc_parc-*.csv.gz"))[0]
all_genes = pd.read_csv(fp, index_col=0).index.unique()
all_genes = pd.Series(sorted(all_genes), name="map")
all_genes.to_csv(nispace_source_data_path / "reference" / "magicc" / "collection-All.collect", index=False)


# %% RSN collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

rsn_files = fetch_reference("rsn", space="MNI152")
rsn_files = [f.name.split("_space")[0] for f in rsn_files]

# All
pd.Series(rsn_files, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "rsn" / "collection-All.collect", index=None)


# %% GRF collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# all grf maps
grf_tab_files = sorted( (nispace_source_data_path / "reference" / "grf" / "tab").glob("*.csv.gz") )
all_grf = pd.Series(pd.read_csv(grf_tab_files[0], index_col=0).index, name="map")
all_grf.to_csv(nispace_source_data_path / "reference" / "grf" / "collection-All.collect", index=False)

# only no autocorrelation
all_grf.loc[all_grf.str.contains("alpha-0.0")] \
    .to_csv(nispace_source_data_path / "reference" / "grf" / "collection-Alpha0.collect", index=False)

# collection split by alpha
collection = {}
for alpha in {float(idx.split("alpha-")[1].split("_")[0]) for idx in all_grf}:
    collection[f"alpha-{alpha}"] = [idx for idx in all_grf if idx.startswith(f"alpha-{alpha:.01f}")]
write_json(collection, nispace_source_data_path / "reference" / "grf" / "collection-ByAlpha.collect")



# %% Neurosynth collections ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# All
(
    pd.read_csv(
        list((nispace_source_data_path / "reference" / "neurosynth" / "tab").glob("*.csv.gz"))[0], 
        index_col=0 
    )
    .index.to_series()
    .to_csv(nispace_source_data_path / "reference" / "neurosynth" / "collection-All.collect", index=False)
)

# Manually defined collection

cognitive_functions = {
    "Perception": [
        "perception",
        "visual",
        "visuospatial",
        "auditory",
        "olfactory",
        "tactile",
        "pain",
        "sensory",
        "multisensory",
    ],
    "Attention": [
        "attention",
        "selective attention",
        "spatial attention",
        "sustained attention",
        "visual attention",
        "attentional control",
        "distraction",
        "fixation",
        "salience",
        "shifting",
    ],
    "Memory": [
        "semantic memory",
        "autobiographical memory",
        "episodic memory",
        "working memory",
        "short term",
        "long term",
        "memory encoding",
        "memory retrieval",
        "recall",
        "repetition",
    ],
    "Language": [
        "language",
        "speech",
        "speech perception",
        "speech production",
        "articulation",
        "phonological",
        "listening",
        "reading",
        "comprehension",
        "semantic",
        "syntactic",
    ],
    "Executive Control": [
        "executive control",
        "executive function",
        "cognitive control",
        "cognitive flexibility",
        "flexibility",
        "set shifting",
        "task switching",
        "inhibition",
        "response inhibition",
        "response selection",
        "planning",
        "problem solving",
        "decision making",
        "reasoning",
        "monitoring",
        "arithmetic",
    ],
    "Reward and Learning": [
        "reward",
        "learning",
        "reward anticipation",
        "adaptation",
        "delay",
        "effort",
        "feedback",
        "goal directed",
        "motivation",
        "prediction",
        "prediction error",
        "conflict",
        "reinforcement learning",
        "punishment",
        "risk",
    ],
    "Emotion & Affect": [
        "emotional",
        "affect",
        "empathy",
        "emotion regulation",
        "valence",
        "mood",
        "positive affect",
        "happy",
        "joy",
        "love",
        "satisfaction",
        "negative affect",
        "anxiety",
        "fear",
        "loss",
        "threat",
    ],
    "Social Cognition": [
        "social cognition",
        "social interaction",
        "theory mind",
        "mentalizing",
        "perspective taking",
        "face recognition",
        "imitation",
        "self referential",
    ],
    "Motor Function": [
        "motor",
        "motor control",
        "movement",
        "sensorimotor",
        "somatosensory",
        "action observation",
        "imagery",
        "skill",
        "coordination",
        "eye movements",
        "gaze",
        "grasping",
        "hand movements",
    ],
    "Arousal & State": [
        "arousal",
        "autonomic",
        "eating",
        "interoceptive",
        "sleep",
        "stress",
    ]
}
write_json(cognitive_functions, nispace_source_data_path / "reference" / "neurosynth" / "collection-CognitiveFunctions.collect")


# %% cortexfeatures collections --------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# All
(
    pd.read_csv(
        list((nispace_source_data_path / "reference" / "cortexfeatures" / "tab").glob("*.csv.gz"))[0], 
        index_col=0 
    )
    .index.to_series()
    .to_csv(nispace_source_data_path / "reference" / "cortexfeatures" / "collection-All.collect", index=False)
)

# MEG
collection = [
    "feature-megpoweralpha_pub-shafiei2022",
    "feature-megpowerbeta_pub-shafiei2022",
    "feature-megpowerdelta_pub-shafiei2022",
    "feature-megpowergamma1_pub-shafiei2022",
    "feature-megpowergamma2_pub-shafiei2022",
    "feature-megpowertheta_pub-shafiei2022",
    "feature-megtimescale_pub-shafiei2022",
]
pd.Series(collection, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "cortexfeatures" / "collection-MEG.collect", index=False)

# Metabolism
collection = [
    "feature-cbf_pub-vaishnavi2010",
    "feature-cbv_pub-vaishnavi2010",
    "feature-cmro2_pub-vaishnavi2010",
    "feature-cmrglc_pub-vaishnavi2010",
    "feature-glycindex_pub-vaishnavi2010",
]
pd.Series(collection, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "cortexfeatures" / "collection-Metabolism.collect", index=False)

# CortexTopology
collection = [
    "feature-thickness_pub-hcps1200",
    "feature-t1t2_pub-hcps1200",
    "feature-saaxis_pub-sydnor2021",
    "feature-geneexpr-abagen",
    "feature-develexpansion_pub-hill2010",
    "feature-evolexpansion_pub-hill2010",
    "feature-evolexpansion_pub-xu2020",
    "feature-specieshomology_pub-xu2020",
]
pd.Series(collection, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "cortexfeatures" / "collection-CortexOrganisation.collect", index=False)


# %% bigbrain collections --------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# All
(
    pd.read_csv(
        list((nispace_source_data_path / "reference" / "bigbrain" / "tab").glob("*.csv.gz"))[0], 
        index_col=0 
    )
    .index.to_series()
    .to_csv(nispace_source_data_path / "reference" / "bigbrain" / "collection-All.collect", index=False)
)

# Cortical layers
collection = [
    "feature-layer1_pub-wagstyl2020",
    "feature-layer2_pub-wagstyl2020",
    "feature-layer3_pub-wagstyl2020",
    "feature-layer4_pub-wagstyl2020",
    "feature-layer5_pub-wagstyl2020",
    "feature-layer6_pub-wagstyl2020",
]
pd.Series(collection, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "bigbrain" / "collection-CorticalLayers.collect", index=False)

# DifferentiationGradients
collection = [
    "feature-histogradient1_pub-paquola2021",
    "feature-histogradient2_pub-paquola2021",
    "feature-microgradient1_pub-paquola2021",
    "feature-microgradient2_pub-paquola2021",
    "feature-funcgradient1_pub-paquola2021",
    "feature-funcgradient2_pub-paquola2021",
    "feature-funcgradient3_pub-paquola2021",
]
pd.Series(collection, name="map") \
    .to_csv(nispace_source_data_path / "reference" / "bigbrain" / "collection-DifferentiationGradients.collect", index=False)

# %% tpm collections -------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# All
with open(nispace_source_data_path / "reference" / "tpm" / "collection-All.collect", "w") as f:
    for fp in sorted((wd / "reference" / "tpm" / "map").glob("*")):
        if fp.is_dir():
            f.write(f"{fp.name}\n")

# %%
