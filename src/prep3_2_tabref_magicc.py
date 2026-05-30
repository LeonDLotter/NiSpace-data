# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# wd
wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

# Nispace
from nispace.utils.utils import vol_to_vect_arr

# local utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_parc_lists, load_parc, load_parc_labels

# nispace data path
nispace_source_data_path = wd

# DIRECTORY TO MAGICC DATA
magicc_data_path = Path("/Volumes/data_m2_2tb/data/magicc_expression_data")

# cortical parcellations only — magicc is surface expression (fsLR); subcortical parcs have no fsLR
PARCS, PARCS_CX, PARCS_SC = load_parc_lists(wd)
print("PARCS_CX:", PARCS_CX)

# overwrite
overwrite = True

# %% mRNA tabulated data ---------------------------------------------------------------------------

# load info file
df_tableS2 = pd.read_csv(magicc_data_path / "SuppTable2.csv", index_col=0)
print(df_tableS2.shape)

# save reproducibility
df_reproducibility = (
    df_tableS2
    .loc[:, ["gene.symbol", "Estimated reproducibility"]]
    .rename(columns={"gene.symbol": "map", "Estimated reproducibility": "reproducibility"})
    .set_index("map")
)
df_reproducibility.to_csv(nispace_source_data_path / "reference" / "magicc" / "tab" / 
                          "dset-magicc_reproducibility.csv.gz")

# load expression data in fslr space
expression_fslr = np.load(magicc_data_path / "ahba_vertex.npy")
print(expression_fslr.shape)

# filter by reproducibility
repr_threshold = 0.5
expression_fslr = expression_fslr[df_reproducibility.reproducibility >= repr_threshold]
df_tableS2 = df_tableS2[df_tableS2["Estimated reproducibility"] >= repr_threshold]
print(df_tableS2.shape)
print(expression_fslr.shape)

# checks
print("Values range:", np.nanmin(expression_fslr), np.nanmax(expression_fslr))
print("Nan vertices:", np.isnan(expression_fslr).sum())
print("Exactly 0 vertices:", (expression_fslr == 0).sum())

# extract data
for parc_name in PARCS_CX:
    print(parc_name)

    # skip if already done
    out_path = nispace_source_data_path / "reference" / "magicc" / "tab" / f"dset-magicc_parc-{parc_name}.csv.gz"
    if out_path.exists() and not overwrite:
        print(f"  Already exists — skipping")
        continue

    parc   = load_parc(wd, parc_name, "fsLR")
    labels = load_parc_labels(wd, parc_name, "fsLR").tolist()
  
    # get parcellation data
    parc_dat_lh = parc[0].agg_data()
    idc_lh = np.trim_zeros(np.unique(parc_dat_lh))
    parc_dat_rh = parc[1].agg_data()
    idc_rh = np.trim_zeros(np.unique(parc_dat_rh))
    
    # extract expression data
    def par_fun(expr_vector):
        expr_lh = vol_to_vect_arr(expr_vector.astype(np.float32), parc_dat_lh, idc_lh, [np.nan])
        expr_rh = vol_to_vect_arr(expr_vector.astype(np.float32), parc_dat_rh, idc_rh, [np.nan])
        return np.concatenate([expr_lh, expr_rh])
    list_parc_expr = Parallel(n_jobs=-1)(
        delayed(par_fun)(expr_vector) 
        for expr_vector in tqdm(expression_fslr)
    )
    
    # to dataframe
    df_parc_expr = pd.DataFrame(
        np.stack(list_parc_expr),
        index=pd.Index(df_tableS2["gene.symbol"], name="map"),
        columns=labels,
    )
    print("Parcellated expression shape:", df_parc_expr.shape)
    print(df_parc_expr.head())
    
    # check for nans
    print("NaN values:", df_parc_expr.isna().sum().sum())
    print("Inf values:", np.isinf(df_parc_expr.values).sum())
    
    # datatype
    df_parc_expr = df_parc_expr.astype(np.float16)
        
    # save
    df_parc_expr.to_csv(out_path)


# %% Collections

import shutil

ref_dir = nispace_source_data_path / "reference" / "magicc"
mrna_ref_dir = nispace_source_data_path / "reference" / "mrna"

# Copy gene-set collections from mRNA
for fp_src in mrna_ref_dir.glob("*.c*"):
    shutil.copy(fp_src, ref_dir / fp_src.name)

# Replace collection-All.collect (magicc genes filtered by reproducibility threshold)
fp = sorted((ref_dir / "tab").glob("dset-magicc_parc-*.csv.gz"))[0]
all_genes = pd.Series(sorted(pd.read_csv(fp, index_col=0).index.unique()), name="map")
all_genes.to_csv(ref_dir / "collection-All.collect", index=False)

# %%
