# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from neuromaps import images
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# wd
wd = Path.cwd().parent
print(f"Working dir: {wd}")

# Nispace
sys.path.append(str(Path.home() / "projects" / "nispace"))
from nispace.utils.utils import vol_to_vect_arr

# nispace data path 
nispace_source_data_path = wd

# DIRECTORY TO MAGICC DATA
magicc_data_path = Path("/Volumes/data_m2_2tb/data/magicc_expression_data")

# All parcellations
PARCS = sorted(
    [p.name for p in (nispace_source_data_path / "parcellation").glob("*") if p.is_dir()]
)
print("PARCS:", PARCS)

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

# extract data
for parc_name in PARCS:
    print(parc_name)
    
    # skip if no fsLR space available
    if not (nispace_source_data_path / "parcellation" / parc_name / "fsLR").exists():
        print(f"Parcellation {parc_name} not found in fsLR space")
        continue
    
    # load parc
    parc = (
        images.load_gifti(
            nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
            f"parc-{parc_name}_space-fsLR_hemi-L.label.gii.gz"
        ),
        images.load_gifti(
            nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
            f"parc-{parc_name}_space-fsLR_hemi-R.label.gii.gz"
        )
    )
    labels = np.concatenate([
        np.loadtxt(
            nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
            f"parc-{parc_name}_space-fsLR_hemi-L.label.txt", dtype=str
        ),
        np.loadtxt(
            nispace_source_data_path / "parcellation" / parc_name / "fsLR" / 
            f"parc-{parc_name}_space-fsLR_hemi-R.label.txt", dtype=str
        )
    ]).tolist()
  
    # get parcellation data
    parc_dat_lh = parc[0].agg_data()
    idc_lh = np.trim_zeros(np.unique(parc_dat_lh))
    parc_dat_rh = parc[1].agg_data()
    idc_rh = np.trim_zeros(np.unique(parc_dat_rh))
    
    # extract expression data
    def par_fun(expr_vector):
        expr_lh = vol_to_vect_arr(expr_vector.astype(np.float32), parc_dat_lh, idc_lh)
        expr_rh = vol_to_vect_arr(expr_vector.astype(np.float32), parc_dat_rh, idc_rh)
        return np.concatenate([expr_lh, expr_rh])
    list_parc_expr = Parallel(n_jobs=-1)(
        delayed(par_fun)(expr_vector) 
        for expr_vector in tqdm(expression_fslr)
    )
    
    # to dataframe
    df_parc_expr = pd.DataFrame(
        np.stack(list_parc_expr).astype(np.float16),
        index=pd.Index(df_tableS2["gene.symbol"], name="map"),
        columns=labels,
        dtype=np.float16
    )
        
    # save
    df_parc_expr.to_csv(nispace_source_data_path / "reference" / "magicc" / "tab" / 
                        f"dset-magicc_parc-{parc_name}.csv.gz")


# %%
