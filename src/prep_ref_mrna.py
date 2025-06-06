# %% Init

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import zipfile
from itertools import combinations
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# wd
wd = Path.cwd().parent
print(f"Working dir: {wd}")

# Abagen (cloned github version)
#sys.path.append(str(Path.home() / "projects" / "abagen"))
from abagen import get_expression_data, images, keep_stable_genes

# Nispace
sys.path.append(str(Path.home() / "projects" / "nispace"))
from nispace.datasets import fetch_template
from nispace.io import load_labels

# nispace data path 
nispace_source_data_path = wd

# All parcellations
PARCS = sorted(
    [p.name for p in (nispace_source_data_path / "parcellation").glob("*") if p.is_dir()]
)
print("PARCS:", PARCS)

PARCS_SC = ["TianS1", "TianS2", "TianS3", "Aseg"]

# %% mRNA tabulated data ---------------------------------------------------------------------------
corr_threshold = 0.1

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

    # space
    if parc == "Glasser":
        space = "fsaverage"
    else:
        space = "MNI152NLin6Asym"

    # load parc
    if "mni" in space.lower():
        parc_path = nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}.label.nii.gz"
        labels_path = parc_path.parent / f"parc-{parc}_space-{space}.label.txt"
    else:
        parc_path = (
            nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}_hemi-L.label.gii.gz",
            nispace_source_data_path / "parcellation" / parc / space / f"parc-{parc}_space-{space}_hemi-R.label.gii.gz"
        )
        labels_path = (
            parc_path[0].parent / f"parc-{parc}_space-{space}_hemi-L.label.txt",
            parc_path[1].parent / f"parc-{parc}_space-{space}_hemi-R.label.txt"
        )
    
    if isinstance(parc_path, tuple):
        tpl_path = fetch_template("fsaverage", desc="pial")
        parc_path = images.check_atlas(
            atlas=(str(parc_path[0]), str(parc_path[1])),
            space="fsaverage",
            geometry=(str(tpl_path[0]), str(tpl_path[1]))
        )
    else:
        parc_path = str(parc_path)
    parc_labels = load_labels(labels_path)

    # parc info
    parc_info = pd.DataFrame({
        "id": np.arange(1, len(parc_labels) + 1),
        "label": parc_labels,
        "hemisphere": [l.split("hemi-")[1].split("_")[0] for l in parc_labels],
        "structure": ["cortex" if parc not in PARCS_SC else "subcortex/brainstem"] * len(parc_labels)
    })
        
    # get data for each donor
    mRNA_dict = get_expression_data(
        atlas=parc_path,
        atlas_info=parc_info,
        lr_mirror="bidirectional",
        n_proc=1,
        verbose=False, # 1
        return_donors=True 
    )
    
    # keep stable genes
    mRNA_list_stable, stability = keep_stable_genes(
        expression=list(mRNA_dict.values()),
        threshold=0.1,
        percentile=False,
        rank=True,
        return_stability=True
    )
    stability = pd.Series(stability, index=mRNA_dict[list(mRNA_dict.keys())[0]].columns, dtype=np.float32)
    stability.name = "reproducibility"
    stability.index.name = "map"
    print(stability.head())
    
    # get genes to extract
    genes_to_extract = mRNA_list_stable[0].columns.to_list()
    print(f"Retaining {len(genes_to_extract)} genes for {parc}")

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
    stability.to_csv(
        nispace_source_data_path / "reference" / "mrna" / "tab" /  f"dset-mrna_parc-{parc}_reproducibility.csv.gz"
    )

#%% Run

# parcellations
print(f"{len(PARCS)} parcellations: {PARCS}")

# Run in parallel
Parallel(n_jobs=1)(
    delayed(par_fun)(parc) 
    for parc in tqdm(PARCS)
)

# %%
