# %%
import sys
import pathlib
import pandas as pd
import tempfile
import os

# add nispace to path
sys.path.append(str(pathlib.Path.home() / "projects" / "nispace"))
from nispace import datasets

github_config_file = pathlib.Path.home() / "projects" / "nispace" / ".github.config"

# %% Parcellations

for parc_name in datasets.parcellation_lib:
    print(parc_name)
    
    for space in datasets.parcellation_lib[parc_name]:
        if space in ["alias", "cortex", "subcortex"]:
            continue
        print(space)
        
        datasets.fetch_parcellation(parc_name, space, return_dist_mat=False)

# %% Templates

for template in [
    "MNI152NLin2009cAsym", "MNI152NLin6Asym", "fsaverage"
]:
    datasets.fetch_template(template)

# %% Reference data

# imaging files
datasets.fetch_reference("pet", github_config_file=github_config_file)
datasets.fetch_reference("cortexfeatures", github_config_file=github_config_file, space="fsaverage")

# parcellation tables
datasets.fetch_reference("pet", parcellation="MelbourneS1", collection="UniqueTracers",
                         github_config_file=github_config_file)

# load the table, change it and save again
NISPACE_DATA_DIR = os.getenv("NISPACE_DATA_DIR")
table_path = pathlib.Path(NISPACE_DATA_DIR) / "reference" / "pet" / "tab" / "dset-pet_parc-TianS1.csv.gz"
table = pd.read_csv(table_path, index_col=0)
val_orig = table.iloc[0,0]
print("original table loc (0,0):", val_orig)
table.iloc[0,0] = 999
val_changed = table.iloc[0,0]
print("changed table loc (0,0):", val_changed)
table.to_csv(table_path)

# fetch again
datasets.fetch_reference("pet", parcellation="MelbourneS1", collection="UniqueTracers",
                         github_config_file=github_config_file, print_references=False)

# check if the table is back to the start
table_path = pathlib.Path(NISPACE_DATA_DIR) / "reference" / "pet" / "tab" / "dset-pet_parc-TianS1.csv.gz"
table = pd.read_csv(table_path, index_col=0)
val_restored = table.iloc[0,0]
print("restored table loc (0,0):", val_restored)
assert val_restored == val_orig, "Table was not re-downloaded?!"

# %%
