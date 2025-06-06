# %%
import sys
import pathlib
import tempfile

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
        
        datasets.fetch_parcellation(parc_name, space, return_dist_mat=False, overwrite=True)

# %% Templates

for template in [
    "MNI152NLin2009cAsym", "MNI152NLin6Asym", "fsaverage"
]:
    datasets.fetch_template(template, overwrite=True)

# %% Reference data

# imaging files
datasets.fetch_reference("pet", github_config_file=github_config_file, overwrite=True)


# parcellation tables
datasets.fetch_reference("pet", parcellation="MelbourneS1", collection="UniqueTracers",
                         github_config_file=github_config_file, overwrite=True)
