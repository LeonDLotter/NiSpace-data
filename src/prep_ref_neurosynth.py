# %% Init

import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed

import nibabel as nib
from nilearn.image import math_img

from nimare.extract import download_abstracts, fetch_neurosynth, download_cognitive_atlas
from nimare.io import convert_neurosynth_to_dataset
from nimare.dataset import Dataset
from nimare.meta.cbma.mkda import MKDAChi2
# NOTE: biopython is required for downloading abstracts but doesnt have to be imported

# working directory
wd = Path.cwd().parent
print(f"Working dir: {wd}")

# DIRECTORY IN WHICH NEUROSYNTH DATA WILL BE STORED
neurosynth_data_path = Path("/Volumes/data_m2_2tb/data/neurosynth")
if not neurosynth_data_path.exists():
    raise FileNotFoundError(f"Neurosynth data path does not exist: {neurosynth_data_path}")

# EMAIL FOR DOWNLOADING ABSTRACTS
email = "l.lotter@fz-juelich.de"

# add nispace to path
sys.path.append(str(Path.home() / "projects" / "nispace"))

# import NiSpace functions
from nispace.datasets import parcellation_lib, fetch_parcellation
from nispace.parcellate import Parcellater
from nispace.io import write_json
from nispace.utils.utils_datasets import download

# nispace data path 
nispace_source_data_path = wd

# %% Download neurosynth data and convert to nimare dataset

for vocab in ["terms", "LDA50", "LDA100", "LDA200"]:
    print(f"Downloading neurosynth data with vocab: {vocab}")
    
    # download neurosynth data
    neurosynth_files = fetch_neurosynth(
        data_dir=neurosynth_data_path,
        version="7",
        overwrite=False,
        source="abstract",
        vocab=vocab,
    )
    # files are saved to a new folder within "neurosynth_data_path" named "neurosynth".
    print(neurosynth_files)
    neurosynth_db = neurosynth_files[0]

    # convert only the term vocab to nimare dataset and download abstracts
    if vocab == "terms":
        print(f"Converting neurosynth data to nimare dataset and downloading abstracts for vocab: {vocab}")
        fp = neurosynth_data_path / "neurosynth" / f"neurosynth_dataset_{vocab}.pkl.gz"
        if not fp.exists():
            neurosynth_dset = convert_neurosynth_to_dataset(
                coordinates_file=neurosynth_db["coordinates"],
                metadata_file=neurosynth_db["metadata"],
                annotations_files=neurosynth_db["features"],
            )
            neurosynth_dset = download_abstracts(neurosynth_dset, email)
            neurosynth_dset.save(fp)
        else:
            neurosynth_dset = Dataset.load(fp)
        print(neurosynth_dset)

# get all terms in neurosynth dataset
all_terms_in_dset = sorted(list(set(neurosynth_dset.get_labels())))
print(f"Number of unique terms in neurosynth dataset: {len(all_terms_in_dset)}")

# load the keep list from neurosynth-web
fp = neurosynth_data_path / "neurosynth" / "analysis_filter_list.txt"
if not fp.exists():
    fp = download(
        "https://raw.githubusercontent.com/neurosynth/neurosynth-web/e8776a0/data/assets/analysis_filter_list.txt", 
        path=fp
    )
terms_to_keep = pd.read_csv(fp, sep="\t")
terms_to_keep = terms_to_keep[terms_to_keep["keep"] == 1]["term"].to_list()
print(f"Number of terms in keep list: {len(terms_to_keep)}")

# %% Download cognitive atlas

# cogatlas = download_cognitive_atlas(
#     data_dir=neurosynth_data_path,
#     overwrite=False,
# )

    
# %% Run meta-analysis on term maps that are in any LDA dataset

# output dir
out_dir = neurosynth_data_path / "term_maps"

# function for parallelization
# the meta-analysis corresponds to the nimare.decode.continuous.CorrelationDecoder
def run_meta_analysis(term, fq_threshold=0.001, neurosynth_dset=neurosynth_dset, save_type="image"):
    
    # term name
    term_save = term.replace(" ", "+")
    
    # save path
    if save_type == "image":
        out_path = out_dir / f"term-{term_save}_stat-z_desc-association.nii.gz"
    elif save_type == "array":
        out_path = out_dir / f"term-{term_save}.npz"
    else:
        raise ValueError(f"Invalid save type: {save_type}")
    
    # skip if file exists
    if out_path.exists():
        return
    
    # get study ids with term
    ids = neurosynth_dset.get_studies_by_label(
        labels=[f"terms_abstract_tfidf__{term}"],
        label_threshold=fq_threshold,
    )
    
    # studies without term
    ids_without = sorted(list(set(neurosynth_dset.ids) - set(ids)))
    
    # subset dataset
    dset_withids = neurosynth_dset.slice(ids)
    dset_withoutids = neurosynth_dset.slice(ids_without)
    
    # run meta-analysis
    meta = MKDAChi2()
    results = meta.fit(dset_withids, dset_withoutids)
    
    # get maps    
    map_association = results.get_map(
        "z_desc-association",
        return_type=save_type,
    )
    map_uniformity = results.get_map(
        "z_desc-uniformity",
        return_type=save_type,
    )
    
    # save maps
    if save_type == "image":
        map_association = math_img("img.astype(np.float32)", img=map_association)
        map_association = map_association.to_filename(out_path)
        map_uniformity = math_img("img.astype(np.float32)", img=map_uniformity)
        map_uniformity = map_uniformity.to_filename(out_dir / out_path.name.replace("desc-association", "desc-uniformity"))
    else:
        np.savez(
            out_path, 
            association=map_association.astype(np.float32),
            uniformity=map_uniformity.astype(np.float32),
        )
    
# run meta-analysis in parallel
_ = Parallel(n_jobs=-1)(
    delayed(run_meta_analysis)(term)
    for term in tqdm(terms_to_keep)
)


# %% parcellate the data

# fetch all "z_desc-association" maps
# that again corresponds to the nimare.decode.continuous.CorrelationDecoder
terms_to_keep = pd.read_table(neurosynth_data_path / "neurosynth" / "analysis_filter_list.txt") \
    .query("keep == 1")["term"].to_list()
map_paths = [
    neurosynth_data_path / "term_maps" / f"term-{term.replace(' ', '+')}_stat-z_desc-association.nii.gz"
    for term in terms_to_keep
]
if not all([fp.exists() for fp in map_paths]):
    raise FileNotFoundError(f"Some maps do not exist: {map_paths}")

# iterate nispace parcellations
for parc in parcellation_lib.keys():
    if "alias" in parcellation_lib[parc]:
        continue
    print(parc)
    
    # space
    parc_space = "MNI152NLin6Asym" if "MNI152NLin6Asym" in parcellation_lib[parc] else "fsLR"
    
    # fetch parcellation
    # the template from nimare ("mni152_2mm") corresponds to the MNI152NLin6Asym template in NiSpace
    parc_img, parc_labels = fetch_parcellation(
        parc,
        space=parc_space,
        return_loaded=True,
    )
    
    # prep parcellator
    parcellater = Parcellater(
        parcellation=parc_img,
        space=parc_space,
        resampling_target="data" if "mni" in parc_space.lower() else "parcellation",
    ).fit()
    
    # parcellate function for one map
    def par_fun(map_path):
        data = parcellater.transform(
            data=map_path,
            space="MNI152NLin6Asym",
            ignore_background_data=True,
            background_value=0,
            min_num_valid_datapoints=None,
            min_fraction_valid_datapoints=None,
        )
        return data.astype(np.float32)
    
    # run in parallel
    data = Parallel(n_jobs=-1)(
        delayed(par_fun)(map_path)
        for map_path in tqdm(map_paths)
    )
    
    # data is a list of arrays: to dataframe
    data = pd.DataFrame(data, columns=parc_labels, index=pd.Index(terms_to_keep, name="map"), dtype=np.float32)
    print("Data shape:", data.shape)
    
    # save
    data.to_csv(nispace_source_data_path / "reference" / "neurosynth" / "tab" / f"dset-neurosynth_parc-{parc}.csv.gz")