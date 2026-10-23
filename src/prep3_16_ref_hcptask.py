# %% Init

from pathlib import Path
import sys
import pandas as pd
import nibabel as nib
from neuromaps import transforms

wd = Path(__file__).parent.parent
print(f"Working dir: {wd}")

sys.path.insert(0, str(Path(__file__).parent))
from utils import split_cifti, parcellate_mapref

DATASET = "hcptask"
ref_dir = wd / "reference" / DATASET

# Source: HCP S1200 group-average task-fMRI Cohen's d maps (997 subjects, MSMAll, 2mm FWHM
# grayordinate smoothing; no unsmoothed version available) from "Human HCP 1200 Group Average +
# Individuals; Structural + fMRI Atlas" on BALSA (reference pkXDZ; HCP_S1200_GroupAvg_BALSA.scene,
# ID: D4w0w; login required), stored in _archive/hcps1200.
# Full 91k-grayordinate CIFTI: cortex -> fsLR 32k (+ fsaverage 41k), subcortex/cerebellum/brainstem
# -> MNI152NLin6Asym 2mm volume (see split_cifti).
cifti_path = wd / "_archive" / "hcps1200" / "HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMAll.dscalar.nii"

# The volume part (subcortex, cerebellum, brainstem) is NOT distributed as images: an MNI map
# without cortex would silently break whole-brain/custom volumetric parcellations in the toolbox.
# It is only used locally for the subcortical tabs. Once maps carry space coverage metadata (or a
# whole-brain volume exists, e.g. via surface-to-volume projection), ship it in MNI152NLin6Asym and
# MNI152NLin2009cAsym like any volumetric dataset.
vol_dir = wd / "_archive" / "hcps1200" / "hcptask_volume"


# %% Map selection and metadata
# The CIFTI holds 86 maps, 39 of which are exact sign flips of others (neg_X = -X, B-A = -(A-B)).
# We keep conditions (vs. baseline) and one direction per contrast (the conventional one, Barch 2013).

DROP = [
    "WM_0BK-2BK", "WM_AVG-BODY", "WM_AVG-FACE", "WM_AVG-PLACE", "WM_AVG-TOOL",
    "GAMBLING_PUNISH-REWARD",
    "MOTOR_AVG-CUE", "MOTOR_AVG-LF", "MOTOR_AVG-LH", "MOTOR_AVG-RF", "MOTOR_AVG-RH", "MOTOR_AVG-T",
    "LANGUAGE_MATH-STORY", "SOCIAL_RANDOM-TOM", "RELATIONAL_MATCH-REL", "EMOTION_SHAPES-FACES",
]

TASK_LABELS = {
    "WM": "Working memory", "GAMBLING": "Gambling", "MOTOR": "Motor", "LANGUAGE": "Language",
    "SOCIAL": "Social cognition", "RELATIONAL": "Relational processing", "EMOTION": "Emotion processing",
}
COND_LABELS = {
    "2BK": "2-back", "0BK": "0-back", "BODY": "body", "FACE": "face", "PLACE": "place", "TOOL": "tool",
    "AVG": "average", "REWARD": "reward", "PUNISH": "punishment", "CUE": "cue", "LF": "left foot",
    "LH": "left hand", "RF": "right foot", "RH": "right hand", "T": "tongue", "STORY": "story",
    "MATH": "math", "TOM": "theory of mind", "RANDOM": "random motion", "REL": "relational",
    "MATCH": "matching", "FACES": "faces", "SHAPES": "shapes",
}

hcp_names = list(nib.load(cifti_path).header.get_axis(0).name)

rows = []
for hcp_name in hcp_names:
    task, contrast = hcp_name.removeprefix("tfMRI_").split("_", 1)
    if contrast.startswith("neg_") or f"{task}_{contrast}" in DROP:
        continue
    if "-" in contrast:
        a, b = contrast.split("-")
        con_type = "contrast"
        con_value = f"{a}vs{b}"
        label = f"{COND_LABELS[a]} vs. {COND_LABELS[b]}"
    else:
        con_type = "condition"
        con_value = contrast.replace("_", "")
        label = " ".join(COND_LABELS[c] for c in contrast.split("_")) + " vs. baseline"
    if task == "MOTOR" and contrast == "AVG":
        label = "average movement vs. baseline"
    rows.append({
        "map": f"task-{task}_contrast-{con_value}_pub-hcps1200",
        "task": TASK_LABELS[task],
        "label": label,
        "type": con_type,
        "hcp_name": hcp_name,
        "doi": "10.1016/j.neuroimage.2013.05.033",
    })
map_info = pd.DataFrame(rows)
assert len(map_info) == 47, len(map_info)
assert map_info["map"].is_unique
map_info.to_csv(ref_dir / "map_info.csv", index=False)
print(map_info[["map", "task", "label", "type"]].to_string())


# %% Convert to NiSpace-format maps

maps = split_cifti(cifti_path, maps=map_info["hcp_name"])

for m, hcp_name in zip(map_info["map"], map_info["hcp_name"]):
    print("Processing map:", m)
    save_dir = ref_dir / "map" / m
    save_dir.mkdir(parents=True, exist_ok=True)

    # cortex: fsLR 32k and fsaverage 41k
    map_fslr = maps[hcp_name]["surface"]
    map_fsa = transforms.fslr_to_fsaverage(data=map_fslr, target_density="41k", method="linear")
    for i_h, h in enumerate(["L", "R"]):
        map_fslr[i_h].to_filename(save_dir / f"{m}_space-fsLR_desc-proc_hemi-{h}.surf.gii.gz")
        map_fsa[i_h].to_filename(save_dir / f"{m}_space-fsaverage_desc-proc_hemi-{h}.surf.gii.gz")

    # subcortex, cerebellum, brainstem: MNI152NLin6Asym 2mm (CIFTI volume grid), 0 outside; local only
    (vol_dir / m).mkdir(parents=True, exist_ok=True)
    maps[hcp_name]["volume"].to_filename(vol_dir / m / f"{m}_space-MNI152NLin6Asym_desc-proc.nii.gz")


# %% Collections

def _maps(**kwargs):
    df = map_info
    for k, v in kwargs.items():
        df = df[df[k].isin(v if isinstance(v, list) else [v])]
    return df["map"].tolist()

def _ids(*contrasts):
    """task_contrast shorthands -> map IDs, in the given order"""
    out = []
    for c in contrasts:
        task, con = c.split("_", 1)
        out.append(f"task-{task}_contrast-{con}_pub-hcps1200")
    assert all(i in map_info["map"].values for i in out), [i for i in out if i not in map_info["map"].values]
    return out

collections = {
    "All": map_info["map"].tolist(),
    # the "focused" contrast of each task (Barch et al., 2013: "more focused on isolating specific
    # cognitive processes (e.g., 2-back versus 0-back, reward versus punishment)"); motor: each
    # movement "versus all other movement types" (= HCP's movement vs. AVG)
    "Canonical": _ids(
        "WM_2BKvs0BK", "GAMBLING_REWARDvsPUNISH",
        "MOTOR_LFvsAVG", "MOTOR_LHvsAVG", "MOTOR_RFvsAVG", "MOTOR_RHvsAVG", "MOTOR_TvsAVG",
        "LANGUAGE_STORYvsMATH", "SOCIAL_TOMvsRANDOM", "RELATIONAL_RELvsMATCH", "EMOTION_FACESvsSHAPES",
    ),
    "Conditions": _maps(type="condition"),
    "Contrasts": _maps(type="contrast"),
}
for name, ids in collections.items():
    pd.Series(ids, name="map").to_csv(ref_dir / f"collection-{name}.collect", index=False)
    print(f"collection-{name}: {len(ids)} maps")

# set collection: one set per task (same grouping as fetch_reference(..., set_by="task"))
SET_NAMES = {
    "WM": "WorkingMemory", "GAMBLING": "Gambling", "MOTOR": "Motor", "LANGUAGE": "Language",
    "SOCIAL": "Social", "RELATIONAL": "Relational", "EMOTION": "Emotion",
}
sets = pd.DataFrame({
    "set": [SET_NAMES[m.split("_")[0].removeprefix("task-")] for m in map_info["map"]],
    "x": map_info["map"],
})
sets.to_csv(ref_dir / "collection-All-Sets.collect", index=False)
print("collection-All-Sets:", sets["set"].value_counts().to_dict())


# %% Parcellate ------------------------------------------------------------------------------------
# cortex parcellations on the fsLR surface part, subcortex parcellations on the (local)
# MNI152NLin6Asym volume part of the CIFTI

parcellate_mapref(wd, DATASET, spaces={
    "cortex": ["fsLR"],
    "subcortex": ["MNI152NLin6Asym"],
}, local_spaces={"MNI152NLin6Asym": vol_dir})

# %%
