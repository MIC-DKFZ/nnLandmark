import json, os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# ------------------------------------------------------------------ paths
root = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4")
imagesTr = root / "imagesTr"
imagesTs = root / "imagesTs"

# ------------------------------------------------------------------ load name→label and invert
name2label_path = root / "name_to_label.json"          # produced earlier
name2label      = json.loads(name2label_path.read_text())    # landmark_<n> → int
labels = {"background": 0, **name2label} 

print(f"{len(labels)-1} foreground labels loaded")
# e.g. {0:'background', 1:'landmark_1', 2:'landmark_2', …}

# ------------------------------------------------------------------ write dataset.json
generate_dataset_json(
    output_folder=root,
    channel_names=({0: 'MRI'}),           
    labels=labels,  
    num_training_cases=len(os.listdir(imagesTr)),
    file_ending=".nii.gz", 
    dataset_name="Dataset742_FeTA_2_4",                    # human-readable or task ID
    license="https://www.synapse.org/Synapse:syn25649159/wiki/610007 Access to the data requires that you are a registered Synapse user (and having accepted the Synapse Terms and Conditions of Use) and agree to the following terms and conditions: By joining the FeTA dataset users team, you acknowledge that the owner of the Fetal Tissue Annotation Challenge Dataset available on Synapse is the University Children’s Hospital Zurich, , and that the owner of the FeTA Biometry Dataset is the Lausanne Children’s Hospital Zurich. Fetal Tissue Annotation, Segmentation, and Biometry Dataset is used only for research and education only. Any other kind of use you will lead to recall of all datasets, stop of collaboration and legal consequences.",
    converted_by="alex",
    dataset_description="Dataset version feta_2.4; https://www.synapse.org/Synapse:syn25649159/files/; Part of the data of FeTA Miccai Challenge",
    citation="Please cite the following when using the FeTA segmentation dataset in your research: Payette, K., de Dumast, P., Kebiri, H. et al. An automatic multi-tissue human fetal brain segmentation benchmark using the Fetal Tissue Annotation Dataset. Sci Data 8, 167 (2021). https://doi.org/10.1038/s41597-021-00946-3 Please cite the following when using the FeTA Biometry dataset in your research: Sanchez, T. et al. Fetal Tissue Annotation Challenge (FeTA) Biometry - MICCAI 2024. Zenodo (2024). https://doi.org/10.5281/zenodo.11192452")
