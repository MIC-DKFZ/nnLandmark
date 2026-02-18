import json, os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# ------------------------------------------------------------------ paths
root = Path("/path/to/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset737_DMGLD_LFC")
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
    dataset_name="Dataset737_DMGLD_LFC",        
)
