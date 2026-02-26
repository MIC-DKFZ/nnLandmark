import json, os
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join
from nnlandmark.dataset_conversion.generate_dataset_json import generate_dataset_json

# ------------------------------------------------------------------ paths
root = Path("/path/to/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset740_PDDCA")
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
    file_ending=".nrrd", 
    dataset_name="Dataset740_PDDCA",                    # human-readable or task ID
    license="Not found, but cite; they say \"public domain databaset\" and \"We give this data to the community in the hopes that it will be helpful.\"",
    converted_by="alex",
    dataset_description="Dataset version 1.4.1; https://www.imagenglab.com/newsite/pddca/; The data here provided have been used for the “Head and Neck Auto Segmentation MICCAI Challenge (2015)”.",
    citation="Raudaschl, P. F., Zaffino, P., Sharp, G. C., Spadea, M. F., Chen, A., Dawant, B. M., … & Jung, F. (2017). Evaluation of segmentation methods on head and neck CT: Auto‐segmentation challenge 2015. Medical Physics, 44(5), 2020-2036."
)
