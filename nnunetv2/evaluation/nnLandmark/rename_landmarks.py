import json
from pathlib import Path

input_file = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/Dataset742_FeTA_2_4/nnLandmark_fabi/prediction_ResEncL/prediction_all_landmark_voxel.json"
output_file = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/Dataset742_FeTA_2_4/nnLandmark_fabi/prediction_ResEncL/renamed_landmarks.json"
name_to_label_file = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/name_to_label.json"

with open(name_to_label_file, "r") as f:
    name_to_label = json.load(f)

# invert: label (int) -> name (str)
label_to_name = {}
for name, lbl in name_to_label.items():
    try:
        label_to_name[int(lbl)] = name
    except Exception:
        continue

with open(input_file, "r") as f:
    data = json.load(f)

renamed_data = {}
for subject_id, landmarks in data.items():
    renamed_landmarks = {}

    # map numeric keys (label values) to names using label_to_name
    for key, value in landmarks.items():
        if isinstance(key, str) and key.isdigit():
            kint = int(key)
            new_name = label_to_name.get(kint)
            if new_name:
                renamed_landmarks[new_name] = value
            else:
                # keep numeric key if no mapping found
                renamed_landmarks[key] = value
        else:
            # non-numeric keys copied unchanged
            renamed_landmarks[key] = value

    renamed_data[subject_id] = renamed_landmarks

with open(output_file, "w") as f:
    json.dump(renamed_data, f, indent=2)

print(f"Renamed landmarks saved to {output_file}")
