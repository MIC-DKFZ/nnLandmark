import json

# Input and output file paths
input_file = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/Dataset737_DMGLD_LFC/nnLandmark_fabi/prediction_ResEncM/prediction_all_landmark_voxel.json"
output_file = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/Dataset737_DMGLD_LFC/nnLandmark_fabi/prediction_ResEncM/renamed_landmarks.json"

# Load the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Initialize the mapping logic
renamed_data = {}
for subject_id, landmarks in data.items():
    renamed_landmarks = {}
    
    # Get all numeric landmark keys and sort them numerically
    numeric_keys = []
    for key in landmarks.keys():
        if isinstance(key, str) and key.isdigit():
            try:
                numeric_keys.append(int(key))
            except ValueError:
                pass
    
    numeric_keys.sort()
    
    # Rename using sorted numeric order (guaranteed 1,2,3,4...)
    for idx, pred_key in enumerate(numeric_keys, start=1):
        value = landmarks[str(pred_key)]
        
        # Calculate the new landmark name
        group = (idx + 1) // 2  # Group number (1-based): 1,1,2,2,3,3,...
        sub_idx = (idx % 2) + 1  # Sub-index (1 or 2): 1,2,1,2,1,2,...
        new_key = f"landmark_{group}_{sub_idx}"
        renamed_landmarks[new_key] = value
    
    # Copy non-landmark keys unchanged
    for key, value in landmarks.items():
        if not (isinstance(key, str) and key.isdigit()):
            renamed_landmarks[key] = value
    
    renamed_data[subject_id] = renamed_landmarks

# Save the renamed landmarks to the output file
with open(output_file, "w") as f:
    json.dump(renamed_data, f, indent=2)

print(f"Renamed landmarks saved to {output_file}")
