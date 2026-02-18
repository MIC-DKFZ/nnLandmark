import os
import json
import re
import numpy as np
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt

# -----------------------------
# Paths (from your message)
# -----------------------------
imagesTs_dir_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/imagesTs"
gt_json_path_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/all_landmarks_voxel.json"
name_to_label_path_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/name_to_label.json"
datasets = [
    #"Dataset732_Afids", 
    #"Dataset735_MML_comp",
    #"Dataset737_DMGLD_LFC",
    "Dataset739_Fposev2",
    #"Dataset740_PDDCA",
    #"Dataset742_FeTA_2_4",
    ]

# Provide up to 7 prediction JSON paths (fills Method 1..7)
pred_json_paths_template = [
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_ResEncM/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_ResEncL/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_BiFormerPlans_128128128/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/baselines/H3DE/{dataset}/BiFormer_Unet/predictions/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large/landmarker_test_results/{dataset}/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large/SRLD_test_results/{dataset}/prediction_all_landmark_voxel.json",
]
d = datasets[0]
out_dir = f"/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large/{d}"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Figure settings
# -----------------------------
N_METHODS_MAX = 7
N_COLS = 1 + N_METHODS_MAX  # GT + up to 7 methods
col_titles = ["GT", "nnLandmark", "nnLandmark ResEncM", "nnLandmark ResEncL", "nnLandmark H3DE", "H3DE", "landmarker", "SRLD"]

half_size = 40
marker_size = 18
marker_color_gt = "lime"
marker_color_pred = "red"
transpose = False  # Whether to transpose loaded images for LFC
invert_feta = False  # Whether to flip FeTA images in left-right direction

# -----------------------------
# Utilities
# -----------------------------
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def find_image_for_subject(images_dir, subject_key):
    """
    Find the image file for a subject. Supports .nii.gz and .nrrd formats.
    """
    candidates = [
        os.path.join(images_dir, f"{subject_key}_0000.nii.gz"),
        os.path.join(images_dir, f"{subject_key}.nii.gz"),
        os.path.join(images_dir, f"{subject_key}.nrrd"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for fn in sorted(os.listdir(images_dir)):
        if subject_key in fn and (fn.endswith(".nii.gz") or fn.endswith(".nrrd")):
            return os.path.join(images_dir, fn)
    return None

def as_landmark_dict(subject_entry, name_to_label):
    """
    Normalize a subject's landmark storage into a dictionary:
        'landmark_#' or other names (e.g., 'chin', 'nose') -> np.array([x, y, z])
    """
    label_to_name = {int(v): k for k, v in name_to_label.items()}

    out = {}
    for k, v in subject_entry.items():
        if v is None:
            continue

        coord = np.asarray(v, dtype=float)
        if coord.shape != (3,):
            continue

        # Case 1: Use the key directly if it matches a landmark name
        if isinstance(k, str) and k in name_to_label:
            out[k] = coord
            continue

        # Case 2: Numeric label (int/float-like string)
        lab = None
        if isinstance(k, (int, np.integer)):
            lab = int(k)
        elif isinstance(k, str):
            kk = k.strip()
            if re.fullmatch(r"\d+(\.0+)?", kk):  # Matches integers or floats like "1.0"
                lab = int(float(kk))

        if lab is not None and lab in label_to_name:
            out[label_to_name[lab]] = coord
            continue

        # Case 3: Use the key directly if no other match is found
        if isinstance(k, str):
            out[k] = coord

    return out

def extract_axial_patch(vol, xyz, half):
    # Ensure the coordinates are integers
    x, y, z = [int(round(c)) for c in xyz]

    # Ensure z is within bounds
    z = np.clip(z, 0, vol.shape[2] - 1)

    # Extract the patch
    x0, x1 = x - half, x + half + 1
    y0, y1 = y - half, y + half + 1

    pad_x0 = max(0, -x0)
    pad_y0 = max(0, -y0)
    pad_x1 = max(0, x1 - vol.shape[0])
    pad_y1 = max(0, y1 - vol.shape[1])

    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(vol.shape[0], x1); y1 = min(vol.shape[1], y1)

    patch = vol[x0:x1, y0:y1, z]
    if any([pad_x0, pad_x1, pad_y0, pad_y1]):
        patch = np.pad(
            patch,
            ((pad_x0, pad_x1), (pad_y0, pad_y1)),
            mode="constant",
            constant_values=np.min(vol)
        )
    return patch, z  # Return the patch and the slice number

def load_mre(pred_json_path, subject_id=None, landmark_name=None):
    """
    Load the MRE from the summary_mm.json file in the same directory as the prediction JSON.

    Args:
        pred_json_path (str): Path to the prediction JSON file.
        subject_id (str, optional): Subject ID to retrieve detailed MRE for a specific subject.
        landmark_name (str, optional): Landmark name to retrieve MRE for a specific landmark.

    Returns:
        float or None: The MRE value if available, otherwise None.
    """
    summary_path = os.path.join(os.path.dirname(pred_json_path), "summary_mm.json")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    # If subject_id and landmark_name are provided, retrieve the detailed MRE
    if subject_id and landmark_name:
        return summary.get("detailed_results", {}).get(subject_id, {}).get(landmark_name, None)

    # If only subject_id is provided, retrieve the subject-level MRE
    if subject_id:
        subject_results = summary.get("detailed_results", {}).get(subject_id, {})
        if subject_results:
            return np.mean(list(subject_results.values()))

    # If no subject_id or landmark_name is provided, return the overall MRE
    return summary.get("MRE_micro", None)

def load_image(image_path):
    """
    Load an image file. Supports .nii.gz and .nrrd formats.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The image data as a NumPy array.
    """
    if image_path.endswith(".nii.gz"):
        img = nib.load(image_path)
        return img.get_fdata().astype(np.float32)
    elif image_path.endswith(".nrrd"):
        img, _ = nrrd.read(image_path)
        return img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
# -----------------------------
# Iterate over datasets
# -----------------------------
for dataset in datasets:
    print(f"Processing dataset: {dataset}")

    # Paths for the current dataset
    imagesTs_dir = imagesTs_dir_template.format(dataset=dataset)
    gt_json_path = gt_json_path_template.format(dataset=dataset)
    name_to_label_path = name_to_label_path_template.format(dataset=dataset)
    pred_json_paths = [p.format(dataset=dataset) for p in pred_json_paths_template]

    # Load data
    gt_all = load_json(gt_json_path)
    name_to_label = load_json(name_to_label_path)
    pred_json_paths = pred_json_paths[:N_METHODS_MAX]
    pred_alls = [load_json(p) for p in pred_json_paths]

    # Load MREs for each method
    mres = [load_mre(p) for p in pred_json_paths]

    # Find all subjects (union of all subjects across GT and predictions)
    all_subjects = set(gt_all.keys())
    for p in pred_alls:
        all_subjects |= set(p.keys())
    all_subjects_sorted = sorted(list(all_subjects))

    print(f"Processing {len(all_subjects_sorted)} subjects for dataset {dataset}...")

    # Get unique landmarks
    first_sub = next(iter(gt_all.keys()))
    landmark_names = as_landmark_dict(gt_all[first_sub], name_to_label).keys()
    landmark_names = sorted(landmark_names)
    n_rows = len(landmark_names)

    # Figure dimensions (fixed for all subjects)
    fig_w = N_COLS * 1.0
    fig_h = n_rows * 1.0

    # Process each subject
    for idx, subject_id in enumerate(pred_alls[0].keys()):
        try:
            # Load image
            img_path = find_image_for_subject(imagesTs_dir, subject_id)
            if img_path is None:
                print(f"  [{idx+1}/{len(all_subjects_sorted)}] {subject_id}: Image not found, skipping")
                continue

            vol = load_image(img_path)
            if transpose:
                vol = np.transpose(vol, (2, 1, 0))
                vol = np.flip(vol, axis=1) 
            if invert_feta:
                vol = np.flip(vol, axis=0)  # Flip left-right for FeTA dataset

            # Display scaling
            p1, p99 = np.percentile(vol, [10, 100])
            vol_disp = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

            # Landmarks
            gt_lms = as_landmark_dict(gt_all.get(subject_id, {}), name_to_label)
            pred_lms_list = [as_landmark_dict(p.get(subject_id, {}), name_to_label) for p in pred_alls]

            if transpose:
                # Adjust landmark coordinates if transposed
                for lm_dict in [gt_lms] + pred_lms_list:
                    for k in lm_dict.keys():
                        x, y, z = lm_dict[k]
                        y = vol.shape[1] - 1 - y
                        lm_dict[k] = np.array([z, y, x])
            if invert_feta:
                # Adjust landmark coordinates for left-right flip
                for lm_dict in [gt_lms] + pred_lms_list:
                    for k in lm_dict.keys():
                        x, y, z = lm_dict[k]
                        x = vol.shape[0] - 1 - x
                        lm_dict[k] = np.array([x, y, z])

            # Create figure
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(fig_w, fig_h), dpi=300)
            if n_rows == 1:
                axes = axes[None, :]

            fig.subplots_adjust(
                left=0.06, right=1,
                bottom=0.02, top=1,
                wspace=0.01, hspace=0.01
            )

            # Diagonal method names as text
            for c, title in enumerate(col_titles):
                ax = axes[0, c]
                ax.text(
                    0.02, 1.02, title,
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    rotation=30,
                    fontsize=9
                )

            # Plot each landmark row
            for r, lm_name in enumerate(landmark_names):
                gt_xyz = gt_lms.get(lm_name)
                if gt_xyz is not None:
                    gt_patch, z_gt = extract_axial_patch(vol_disp, gt_xyz, half_size)
                else:
                    gt_patch = np.zeros((2 * half_size + 1, 2 * half_size + 1))

                # GT column
                ax = axes[r, 0]
                ax.imshow(gt_patch.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
                ax.set_ylabel(lm_name, rotation=90, labelpad=10, fontsize=8, va="center")
                ax.set_frame_on(False)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(
                        0.98, 0.98, f"Slice {z_gt}",
                        transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=8, color="white",
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.0)
                    )
                ax.scatter([half_size], [half_size], s=marker_size, c=marker_color_gt, marker="x")

                # Method columns
                for mi in range(N_METHODS_MAX):
                    c = 1 + mi
                    ax = axes[r, c]
                    pred_xyz = pred_lms_list[mi].get(lm_name)
                    if pred_xyz is not None:
                        pred_patch, z_pred = extract_axial_patch(vol_disp, pred_xyz, half_size)
                        ax.imshow(pred_patch.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
                        ax.scatter([half_size], [half_size], s=marker_size, c=marker_color_pred, marker="x")
                        # Add slice number for each prediction
                        ax.text(
                            0.98, 0.98, f"Slice {z_pred}",
                            transform=ax.transAxes,
                            ha="right", va="top",
                            fontsize=8, color="white",
                            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.0)
                        )
                        # Add landmark-level MRE for each prediction at the top-left
                        landmark_mre = load_mre(pred_json_paths[mi], subject_id=subject_id, landmark_name=lm_name)
                        if landmark_mre is not None:
                            ax.text(
                                0.02, 0.02, f"MRE: {landmark_mre:.2f} mm",
                                transform=ax.transAxes,
                                ha="left", va="bottom",
                                fontsize=8, color="white",
                                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.0)
                            )
                    else:
                        ax.axis("off")
                    ax.set_frame_on(False)
                    ax.set_xticks([]); ax.set_yticks([])

            # Rasterize images for PDF
            for ax_row in axes:
                for ax in ax_row:
                    for im in ax.get_images():
                        im.set_rasterized(True)

            # Save PDF
            out_pdf = os.path.join(out_dir, f"qualitative_{dataset}_{subject_id}.pdf")
            fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"  [{idx+1}/{len(all_subjects_sorted)}] {subject_id} -> {out_pdf}")

        except Exception as e:
            print(f"  [{idx+1}/{len(all_subjects_sorted)}] {subject_id}: Error: {e}")

    print(f"\nDone processing dataset {dataset}! PDFs saved to {out_dir}/")