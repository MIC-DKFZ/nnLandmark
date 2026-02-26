import os
import json
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# -----------------------------
# Paths (from your message)
# -----------------------------
imagesTs_dir_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/imagesTs"
gt_json_path_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/all_landmarks_voxel.json"
name_to_label_path_template = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/nnunet_data/nnUNet_raw/{dataset}/name_to_label.json"
datasets = ["Dataset732_Afids", "Dataset735_MML_comp"]

# Provide up to 7 prediction JSON paths (fills Method 1..7)
pred_json_paths_template = [
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_ResEncM/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_ResEncL/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/evaluation/{dataset}/nnLandmark_fabi/prediction_BiFormerPlans_128128128/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/baselines/H3DE/{dataset}/BiFormer_Unet_128/predictions/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large/landmarker_test_results/{dataset}/prediction_all_landmark_voxel.json",
    "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large/SRLD_test_results/{dataset}/prediction_all_landmark_voxel.json",
]

out_dir = "/home/a332l/Desktop/E132-Projekte/Projects/2024_Ertl_nnLandmark/paper/qualitative_results_large"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Figure settings
# -----------------------------
N_METHODS_MAX = 7
N_COLS = 1 + N_METHODS_MAX  # GT + up to 7 methods
col_titles = ["GT", "nnLandmark", "nnLandmark ResEncM", "nnLandmark ResEncL", "nnLandmark H3DE", "H3DE", "landmarker", "SRLD"]

half_size = 20
marker_size = 18
marker_color_gt = "lime"
marker_color_pred = "red"

# -----------------------------
# Utilities
# -----------------------------
def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def find_image_for_subject(images_dir, subject_key):
    candidates = [
        os.path.join(images_dir, f"{subject_key}_0000.nii.gz"),
        os.path.join(images_dir, f"{subject_key}.nii.gz"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for fn in sorted(os.listdir(images_dir)):
        if subject_key in fn and fn.endswith(".nii.gz"):
            return os.path.join(images_dir, fn)
    return None

def as_landmark_dict(subject_entry, name_to_label):
    """
    Normalize a subject's landmark storage into dict:
        'landmark_#' -> np.array([x,y,z])
    """
    label_to_name = {int(v): k for k, v in name_to_label.items()}

    out = {}
    for k, v in subject_entry.items():
        if v is None:
            continue

        coord = np.asarray(v, dtype=float)
        if coord.shape != (3,):
            continue

        # Case 1: already a landmark name
        if isinstance(k, str) and k.startswith("landmark_"):
            out[k] = coord
            continue

        # Case 2: numeric label (int/float-like string)
        lab = None
        if isinstance(k, (int, np.integer)):
            lab = int(k)
        elif isinstance(k, str):
            kk = k.strip()
            if re.fullmatch(r"\d+(\.0+)?", kk):
                lab = int(float(kk))

        if lab is not None and lab in label_to_name:
            out[label_to_name[lab]] = coord

    return out

def extract_axial_patch(vol, xyz, half):
    x, y, z = [int(round(c)) for c in xyz]
    z = np.clip(z, 0, vol.shape[2] - 1)

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
    return patch

def lm_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return 1_000_000

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

    # Find all common subjects
    common = set(gt_all.keys())
    for p in pred_alls:
        common &= set(p.keys())
    common_sorted = sorted(list(common))

    if not common_sorted:
        print(f"No common subjects found for dataset {dataset}.")
        continue

    # Process only the first subject
    subject_id = common_sorted[0]
    print(f"Processing first subject: {subject_id}")

    # Load image
    img_path = find_image_for_subject(imagesTs_dir, subject_id)
    if img_path is None:
        print(f"Image not found for subject {subject_id}, skipping.")
        continue

    img = nib.load(img_path)
    vol = img.get_fdata().astype(np.float32)

    # Display scaling
    p1, p99 = np.percentile(vol, [1, 99])
    vol_disp = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Landmarks
    gt_lms = as_landmark_dict(gt_all[subject_id], name_to_label)
    pred_lms_list = [as_landmark_dict(p[subject_id], name_to_label) if subject_id in p else {} for p in pred_alls]

    # Get unique landmarks
    landmark_names = sorted(gt_lms.keys(), key=lm_sort_key)
    n_rows = len(landmark_names)

    # Figure dimensions
    fig_w = N_COLS * 1.0
    fig_h = n_rows * 1.0

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
        gt_xyz = gt_lms[lm_name]
        gt_patch = extract_axial_patch(vol_disp, gt_xyz, half_size)

        # GT column
        ax = axes[r, 0]
        ax.imshow(gt_patch.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
        ax.scatter([half_size], [half_size], s=marker_size, c=marker_color_gt, marker="x")
        ax.set_ylabel(lm_name, rotation=90, labelpad=10, fontsize=8, va="center")
        ax.set_frame_on(False)
        ax.set_xticks([]); ax.set_yticks([])

        # Method columns
        for mi in range(N_METHODS_MAX):
            c = 1 + mi
            ax = axes[r, c]
            if mi < len(pred_lms_list) and lm_name in pred_lms_list[mi]:
                pred_xyz = pred_lms_list[mi][lm_name]
                pred_patch = extract_axial_patch(vol_disp, pred_xyz, half_size)
                ax.imshow(pred_patch.T, cmap="gray", origin="lower", interpolation="nearest", aspect="auto")
                ax.scatter([half_size], [half_size], s=marker_size, c=marker_color_pred, marker="x")
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

    print(f"Figure saved to {out_pdf}")