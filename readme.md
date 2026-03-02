# Welcome to nnLandmark!

This repository contains the implementation of nnLandmark, a self-configuring framework for 3D medical landmark detection.

🚀 nnLandmark is accepted to MIDL 2026! 
Find the preprind on arxiv: &nbsp; &nbsp;   [![arXiv](https://img.shields.io/badge/arXiv-2404.03010-B31B1B.svg)](https://arxiv.org/abs/2504.06742)

Copyright German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license.

## Installation
The repository you see here is a fork of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Please head over there to read more about it.

We strongly recommend installing this in a dedicated virtual environment (for example conda).
We recommend using a Linux based operating system, for example Ubuntu. Windows should work as well but is not tested.

Some dependencies should be installed manually:
- Install python, we used 3.13.
- Install pytorch according to the instructions on the [pytorch website](https://pytorch.org/get-started/locally/). We recommend version 2.8. 
- Pick the correct CUDA version for your system, we used 12.8.

Now you can just clone this repository and install it:

```commandline
git clone https://github.com/MIC-DKFZ/nnLandmark.git`
cd nnLandmark
pip install -e .
```

## Data Format

### Path setup
We are using the same path system as nnU-Net, defined as environment variables pointing it to raw data, preprocessed data and results. Set them with

```
export nnLM_results=/home/isensee/nnLM_results
export nnLM_preprocessed=/home/isensee/nnLM_preprocessed
export nnLM_raw=/home/isensee/nnLM_raw
```
Make sure at least `$nnLM_preprocessed` (but ideally all of them) are on a fast storage such as a local SSD or very good network drive! 

RECOMMENDED: Add these lines to your `.bashrc` file (or whatever you are using) so that the environment variables are set automatically. If you don't do this you need to export them every time you open a new terminal.

### Images and Labels
 Here we follow the nnU-Net format. The training data is stored in imagesTr and labelsTr folders. The labels are multi-label segmetnation maps. Each landmark class belongs to a specific label value, this must be consistent throughout the dataset! The landmark location is represented by a 3x3x3 cube round the target voxel. Generally the size is irrelevant, as during training the location will be extracted be the center of mass of the segmentation. However, it must be ensured that proximate labels do not overlap, as this would distort the location. 

### Additional JSONs

- dataset.json: Just as in nnU-Net.
- name_to_label.json: Contains all landmark class names as keys and the respective segmentation label values (starting from 1).

```bash
{
  "landmark_1": 1,
  "landmark_2": 2,
}
```

- spacing.json: This spacing information is used in the evaluation. For each case it contains a image_spacing, taken from the image metadata, and annotation_spacing, taken from the landmark annotation files. This is because some datasets are published with no/wrong image spacing. nnLandmark defaults to look for image_spacing and, if it's null, falls back to annotation_spacing. 

```bash
{
  "case_xyz": 
  {
    "image_spacing": [
      0.5,
      0.5,
      0.5
    ],
    "annotation_spacing": null
  }
}
```

- all_landmarks_voxel.json: Voxel coordinate annotations for all cases (train and test). 

```bash
{
  "case_xyz": 
  {
    "landmark_1": [
      13,
      19,
      89
    ],
    "landmark_2": [
      19,
      75,
      85
    ],
  }
}
```

### Public Dataset Conversion Scripts

We provide dataset conversion scripts under nnlandmark/dataset_conversion/nnLandmark for the following public landmark detection datasets. The folder also contains all train/test splits of the datasets, either the official, published splits or, if not available, the custom split we created and used in the nnLandmark paper MIDL 2026.
Please check the respective licenses of the datasets before using them!

- AFIDs: https://github.com/afids/afids-data
- MML: https://github.com/ithet1007/mmld_code
- DMGLD LFC: https://github.com/lhaof/DGMLD
- PDDCA 1.4.1: https://www.imagenglab.com/newsite/pddca/ 
- FeTA 2.4: https://www.synapse.org/Synapse:syn25649159/wiki/610007 


## Experiment Planning and Preprocessing

We use the experiment planning and preprocessing functionality of nnU-Net as is. 

```bash
nnLM_plan_and_preprocess \
     -d DATASET_ID \
     -c 3d_fullres \
     --verify_dataset_integrity
```
To add the experiment plans for using the ResEncM architecture, our recommendation for the best results, :

```bash
nnLM_plan_experiment \
    -d DATASET \
    -pl nnUNetPlannerResEncM
```


## Training

Start a nnU-Net training with the nnLandmark trainer. For using the ResEncM architecture plans, add the respective flag:

```bash
nnLM_train \
    DATASET_NAME_OR_ID \
    3d_fullres \
    FOLD \
    -p nnUNetResEncUNetMPlans
```


## Predictions

Use the costum nnLandmark predict script to predict a raw image folder. For using the ResEncM architecture plans, add the respective flag:

```bash
nnLM_predict \
    -i /path/to/nnUNet_raw/DATASET_ID/imagesTs/ \
    -o /path/to/evaluation/DATASET_ID/predictions/ \
    -d DATASET_ID \
    -c 3d_fullres\
    -p nnUNetResEncUNetMPlans
```

This scrip will create:

- dataset.json, plans.json, predict_from_raw_data_args.json as in nnU-Net
- Multi-label segmentation .nii.gz for each case. Each landmark is represented by a label containing the top 27 voxels of the predicted heatmap.  
- Prediction jsons for each case, containing voxel coordinates and a likelihood for each landmark.


## Evaluation

Use the custom nnLandmark evaluation script:

```bash
nnLM_evaluate \
    -d DATASET_ID \
    -pred /path/to/evaluation/DATASET_ID/predictions/
```

This script will create:

- prediction_all_landmark_voxel.json: Predictions of all cases in voxel coordinates.
- summary_voxel.py: Metrics in voxel
- summary_mm.py: Metrics in mm

### FeTA Biometry Measurement Evaluation

To use the custom nnLandmark feta measurement evaluation script, the landmarks, which act as control points for the measurements, must comply to the following naming convention: "landmark_1_1", "landmark_1_2", "landmark_2_1" etc. The euclidean distance is then taken between the two pairs, "_1" and "_2" of each landmark_x.

```bash
python nnlandmark/evaluation/nnLandmark/evaluate_feta_measurements.py \
    --predictions /path/to/evaluation/Dataset742_FeTA_2_4/nnLandmark_trainer/prediction/prediction_all_landmark_voxel.json \
    --ground_truth /path/to/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/all_landmarks_voxel.json \
    --spacing /path/to/nnunet_data/nnUNet_raw/Dataset742_FeTA_2_4/spacing.json. \
```

This script will create a measurements.py. 

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{ertl2026nnlandmark,
      title={nnLandmark: A Self-Configuring Method for 3D Medical Landmark Detection}, 
      author={Alexandra Ertl and Stefan Denner and Robin Peretzke and Shuhan Xiao and David Zimmerer and Maximilian Fischer and Markus Bujotzek and Xin Yang and Peter Neher and Fabian Isensee and Klaus H. Maier-Hein},
      year={2026},
      eprint={2504.06742},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.06742}, 
}
```

## Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).