import itertools
import concurrent.futures
from copy import deepcopy
from typing import Tuple, List

import SimpleITK as sitk
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.morphology.morphology_helper import generate_ball
from batchgenerators.utilities.file_and_folder_operations import join, subfiles, maybe_mkdir_p, save_json

from nnunetv2.dataset_conversion.Dataset119_ToothFairy2_All import load_json
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.data_augmentation.kaggle_byu_motor_regression import paste_tensor_optionalMax


def generate_segmentation(shape, coordinates: dict[int, List[int]], radius: int = 1, use_max = False):
    sphere = generate_ball([radius] * 3, dtype=np.uint8)
    seg = np.zeros(shape, dtype=np.uint8)

    for lb, ci in coordinates.items():
        bbox = [[i - radius, i + radius + 1] for i in ci]
        paste_tensor_optionalMax(seg, sphere * lb, bbox, use_max=use_max)
        # insert_crop_into_image(seg, sphere * lb, bbox)

    return seg


def process_case(av: str,
                 image_file: str,
                 source_dir: str,
                 imagesTr: str,
                 labelsTr: str,
                 target_spacing: float,
                 landmark_idx_dict: dict) -> Tuple[str, dict]:
    import torch
    from torch.nn.functional import interpolate

    img_path = join(source_dir, 'volumes', image_file)
    json_path = join(source_dir, 'landmark_anno', av + '_landmark.json')

    img = sitk.ReadImage(img_path)
    shape = sitk.GetArrayFromImage(img).shape

    dct = load_json(json_path)
    landmarks = dct['landmarks']
    spacing = dct['spacing']
    landmarks = {landmark_idx_dict[i]: [j[2], j[1], j[0]] for i, j in landmarks.items()}

    # resize image to target spacing if needed
    if spacing != target_spacing:
        img_arr = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
        new_shape = [int(np.round(i / spacing * target_spacing)) for i in img_arr.shape]
        img_resized = interpolate(img_arr[None, None], size=new_shape, mode='trilinear')[0, 0]
        for l in landmarks.keys():
            converted = convert_coordinates([landmarks[l]], img_arr.shape, img_resized.shape)[0]
            landmarks[l] = converted
        img_new = sitk.GetImageFromArray(img_resized.numpy())
        img_new.SetSpacing([target_spacing] * 3)
        img_new.SetOrigin(img.GetOrigin())
        img_new.SetDirection(img.GetDirection())
        img = img_new
        shape = new_shape

    seg = generate_segmentation(shape, landmarks, radius=4)
    seg = sitk.GetImageFromArray(seg)
    seg.SetSpacing([target_spacing] * 3)
    seg.SetOrigin(img.GetOrigin())
    seg.SetDirection(img.GetDirection())

    img.SetSpacing([target_spacing] * 3)
    sitk.WriteImage(img, join(imagesTr, av + '_0000.nii.gz'))
    sitk.WriteImage(seg, join(labelsTr, av + '.nii.gz'))

    return av, landmarks


if __name__ == '__main__':
    ###
    # This dataset is an absolute mess.
    # Nothing fits, not all images are in the same orientation, many images don't have proper geometries and spacings. Some images DO have the spacing set.
    # File format is inconsistent.
    # How can one publish like this?
    # If you are wondering about the transpose mess, I was wondering about that as well...
    # We resize all images to a 0.5 mm spacing during dataset conversion just because this dataset is such a mess...
    ###
    torch.set_num_threads(16)

    source_dir = '/media/isensee/raw_data/FPOSE_original'
    dataset_name = 'Dataset737_FPOSE'
    base_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(base_dir)

    json_files = subfiles(join(source_dir, 'landmark_anno'), join=False)

    # not all volumes are annotated
    annotated_volumes = [i[:-14] for i in json_files]

    imagesTr = join(base_dir, 'imagesTr')
    imagesTs = join(base_dir, 'imagesTs')
    labelsTr = join(base_dir, 'labelsTr')
    maybe_mkdir_p(labelsTr)
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(imagesTr)

    image_files = subfiles(join(source_dir, 'volumes'), join=False)

    # discover landmarks
    all_landmarks = set([k for i in json_files for k in load_json(join(source_dir, 'landmark_anno', i))['landmarks'].keys()])

    idx_landmark_dict = {i+1: j for i, j in enumerate(all_landmarks)}
    landmark_idx_dict = {j: i for i, j in idx_landmark_dict.items()}

    target_spacing = 0.5
    all_landmarks = {}

    remaining = deepcopy(image_files)
    # Parallel execution
    args = []
    for av in annotated_volumes:
        image = [i for i in image_files if i.startswith(av)]
        assert len(image) == 1
        image = image[0]
        args.append((av, image, source_dir, imagesTr, labelsTr, target_spacing, landmark_idx_dict))

    all_landmarks = {}

    with concurrent.futures.ProcessPoolExecutor(14) as executor:
        for name, lm in executor.map(process_case, *zip(*args)):
            all_landmarks[name] = lm

    # cannot use remaining images because they don't have json files with reliable spacing information.
    # Not using these with potentially incorrect spacing...

    # yangxinknow@gmail.com, nidong@szu.edu.cn
    # https://arxiv.org/html/1910.04935v2
    generate_dataset_json(
        base_dir,
        {0: 'US'},
        {'background': 0, **landmark_idx_dict},
        len(annotated_volumes) - len(remaining),
        '.nii.gz',
        citation='See license!',
        regions_class_order=None,
        dataset_name=dataset_name,
        reference='https://arxiv.org/html/1910.04935v2',
        overwrite_image_reader_writer='NibabelIOWithReorient',
        license='The authors of the paper generously shared the dataset with Alexandra Ertl under the following conditions:\n'
                '"Since these two large-scale 3D ultrasound datasets are expensive and time-consuming in building and under some certain IRB constraints in release, please strictly follow the following protocol:\n'
                'a. These two datasets can only be used for pure research purposes. Business uses are not allowed.\n'
                'b. These two datasets can only be accessed by Alexandra Ertl, Shuhan Xiao, Professor Klaus Maier-Hein and Dr. Fabian Isensee.\n'
                'c. Please cite the related paper as necessary references if the associated dataset is used:\n'
                ' [1] Chaoyu Chen, Xin Yang, et al. "FetusMapV2: Enhanced fetal pose estimation in 3D ultrasound", Medical Image Analysis, Volume 91, 2024, 103013, ISSN 1361-8415\n'
                ' [2] C. Chen et al., "Region Proposal Network with Graph Prior and Iou-Balance Loss for Landmark Detection in 3D Ultrasound," 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI)\n'
                'Our team is very interested in working with you and willing to contribute more to the general landmark detection task, like the discussion, method development, benchmarking and testing, hosting Challenge and extensive application in many hospitals.\n'
                'If it is possible, please add these names as co-author in 1 high quality paper based on these two datasets: Dong Ni, Xin Yang, Chaoyu Chen, Wenlong Shi. This is not a prerequisite in sharing these two datasets, but we hope the team efforts can be recognized to some extent."',
        converted_by='Fabian Isensee',
        note='This is a landmark detection dataset. Not to be used for semantic segmentation.',
    )

    save_json(all_landmarks, join(base_dir, 'landmark_coordinates.json'))