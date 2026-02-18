from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json
from nnunetv2.paths import nnUNet_raw

if __name__ == '__main__':
    source_train_coords = load_json('/home/isensee/drives/E132-Rohdaten/nnUNetv2/Dataset186_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartley_corrected/train_coordinates.json')
    source_train_coords_origshape = load_json('/home/isensee/drives/E132-Rohdaten/nnUNetv2/Dataset186_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartley_corrected/train_coordinates_forOrigShapes.json')

    # overwrite all 189 coordinates with the ones from 186
    ds = ['Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512']
    for d in ds:
        tc = load_json(join(nnUNet_raw, d, 'train_coordinates.json'))
        tcos = load_json(join(nnUNet_raw, d, 'train_coordinates_forOrigShapes.json'))

        tc.update(source_train_coords)
        tcos.update(source_train_coords_origshape)
        save_json(tc, join(nnUNet_raw, d, 'train_coordinates.json'))
        save_json(tcos, join(nnUNet_raw, d, 'train_coordinates_forOrigShapes.json'))
