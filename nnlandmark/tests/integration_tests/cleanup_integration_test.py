import shutil

from batchgenerators.utilities.file_and_folder_operations import isdir, join

from nnlandmark.paths import nnLM_raw, nnLM_results, nnLM_preprocessed

if __name__ == '__main__':
    # deletes everything!
    dataset_names = [
        'Dataset996_IntegrationTest_Hippocampus_regions_ignore',
        'Dataset997_IntegrationTest_Hippocampus_regions',
        'Dataset998_IntegrationTest_Hippocampus_ignore',
        'Dataset999_IntegrationTest_Hippocampus',
    ]
    for fld in [nnLM_raw, nnLM_preprocessed, nnLM_results]:
        for d in dataset_names:
            if isdir(join(fld, d)):
                shutil.rmtree(join(fld, d))

