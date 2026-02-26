from batchgenerators.utilities.file_and_folder_operations import *

from nnlandmark.batch_running.collect_results_custom_Decathlon import collect_results, summarize
from nnlandmark.paths import nnLM_results

if __name__ == '__main__':
    use_these_trainers = {
        'nnUNetTrainer': ('nnUNetPlans', ),
    }
    all_results_file = join(nnLM_results, 'hrnet_results.csv')
    datasets = [2, 3, 4, 17, 20, 24, 27, 38, 55, 64, 82]
    collect_results(use_these_trainers, datasets, all_results_file)

    folds = (0, )
    configs = ('2d', )
    output_file = join(nnLM_results, 'hrnet_results_summary_fold0.csv')
    summarize(all_results_file, output_file, folds, configs, datasets, use_these_trainers)

