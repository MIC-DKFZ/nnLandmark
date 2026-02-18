from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_results


if __name__ == '__main__':
    dataset = 'Dataset142_Kaggle2025_BYU_FlagellarMotors'
    base = join(nnUNet_results, dataset)
    folds = (0, )
    scores = ['f2_max', 'f2_at_0.5']
    with open(join(nnUNet_results, f'{dataset}_results.csv'), 'w') as f:
        f.write(' ')
        for s in scores:
            for fl in folds:
                f.write(f',{s}_fold_{fl}')
        f.write('\n')
        for experiment in subdirs(base, join=False):
            exp_dir = join(base, experiment)
            f.write(experiment)
            for s in scores:
                for fold in folds:
                    fold_dir = join(exp_dir, f'fold_{fold}')
                    validation_dir = join(fold_dir, 'validation')
                    scores_file = join(validation_dir, 'scores.json')
                    if not isfile(scores_file):
                        print(experiment, fold , 'is not done yet')
                        f.write(',NA')
                        continue
                    else:
                        res = load_json(scores_file)
                        sc = res[s]
                        f.write(f',{round(sc, 3)}')
            f.write('\n')

