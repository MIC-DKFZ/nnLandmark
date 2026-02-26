# lsf22-gpu01
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 185 3d_fullres 1 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 185 3d_fullres 2 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=2 nnUNetv2_train 185 3d_fullres 3 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=3 nnUNetv2_train 142 3d_fullres 4 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=4 nnUNetv2_train 142 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=5 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA "
#screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=6 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK50Loss "
#screen -dm bash -c ". ~/load_env_kaggle2025_byu.sh && CUDA_VISIBLE_DEVICES=7 nnUNetv2_train 142 3d_fullres 0 -tr MotorRegressionTrainer_MSELoss_lr1en3 "


bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=12G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 185 3d_fullres_filt24 all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA -p nnUNetResEncUNetMPlans"


bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 1 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 2 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 3 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 4 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling"

bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres 1 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres 2 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres 3 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres 4 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 186 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA"

bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 0 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 1 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 2 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 3 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres 4 -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"
bsub -q gpu-debian -gpu num=1:j_exclusive=yes:gmem=33G ". ~/load_env_kaggle2025_byu.sh && nnUNetv2_train 187 3d_fullres all -tr MotorRegressionTrainer_BCEtopK20Loss_moreDA_FPoversampling -p nnUNetResEncUNetMPlans"

