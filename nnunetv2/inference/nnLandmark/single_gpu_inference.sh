python inference.py --input-dir /media/isensee/raw_data/bact_motors/official/test \
    --output-file /media/isensee/raw_data/bact_motors/official/test_out.csv \
    --ckpt-dir /media/isensee/data/results_nnUNet_remake/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 \
    --fold "('all',)"\
    --threshold 0.15\
    --gpu-id 0\
    --num-gpus 1
