# This will distribute the tomograms across the available GPUs.
# Set --num-gpus to the number of GPUs you are using
# For each GPU, start one inference. Set --gpu-id 0 for each of these.
# Give each GPU a unique output file!
# Example: For 4 GPUs, --num-gpus should be 4, and there should be 4 inference scripts running with --gpu-id 0, 1, 2, and 3
# the '&' at the end of the call is important to make the scripts run in parallel!


# here is an example for 2 GPUs
python inference.py --input-dir /media/isensee/raw_data/bact_motors/official/test \
    --output-file /media/isensee/raw_data/bact_motors/official/submission_gpu0.csv \
    --ckpt-dir /media/isensee/data/results_nnUNet_remake/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 \
    --fold "('all',)"\
    --threshold 0.15\
    --gpu-id 0\
    --num-gpus 2 &

python inference.py --input-dir /media/isensee/raw_data/bact_motors/official/test \
    --output-file /media/isensee/raw_data/bact_motors/official/submission_gpu1.csv \
    --ckpt-dir /media/isensee/data/results_nnUNet_remake/Dataset189_Kaggle2025_BYU_FlagellarMotors_mergedExternalBartleyNonBartley_512/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256 \
    --fold "('all',)"\
    --threshold 0.15\
    --gpu-id 1\
    --num-gpus 2 &

wait

# write header from first file, then append data (skipping headers) from both. Adapt this when using more GPUs! Remember to remove the header of all but the first file!
(head -n 1 submission_gpu0.csv && tail -n +2 -q submission_gpu0.csv submission_gpu1.csv) > submission.csv