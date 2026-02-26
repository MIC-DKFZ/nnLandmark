1. Preprocess with preprocess_non_bartley_external_data.py
2. Predict using predict_non_bartley_external_data.py, use jobs.sh for inspiration
3. Sampling of new tomograms with data_sampling.py
4. Create a new nnunet dataset from the sampled cases with create_nnunet_dataset.py

None of these scripts are made to be just run as is. You will need to adjust paths in code!