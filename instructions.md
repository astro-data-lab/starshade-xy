# Instructions for CNN train and test

Experimental Data
---------------------
To process experimental data (add spiders to image and get true positions):

1. Run script `lab_experiments/processing_data/process_experiment.py`
    - Make sure `data_dir`, `session`, `run` are pointed to the location of the data for a given experiment run.

    - Toggle `is_median` parameter to choose to take median of multiple exposure images at each position.

Simulated Data
---------------------
To generate simulated images for training:

1. Run script `quadrature_code/generate_images.py` to solve diffraction equation
    - Select `width` and `num_steps`.


2. Run script `quadrature_code/add_noise.py` to add noise to images
    - Provide list of SNRs to train over with parameter `multi_SNRs`.

Training CNN
---------------------
To train, test, and plot results of CNN:

1. Run script `cnn.py` to train on simulated data.

2. Run script `testmodel.py` to test CNN on experimental data.

3. Run script `plot_results.py` to plot map and histogram of errors.
