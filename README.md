# Position Sensing for Starshade Formation Flying

_for details see the [paper](https://arxiv.org/abs/2204.03853)_

A starshade need to be aligned with its telescopes, e.g. the Roman Space Telescope.
The only way of assessing good alignment comes from the image of the obscured pupil as observed from Roman.
The occluded image shows in its center the spot of Arago, whose position is thus an estimate for any lateral offset.

Current estimation procedures either use a large precomputed template library to match the offsets, or a non-linear model fitting code. So, we design a simple CNN to perform the task `Image -> (dx', dy')`. To further calibrate the model and provide uncertainties, we use a simulation-based inference technique DELFI, which performs the task `(dx', dy') -> (dx'', dy'', Sigma)`, where `Sigma` is the 2x2 covariance matrix of the calibrated positions `(dx'', dy'')`.

## Instructions for Running the Code

### Experimental Data

To process experimental data (add spiders to image and get true positions):

1. Run script `lab_experiments/processing_data/process_experiment.py`
    - Make sure `data_dir`, `session`, `run` are pointed to the location of the data for a given experiment run.

    - Toggle `is_median` parameter to choose to take median of multiple exposure images at each position.

### Simulated Data

To generate simulated images for training:

1. Run script `quadrature_code/generate_images.py` to solve diffraction equation
    - Select `width` and `num_steps`.


2. Run script `quadrature_code/add_noise.py` to add noise to images
    - Provide list of SNRs to train over with parameter `multi_SNRs`.

### Training CNN

To train, test, and plot results of CNN:

1. Run script `cnn.py` to train on simulated data.

2. Run script `testmodel.py` to test CNN on experimental data.

3. Run script `plot_results.py` to plot map and histogram of errors.

### Training DELFI

1. Run script `delfi/build_gauss_testset.py` to create a training with a Gaussian distribution in `(dx, dy)`

2. Run script `delfi/fit_gmm.py` (requires [pyGMMis](https://github.com/pmelchior/pygmmis))

3. Run script `delfi/run_delfi.py` for position and uncertainty estimates