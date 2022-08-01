## Machine Learning for Feature Extraction from Ultrasound

# Introduction

This repository contains the code supporting the paper 

`Automatic Extraction of Muscle Fascicle Pennation Angle from Raw Ultrasound Data` 

presented at IEEE Sensors Applications Symposium (SAS) 2022.

The repository contains:
- An automatic labeling tool to extract the pennation angle from ultrasound images
- Implementations of four machine learning methods to predict the pennation angle from raw ultrasound data
- Tests scripts

# Structure of the repository

This repository contains:
 
- `scripts` folder contains python scripts that perform the feature extraction, ML models training and predictions.
    - The `extract_features` folder contains the file `extract_features_from_images.py` to extract pennation angles from ultrasound images and the file `extract_features_from_raw` to extract hand-crafted features from raw ultrasound data (used for some of the ML methods).
    - The `train_models` folder contains files to run crossvalidation with the four ML methods and its subfolder, `gridsearch` contains files to run parameter search.

- The `utils` folder contains functions used by the scripts
- The `tests` folder contains tests scripts and sample datasets


# Installation

To install the project the [miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager can be used and a virtual environment created.

To do it:
1. To clone the main repository with all submodules execute the following command in the terminal: <br /> 
`git clone --recurse-submodules git@github.com:soleym/ml-muscle-feature-extraction.git`

2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Add additional channels for necessary packages by executing the following commands in terminal
```bash
conda config --append channels plotly
conda config --append channels conda-forge
```
4. Move to the repository directory
5. Execute the command below to create a virtual environment named `muscle_env`:
```bash
conda create --name muscle_env python=3.6
```
6. Activate conda environent:
```bash
conda activate muscle_env
```
7. Then type the following command in terminal to make the script  `install_requirements.sh` executable:
```bash
chmod +x install_requirements.sh
```
8. Then execute the script to install the packages from `requirements.txt`:
```bash
./install_requirements.sh
```
The scripts goes through the packages listed in `requirements.txt` and tries to install them with conda. If not successful, it tries to do the same with pip.


# Usage

1. To extract the pennation angles from ultrasound images, the methods from the `scripts/extract_features/extract_features_from_images.py` should be used. The test script `test/code/image_annotation/main.py` provides a relevant example for this.

    - If you extracted the pennatelion angles, then you can train the AE+XGBoost algorithm by running the `scripts/train_models/train_AE_xgb.py`. The script will save the model in the `result` folder of the root directory.

    - After training the AE+XGBoost algorithm, you can run the inference script `test/code/predict_ae_xgboost/main.py` to test the algorithm on the sample dataset and compare the ML prediction with the labels obtained by the image annotation tool.


2. To extract hand-crafted features from raw ultrasound data, the methods from the `scripts/extract_features/extract_features_from_images.py` should be used. The test script `test/code/extract_handcrafted_features/main.py` provides a corresponding example.

    - If you extracted the handcrafted features, then you can train the XGBoost algorithm by running the `scripts/train_models/train_handcrafted_xgb.py`. The script will save the model in the `results` folder of the root directory.

    - After training the AE+XGBoost algorithm, you can run the inference script `test/code/predict_handcrafted_xgboost/main.py` to test the algorithm on the sample dataset and compare the ML prediction with the labels obtained by the image annotation tool.


3. Additionally, you can run the cross-validation procedure for any of the four ML methods described in the paper. In this case, please, run `scripts/train_models/crossval_xxx.py` where `xxx` stands for the method of interest. The predicted pennation angles are saved as .csv files in the `results` folder.

**NOTE**: All the above scripts operate by default with the sample datasets. To get the meaningful results for the ML models, please, provide the custom dataset compliant with the format of the input data (check `test/data/README.md` for further details).


# License
All source code is released under Apache v2.0 license unless noted otherwise, please refer to the LICENSE file for details.
Example datasets under `tests/data` folder are provided under a [Creative Commons Attribution No Derivatives 4.0 International License][cc-by-nd] 

[cc-by-nd]: https://creativecommons.org/licenses/by-nd/4.0/

