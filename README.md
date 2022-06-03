## Machine Learning for Feature Extraction from Ultrasound

# Introduction

This repository contains the code for a semester project in Spring semester 2021: Machine Learning for extracting muscle features using Ultrasound

It contains:
- An automatic labeling tool to extract the pennation angle from ultrasound images
- Four different machine learning methods to predict the pennation angle from raw ultrasound data

# Structure of the repository

This repository contains:
 
- `scripts` folder contains python scripts that perform the feature extraction and ML predictions
    - The `extract_features` folder contains files contains the file `extract_features_from_images.py` to extract pennation angles from ultrasound images and the file `extract_features_from_raw` to extract hand-crafted features from raw ultrasound data (used for some of the ML methods)
    - The `train_models` folder contains files to run crossvalidation with the four ML methods and its subfolder, `gridsearch` contains files to run gridsearch
    - the `visualize` folder contains scripts to visualize the raw data and the results from the labeling tool and ML predictions.

- The `utils` folder contains functions used by the scripts


# Installation

To install the project the [miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager can be used and a virtual environment created.

To do it:
1. Download the repository to your local PC, along with the [PyBF library](https://iis-git.ee.ethz.ch/vsergei/pybf/-/tags/student_projects_spring_2021) (tag - `student_projects_spring_2021`)
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
6. Then type the following command in terminal to make the script  `install_requirements.sh` executable:
```bash
chmod +x install_requirements.sh
```
7. Then execute the script to install the packages from `requirements.txt`:
```bash
./install_requirements.sh
```
The scripts goes through the packages listed in `requirements.txt` and tries to install them with conda. If not successful, it tries to do the same with pip.


# Usage

Before running the scripts, a `results` folder should be created in the main repository and a `frames` subfolder should be created in this folder. Each script should be run from the folder containing that particular script.

- To extract pennation angles from ultrasound images, the `scripts/extract_features/extract_features_from_images.py` file should be run.
The path to the image dataset and the path to the pybf library should be specified at the beginning of the file. The results are written to the `results` folder.
- To extract hand-crafted features from raw ultrasound data, the `scripts/extract_features/extract_features_from_images.py` file should be run.
The path to the dataset and the path to the pybf library should be specified at the beginning of the file. The output .csv file containing the features is written to the `results` folder.
- To run cross-validation for any of the four ML methods, `scripts/train_models/crossval_xxx.py` should be run, where xxx stands for the method of interest. For each file, the path to the raw dataset and pybf library or the .csv file with the hand-crafted features should be specified, as well as the path to the .csv file containing the ground truth pennation angles. The predicted pennation angles are saved as .csv files in the `results` folder.


# Licence 

