# 5LSM0: Neural Networks for Computer Vision

We have used the Fast-SCNN model architecture for training semantic segmentation on the Cityscapes dataset.

This repository contains the modifications and training scripts that were used for training the models.

The trained models and slurm job outputs are provided as well.

There are several Excel sheets with calculations and measurements.

## Overview

We have modified and integrated the code of the Fast-SCNN model architecture to make it compatible with the existing training script.

We have made 2 versions of the training script: perform training with Curriculum Learning or perform training with Curriculum Learning and Knowledge Distillation.

There are 2 different slurm job scripts to either launch the training based on Curriculum Learning or the training based on Curriculum Learning and Knowledge Distillation.

There is a separate Python tool that can be used to evaluate the efficiency of a trained model on the validation dataset.
This tool is performed locally on the computer, as we need to have access to the thop library which was not available on the super cluster.

The trained models are available in the models folder.

The outputs of the slurm jobs that were run for training the models are available in the slurms folder.

We have provided an Excel sheet with the choice of resolutions for images and patches that are used during Curriculum Learning.

There is an Excel sheet with the durations of the simulations to indicate that training becomes more difficult during the curriculum.

And finally, there is an Excel sheet with the efficiency measurements for the different trained models.

### Fast-SCNN model architecture

Fast-SCNN is a lightweight and efficient model architecture designed for real-time semantic segmentation on edge devices.

- **fast_scnn.py**:

We could not find the official implementation of the Fast-SCNN model architecture from the researchers at Toshiba Research Europe and Cambridge University.

But we found an unofficial implementation on GitHub from an AI researcher at Baidu.

This implementation adopts all the main principles from the Fast-SCNN model architecture.

### Training with Curriculum Learning (Train a Teacher model)

Two files need to be modified before starting to train with Curriculum Learning:

- **main.sh**:

You have to update the new experiment id and provide the location of the previous trained Teacher model at a lower resolution.

If you start the training at the lowest resolution, then you have to specify "none" for the previous trained Teacher model.

- **train.py**:

You have to make sure that the resized image dimensions and patch dimensions point to the resolution of the current curriculum:
resized_image_width, resized_image_height, patch_width, patch_height

You can choose between 5 profiles: from low resolution, to medium resolution, high resolution, higher resolution and highest resolution.

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each training run. And each time, you have to update the location of the previous trained Teacher model at a lower resolution.

### Training with Curriculum Learning and Knowledge Distillation (Train a Student model)

Two files need to be modified before starting to train with Curriculum Learning and Knowledge Distillation:

- **main_distillation.sh**:

You have to update the new experiment id and provide the location of the previous trained Student model at a lower resolution.

If you start the training at the lowest resolution, then you have to specify "none" for the previous trained Student model.

You also have to provide the location of the available trained Teacher model (who has learned the entire curriculum already).

- **train_distillation.py**:

You have to make sure that the resized image dimensions and patch dimensions point to the resolution of the current curriculum:
resized_image_width, resized_image_height, patch_width, patch_height

You can choose between 5 profiles from low resolution, to medium resolution, high resolution, higher resolution and highest resolution.

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each training run. And each time, you have update the location of the previous trained Student model at a lower resolution.

You keep the same available trained Teacher model during the entire curriculum.

### Slurm job scripts to start training

There are two different slurm job scripts to either train with Curriculum Learning or Curriculum Learning with Knowledge Distillation.

 **jobscript_slurm.sh**:

This is the slurm job script to launch the training with Curriculum Learning.

 **jobscript_slurm_distillation.sh**:

This is the slurm job script to launch the training with Curriculum Learning and Knowledge Distillation.

### Tool to evaluate the efficiency of a trained model on the validation dataset

- **evaluate_efficiency.sh**:

You have to update the new experiment id and provide the location of the previous trained Student model at a lower resolution.

If you start the training at the lowest resolution, then you have to specify "none" for the previous trained Student model.

You also have to provide the location of the available trained Teacher model (who has learned the entire curriculum already).

- **train_distillation.py**:

You have to make sure that the resized image dimensions and patch dimensions point to the resolution of the current curriculum:
resized_image_width, resized_image_height, patch_width, patch_height

You can choose between 5 profiles from low resolution, to medium resolution, high resolution, higher resolution and highest resolution.

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each training run.

You have to update the location of the previous trained Student model at a lower resolution.

You keep the same available trained Teacher model during the entire curriculum.

### Trained models

- **models/**:

This folder containes the trained models:
Models 1-5 belong to Curriculum Learning run 1 (no weight decay during training)
Models 6-10 belong to Curriculum Learning run 2 (weight decay during training)
Models 11-15 belong to Curriculum Learning and Knowledge Distillation run 1 (no weight decay during training)
Models 16-20 belong to Curriculum Learning and Knowledge Distillation run 2 (weight decay during training)

### Slurm job outputs

- **slurms/**:

This folder contains the outputs of the slurm jobs that were run for training the models are available in the slurms folder:
Outputs 1-5 belong to Curriculum Learning run 1 (no weight decay during training)
Outputs 6-10 belong to Curriculum Learning run 2 (weight decay during training)
Outputs 11-15 belong to Curriculum Learning and Knowledge Distillation run 1 (no weight decay during training)
Outputs 16-20 belong to Curriculum Learning and Knowledge Distillation run 2 (weight decay during training)

### Resolutions for images and patches used in the curriculum

- **01. Resolutions for Images and Patches.xlsx**:

This is an Excel sheet with the choice of resolutions for images and patches that are used during Curriculum Learning.

### Durations of simulations

- **02. Durations for Simulations.xlsx**:

This is an Excel sheet with the durations of the simulations to indicate that training becomes more difficult during the curriculum.

### Efficiency of trained models

- **03. Evaluation of Efficiency on Validation Dataset.xlsx**:

This is an Excel sheet with the efficiency measurements for the different trained models.