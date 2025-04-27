# 5LSM0: Neural Networks for Computer Vision

## Semantic Segmentation on Cityscapes using Fast-SCNN

We have used the Fast-SCNN model architecture for training semantic segmentation on the Cityscapes dataset.

The focus of the model architecture is to achieve a high efficiency and at the same time meet the minimum model performance.

This repository contains the modifications and training scripts that were used for training the models.

The trained models and slurm job outputs are provided as well.

There are several Excel sheets with calculations and measurements.

There is a folder with training loss and validation graphs that were captured from the Weights & Biases platform.

There is a folder with the original paper about the Fast-SCNN model architecture and a presentation about Fast-SCNN from BMVC 2019.

## Overview

We have modified and integrated the code of the Fast-SCNN model architecture to make it compatible with the existing training framework.

We have made 2 versions of the training script: perform training with Curriculum Learning or perform training with Curriculum Learning and Knowledge Distillation.

There are 2 different slurm job scripts to either launch the training based on Curriculum Learning or the training based on Curriculum Learning and Knowledge Distillation.

There is also a variant of Curriculum Learning with a Dice Loss component included in the Loss calculation.

There is also a variant of Curriculum Learning and Knowledge Distillation with a Dice Loss component included in the Loss calculation.

There is a separate Python tool that can be used to evaluate the efficiency of a trained model on the validation dataset.
This tool is performed locally on the computer, as we need to have access to the thop library which was not available on the super cluster.

The trained models are available in the models folder.

The outputs of the slurm jobs that were run for training the models are available in the slurms folder.

We have provided an Excel sheet with the choice of resolutions for images and patches that are used during Curriculum Learning.

There is an Excel sheet with the durations of the simulations to indicate that the training becomes more difficult when moving further through the curriculum.

There is an Excel sheet with the efficiency measurements for the different trained models.

There is a folder with training loss and validation graphs that were captured from the Weights & Biases platform.

There is a folder with the original paper about the Fast-SCNN model architecture and a presentation about Fast-SCNN from BMVC 2019.

### Fast-SCNN model architecture

Fast-SCNN is a lightweight and efficient model architecture designed for real-time semantic segmentation on edge devices.

- **fast_scnn.py**:

We could not find the official implementation of the Fast-SCNN model architecture from the researchers at Toshiba Research Europe and Cambridge University.

But we found an unofficial implementation on GitHub from an AI researcher at Baidu.

This implementation adopts all the main principles from the Fast-SCNN model architecture.

We had to make only slight changes to integrate this model in the existing training framework.

### Training with Curriculum Learning (Train a Teacher model)

The following files need to be modified before starting to train with Curriculum Learning:

- **main.sh**:

You have to update the new experiment id and provide the location of the previous trained Teacher model which was trained at a lower resolution.

The choice for the experiment id and the previous trained Teacher model depends on the stage in which you are during the curriculum.

If you start the training at the initial lowest resolution, then you have to specify "none" for the previous trained Teacher model.

- **main_dice.sh**:

There is also a variant of Curriculum Learning with a Dice Loss component included in the Loss calculation.

- **train.py**:

You have to make sure that the resized image dimensions and patch dimensions point to the resolution of the current curriculum:

resized_image_width, resized_image_height, patch_width, patch_height

You can choose between 5 profiles: from low resolution, to medium resolution, high resolution, higher resolution and highest resolution.

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each stage.

Each time you advance in the curriculum, you have to update the location of the previous trained Teacher model which was trained at a lower resolution.

- **train_dice.py**:

There is also a variant of Curriculum Learning with a Dice Loss component included in the Loss calculation.

### Training with Curriculum Learning and Knowledge Distillation (Train a Student model)

The following files need to be modified before starting to train with Curriculum Learning and Knowledge Distillation:

- **main_distillation.sh**:

You have to update the new experiment id and provide the location of the previous trained Student model which was trained at a lower resolution.

The choice for the experiment id and the previous trained Student model depends on the stage in which you are during the curriculum.

If you start the training at the initial lowest resolution, then you have to specify "none" for the previous trained Student model.

You have to provide the location of the trained Teacher model (who has learned the entire curriculum already).

The same trained Teacher model will be used during the entire curriculum of the Student.

- **main_distillation_dice.sh**:

There is also a variant of Curriculum Learning and Knowledge Distillation with a Dice Loss component included in the Loss calculation.

- **train_distillation.py**:

You have to make sure that the resized image dimensions and patch dimensions point to the resolution of the current curriculum:

resized_image_width, resized_image_height, patch_width, patch_height

You can choose between 5 profiles from low resolution, to medium resolution, high resolution, higher resolution and highest resolution.

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each stage.

Each time you advance in the curriculum, you have to update the location of the previous trained Student model which was trained at a lower resolution.

The same trained Teacher model will be used during the entire curriculum of the Student.

- **train_distillation_dice.py**:

There is also a variant of Curriculum Learning and Knowledge Distillation with a Dice Loss component included in the Loss calculation.

### Slurm job scripts to start training

There are different slurm job scripts to either train with Curriculum Learning or train with Curriculum Learning and Knowledge Distillation.

 **jobscript_slurm.sh**:

This is the slurm job script to launch the training with Curriculum Learning.

**jobscript_slurm_dice.sh**:

There is also a variant of Curriculum Learning with a Dice Loss component included in the Loss calculation.

 **jobscript_slurm_distillation.sh**:

This is the slurm job script to launch the training with Curriculum Learning and Knowledge Distillation.

**jobscript_slurm_distillation_dice.sh**:

There is also a variant of Curriculum Learning and Knowledge Distillation with a Dice Loss component included in the Loss calculation.

### Tool to evaluate the efficiency of a trained model on the validation dataset

- **main_efficiency.sh**:
- **evaluate_efficiency.py**:

You have to copy the trained model that you want to evaluate in the models folder to the file model.pth in the same folder.

You have to make sure that the resized image dimensions point to the resolution on which the model has been trained:

resized_image_width, resized_image_height

### Trained models

- **models/**:

This folder containes the trained models:

Models 1-5 belong to Curriculum Learning Run 1 (no weight decay during training)

Models 6-10 belong to Curriculum Learning Run 2 (weight decay during training)

Models 11-15 belong to Curriculum Learning Run 3 (with Knowledge Distillation) (no weight decay during training)

The trained Teacher model used for Knowledge Distillation belongs to Model 5 which corresponds to the Teacher that completed the entire curriculum (no weight decay during training)

Models 16-20 belong to Curriculum Learning Run 4 (with Knowledge Distillation) (weight decay during training)

The trained Teacher model used for Knowledge Distillation belongs to Model 10 which corresponds to the Teacher that completed the entire curriculum (weight decay during training)

Models 21-25 belong to Curriculum Learning Run 5 (with Dice Loss component) (no weight decay during training)

New trained models that will be available

Models 26-30 belong to Curriculum Learning Run 6 (with Dice Loss component) (weight decay during training)

Not available due to time constraints

Models 31-35 belong to Curriculum Learning Run 7 (with Knowledge Distillation) (with Dice Loss component) (no weight decay during training)

The trained Teacher model used for Knowledge Distillation belongs to Model 25 which corresponds to the Teacher that completed the entire curriculum (no weight decay during training)

New trained models that will be available

Models 36-40 belong to Curriculum Learning Run 8 (with Knowledge Distillation) (with Dice Loss component) (weight decay during training)

The trained Teacher model used for Knowledge Distillation belongs to Model 30 which corresponds to the Teacher that completed the entire curriculum (weight decay during training)

Not available due to time constraints

### Slurm job outputs

- **slurms/**:

This folder contains the outputs of the slurm jobs that were run for training the models:

Outputs 1-5 belong to Curriculum Learning Run 1 (no weight decay during training)

Outputs 6-10 belong to Curriculum Learning Run 2 (weight decay during training)

Outputs 11-15 belong to Curriculum Learning Run 3 (with Knowledge Distillation) (no weight decay during training)

Outputs 16-20 belong to Curriculum Learning Run 4 (with Knowledge Distillation) (weight decay during training)

Outputs 21-25 belong to Curriculum Learning Run 1 (with Dice Loss component) (no weight decay during training)

New trained models that will be available

Outputs 26-30 belong to Curriculum Learning Run 2 (with Dice Loss component) (weight decay during training)

Not available due to time constraints

Outputs 31-35 belong to Curriculum Learning Run 3 (with Knowledge Distillation) (with Dice Loss component) (no weight decay during training)

New trained models that will be available

Outputs 36-40 belong to Curriculum Learning Run 4 (with Knowledge Distillation) (with Dice Loss component) (weight decay during training)

Not available due to time constraints

### Measurements

- **measurements/**:

- **Resolutions for Images and Patches.xlsx**:

This is an Excel sheet with the choice of resolutions for images and patches that are used during Curriculum Learning.

- **Durations for Simulations.xlsx**:

This is an Excel sheet with the durations of the simulations to indicate that training becomes more difficult when moving further through the curriculum.

- **Evaluation of Efficiency on Validation Dataset.xlsx**:

This is an Excel sheet with the efficiency measurements for the different trained models.

### Training loss and validation graphs

- **graphs/**:

This folder contains training loss and validation graphs that were captured from the Weights & Biases platform.

### References

- **references/**:

This folder contains the original paper about the Fast-SCNN model architecture and a presentation about Fast-SCNN from BMVC 2019.