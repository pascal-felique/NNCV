# 5LSM0: Neural Networks for Computer Vision

We have used the Fast-SCNN model architecture for training semantic segmentation on the Cityscapes dataset.

This repository contains the modifications and training scripts that were used for training the models.

## Overview

We have modified and integrated the code of the Fast-SCNN model architecture to make it compatible with the existing training script.

We have made 2 versions of the training script: perform training with Curriculum Learning or perform training with Curriculum Learning and Knowledge Distillation.

There are 2 different slurm job scripts to either launch the training based on Curriculum Learning or the training based on Curriculum Learning and Knowledge Distillation.

There is a separate Python tool that can be used to evaluate the efficiency of a trained model on the validation dataset.

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

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each training run.

And each time, you have to update the location of the previous trained Teacher model at a lower resolution.

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

To follow the curriculum correctly during Curriculum Learning, you have to increase the resolution in each training run.

And each time, you have update the location of the previous trained Student model at a lower resolution.

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