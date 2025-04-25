"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser
from tqdm import tqdm  # Import tqdm

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomCrop,
    ColorJitter
)

# Switch between neural network models
#from unet import Model
from fast_scnn import Model

# Define image sizes

# 576×288
resized_image_width1=576
resized_image_height1=288
patch_width1=448
patch_height1=224

# 768×384
resized_image_width2=768
resized_image_height2=384
patch_width2=576
patch_height2=288

# 1024×512
resized_image_width3=1024
resized_image_height3=512
patch_width3=768
patch_height3=384

# 1536x768
resized_image_width4=1536
resized_image_height4=768
patch_width4=1152
patch_height4=576

# 2048×1024
resized_image_width5=2048
resized_image_height5=1024
patch_width5=1536
patch_height5=768

####################################
# Select image size and patch size
####################################
resized_image_width=resized_image_width2
resized_image_height=resized_image_height2
patch_width=patch_width2
patch_height=patch_height2

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

    ce_loss = F.cross_entropy(student_logits, labels, ignore_index=255)
    total_loss = alpha * kd_loss + (1 - alpha) * ce_loss

    return total_loss, ce_loss, kd_loss

# Function to calculate mean and standard deviation
def calculate_mean_std(dataloader):
    mean = torch.zeros(3)  # For 3 channels (RGB)
    sum_squared = torch.zeros(3)  # To accumulate squared pixel values
    total_pixels = 0  # Total number of pixels in the dataset

    # Iterate through the data loader to accumulate values
    for images, _ in tqdm(dataloader, desc="Calculating Mean and Std", dynamic_ncols=True):
        b, c, h, w = images.shape  # Batch size, channels, height, width
        num_pixels = b * h * w  # Total pixels in this batch

        # Accumulate sum of pixel values for each channel
        mean += images.sum(dim=[0, 2, 3])
        # Accumulate sum of squared pixel values for each channel
        sum_squared += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += num_pixels  # Update total pixel count

    # Finalize the mean and std by dividing by the total number of pixels
    mean /= total_pixels
    std = torch.sqrt((sum_squared / total_pixels) - mean ** 2)  # Standard deviation formula

    return mean, std

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--trained-teacher-model-path", type=str, required=True, help="Path to available trained teacher model .pth file")
    parser.add_argument("--previous-student-model-path", type=str, required=True, help="Path to previous trained student model .pth file")
    parser.add_argument("--alpha", type=float, default=0.5, help="Distillation loss weight (0.0 to 1.0)")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for soft distillation")


    return parser


def main(args):
    
    # Check temperature setting for distillation
    assert args.temperature > 1.0, "Temperature should be greater than 1.0 for distillation!"

    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define the draft transform without normalization
    draft_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ])

    # Load the draft training dataset
    draft_train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=draft_transform
    )

    draft_train_dataset = wrap_dataset_for_transforms_v2(draft_train_dataset)

    draft_train_dataloader = DataLoader(
        draft_train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )

    # Calculate mean and std for the draft dataset
    mean, std = calculate_mean_std(draft_train_dataloader)
    print(f"Mean: {mean}, Std: {std}")

    # Define the transforms to apply to the data

    # Training transform (resize first, then random crop and jitter)
    train_transform = Compose([
        ToImage(),
        RandomHorizontalFlip(p=0.5),  # Random horizontal flip for augmentation
        Resize(size=(resized_image_height, resized_image_width)),  # Resize the image
        RandomCrop(size=(patch_height, patch_width)),  # Random crop to the desired patch size
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Photometric augmentations
        ToDtype(torch.float32, scale=True),  # Normalize pixel scale
        Normalize(mean.tolist(), std.tolist())  # Apply mean/std normalization
    ])
    
    # Validation transform (resize only)
    valid_transform = Compose([
        ToImage(),
        Resize((resized_image_height, resized_image_width)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean.tolist(), std.tolist()),
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=train_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=valid_transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )


    # Define the student model
    student_model = Model(
        in_channels=3,
        n_classes=19,
    ).to(device)

    # Load weights and biases from previous run if previous student model path is different from none
    if args.previous_student_model_path != "none":
        student_model.load_state_dict(torch.load(args.previous_student_model_path, map_location=device))
        print("Load weights and biases from previous student run")
    else:
        print("Start training from scratch with random weights and biases")


    # Load the teacher model
    teacher_model = Model(
        in_channels=3,
        n_classes=19,
    ).to(device)

    print("Load weights and biases from available trained teacher model")
    teacher_model.load_state_dict(torch.load(args.trained_teacher_model_path, map_location=device))
    teacher_model.eval()


    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(student_model.parameters(), lr=args.lr) # disable weight_decay=1e-4 for now

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        student_model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)    # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)   # Remove channel dimension

            optimizer.zero_grad()

            student_outputs = student_model(images)

            with torch.no_grad():
                teacher_outputs = teacher_model(images).detach()

            total_loss, ce_loss, kd_loss = distillation_loss(
                student_outputs, teacher_outputs, labels, T=args.temperature, alpha=args.alpha
            )

            total_loss.backward()
            optimizer.step()

            wandb.log({
                "train_total_loss": total_loss.item(),
                "train_ce_loss": ce_loss.item(),
                "train_kd_loss": kd_loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

            
        # Validation
        student_model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = student_model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_student_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(student_model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        student_model.state_dict(),
        os.path.join(
            output_dir,
            f"final_student_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
