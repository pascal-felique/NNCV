import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToImage, ToDtype
from argparse import ArgumentParser
from tqdm import tqdm
from thop import profile
import numpy as np

# Import your model
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
resized_image_width=resized_image_width1
resized_image_height=resized_image_height1
patch_width=patch_width1
patch_height=patch_height1

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mean Dice score over dataset
def compute_mean_dice_over_dataset(preds, targets, num_classes=19, ignore_index=255):
    total_inter = torch.zeros(num_classes)
    total_union = torch.zeros(num_classes)

    for pred, target in zip(preds, targets):
        mask = (target != ignore_index)
        pred = pred[mask]
        target = target[mask]

        for cls in range(num_classes):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            inter = (pred_inds & target_inds).sum().item()
            union = pred_inds.sum().item() + target_inds.sum().item()
            total_inter[cls] += inter
            total_union[cls] += union

    dice_scores = (2.0 * total_inter) / (total_union + 1e-6)  # avoid division by zero
    valid = total_union > 0
    return dice_scores[valid].mean().item() if valid.any() else 0.0


def get_args_parser():
    parser = ArgumentParser("Inference and Efficiency Eval")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--current-model-path", type=str, required=True, help="Path to trained model .pth file")
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validation transform
    valid_transform = Compose([
        ToImage(),
        Resize((resized_image_height, resized_image_width)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.2855, 0.3228, 0.2819], std=[0.1831, 0.1864, 0.1837]), # Pre-calculated values
    ])

    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=valid_transform)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Load model
    model = Model(in_channels=3, n_classes=19).to(device)
    model.load_state_dict(torch.load(args.current_model_path, map_location=device))
    model.eval()

    # 5. Calculate FLOPS on one image
    dummy_input = torch.randn(1, 3, resized_image_height, resized_image_width).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"FLOPs per image: {flops:.2e}")

    # 6. Inference and Dice score
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Running inference"):
            images = images.to(device)
            targets = convert_to_train_id(targets).long().squeeze(1).to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu())
            all_targets.extend(targets.cpu())

    mean_dice = compute_mean_dice_over_dataset(all_preds, all_targets)

    efficiency = mean_dice / flops
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Efficiency (Dice / FLOPs): {efficiency:.6e}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
