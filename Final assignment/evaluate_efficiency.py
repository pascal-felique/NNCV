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

# Define image resolutions
# 768×384
image_width1=768
image_height1=384

# 576×288
image_width2=576
image_height2=288

# 512×256
image_width3=512
image_height3=256

# 480×240
image_width4=480
image_height4=240

# 384×192
image_width5=384
image_height5=192

###########################
# Select image resolution
###########################
image_width=image_width1
image_height=image_height1

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Dice score function
def dice_score(pred, target, num_classes=19, ignore_index=255):
    dice = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item()
        if union == 0:
            continue
        dice.append((2. * intersection) / union)
    return np.mean(dice) if dice else 0.0

def get_args_parser():
    parser = ArgumentParser("Inference and Efficiency Eval")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes dataset")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pth file")
    return parser

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Draft transform to calculate normalization
    draft_transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ])
    draft_dataset = Cityscapes(args.data_dir, split="train", mode="fine", target_type="semantic", transforms=draft_transform)
    draft_dataset = wrap_dataset_for_transforms_v2(draft_dataset)
    draft_loader = DataLoader(draft_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Calculate mean and std
    mean = torch.zeros(3)
    sum_squared = torch.zeros(3)
    total_samples = 0
    for images, _ in tqdm(draft_loader, desc="Calc mean/std"):
        batch_size = images.size(0)
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        sum_squared += (images ** 2).mean(dim=[0, 2, 3]) * batch_size
        total_samples += batch_size
    mean /= total_samples
    std = torch.sqrt((sum_squared / total_samples) - mean.pow(2))

    # 3. Final transform
    valid_transform = Compose([
        ToImage(),
        Resize((image_width, image_height)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean.tolist(), std.tolist()),
    ])

    valid_dataset = Cityscapes(args.data_dir, split="val", mode="fine", target_type="semantic", transforms=valid_transform)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Load model
    model = Model(in_channels=3, n_classes=19).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 5. Calculate FLOPS on one image
    dummy_input = torch.randn(1, 3, image_height, image_width).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))
    print(f"FLOPs per image: {flops:.2e}")

    # 6. Inference and Dice score
    dice_scores = []
    with torch.no_grad():
        for images, targets in tqdm(valid_loader, desc="Running inference"):
            images = images.to(device)
            targets = convert_to_train_id(targets).long().squeeze(1).to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for p, t in zip(preds, targets):
                dice_scores.append(dice_score(p.cpu(), t.cpu()))

    mean_dice = np.mean(dice_scores)
    efficiency = mean_dice / flops
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Efficiency (Dice / FLOPs): {efficiency:.6e}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
