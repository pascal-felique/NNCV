wandb login

python3 train_unet.py \
    --data-dir "./data/cityscapes" \
    --batch-size 16 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "unet-training-experiment-44" \
    --previous-model-path "./models/unet-training-experiment-43.pth" \