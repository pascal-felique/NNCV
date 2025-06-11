wandb login

python3 train_unet.py \
    --data-dir "./data/cityscapes" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "unet-training-experiment-42" \
    --previous-model-path "./models/unet-training-experiment-41.pth" \