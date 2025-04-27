wandb login

python3 train_dice.py \
    --data-dir "./data/cityscapes" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "fastscnn-training-experiment-25" \
    --previous-model-path "./models/fastscnn-training-experiment-24.pth" \
    --dice-loss-weight 1.0 \