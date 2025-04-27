wandb login

python3 train_dice.py \
    --data-dir "./data/cityscapes" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "fastscnn-training-experiment-22" \
    --previous-model-path "./models/fastscnn-training-experiment-21.pth" \
    --dice-loss-weight 1.0 \