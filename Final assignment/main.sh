wandb login

python3 train.py \
    --data-dir "./data/cityscapes" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "fastscnn-training-experiment-05" \
    --model-path "./models/fastscnn-training-experiment-04.pth" \