wandb login

python3 train_distillation.py \
    --data-dir "./data/cityscapes" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 8 \
    --seed 42 \
    --experiment-id "fastscnn-training-experiment-99" \
    --trained-teacher-model-path "./models/fastscnn-training-experiment-05.pth" \
    --previous-student-model-path "none" \
    --alpha 0.5 \
    --temperature 4.0 \