# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_img.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_dia.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_img_noise.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_dia_noise.yaml

# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_img_train.yaml
# CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/predict_split_dia_train.yaml

CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora.yaml
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_dora-256.yaml
CUDA_VISIBLE_DEVICES=0 FORCE_TORCHRUN=1 llamafactory-cli train examples/www2025_eval/merge_lora.yaml

shutdown -h now
