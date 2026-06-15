#!/bin/bash

ARCH=$1
EPS=$2
STEPS=$3

echo "Running grid job:"
echo "ARCH=$ARCH"
echo "EPS=$EPS"
echo "STEPS=$STEPS"

mkdir -p data src checkpoints results

if [ -f train.npz ]; then mv train.npz data/train.npz; fi
if [ -f train_adv_grid.py ]; then mv train_adv_grid.py src/train_adv_grid.py; fi

echo "Checking GPU:"
nvidia-smi || true

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python src/train_adv_grid.py \
  --arch "$ARCH" \
  --eps_int "$EPS" \
  --pgd_steps "$STEPS" \
  --epochs 90 \
  --batch_size 128 \
  --data_path data/train.npz \
  --checkpoint_dir checkpoints \
  --results_dir results

echo "Job finished."
ls -lh checkpoints
ls -lh results
