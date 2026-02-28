#!/bin/bash
set -e

source venv/bin/activate

# 1. Train Blended Trigger Model
echo -e "\n=== Training Blended Trigger Model ==="
python train.py --epochs 10 --trigger-type blending --poisoned-model-path models/blended_model.pth

# 2. Train Clean-Label Model
echo -e "\n=== Training Clean-Label Model ==="
python train.py --epochs 10 --trigger-type clean_label --poisoned-model-path models/clean_label_model.pth

# 3. Train Dynamic Trigger Model
echo -e "\n=== Training Dynamic Trigger Model ==="
python train.py --epochs 10 --trigger-type dynamic --poisoned-model-path models/dynamic_model.pth

echo -e "\n=== All Attacks Trained Successfully ==="
