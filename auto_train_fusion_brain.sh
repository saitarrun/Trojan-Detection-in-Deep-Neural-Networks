#!/bin/bash

# Gemini: Fusion Brain Auto-Training Pipeline (TrojAI Edition)
# This script automates the entire loop for high-res model security auditing.

echo "=========================================================="
echo "🚀 Starting Gemini Fusion Brain Orchestration"
echo "=========================================================="

# 1. Download Architectural Baselines
echo -e "\n[Step 1/3] Fetching high-res architectures inside Docker..."
docker compose exec worker python3 get_sample_models.py

# 2. Simulate TrojAI-Style Attacks
echo -e "\n[Step 2/3] Simulating Polygon and Filter attacks inside Docker..."
docker compose exec worker python3 simulate_trojai_attacks.py

# 3. Perform Final Meta-Classifier Training
echo -e "\n[Step 3/3] Auditing model zoo and training the Fusion Brain..."
mkdir -p training_zoo
cp models/*.pth training_zoo/ 2>/dev/null
cp sample_external_models/*.pt training_zoo/ 2>/dev/null
cp sample_trojai_models/*.pt training_zoo/ 2>/dev/null

docker compose exec worker python3 train_meta_classifier.py --mode both --model-dir training_zoo

echo -e "\n=========================================================="
echo "✅ Fusion Brain Training Complete!"
echo "Your meta_classifier.pkl is now trained on a diverse TrojAI model zoo."
echo "=========================================================="
