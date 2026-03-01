#!/bin/bash

DEST="test_uploads"
mkdir -p $DEST

echo "=== Organizing Master Test Suite (9 Curated Models) ==="

# 1. Clean Baseline (CIFAR-10)
cp models/clean_resnet20_cifar.pt $DEST/01_clean_baseline_resnet20.pt 2>/dev/null || cp sample_external_models/clean_resnet20_cifar.pt $DEST/01_clean_baseline_resnet20.pt 2>/dev/null

# 2. Classic BadNet (CIFAR-10)
cp models/poisoned_model.pth $DEST/02_poisoned_badnet_checkerboard.pth 2>/dev/null

# 3. Blending Attack (Subtle Trigger)
cp models/blended_model.pth $DEST/03_poisoned_blending_attack.pth 2>/dev/null

# 4. Dynamic Trigger (Input Dependent)
cp models/dynamic_model.pth $DEST/04_poisoned_dynamic_trigger.pth 2>/dev/null

# 5. High-Accuracy Clean (ResNet20)
cp sample_external_models/clean_resnet20_cifar.pt $DEST/05_high_accuracy_clean.pt 2>/dev/null

# 6. TrojAI Polygon (High-Res 224x224)
cp sample_trojai_models/poisoned_densenet_polygon.pt $DEST/06_trojai_poisoned_polygon.pt 2>/dev/null

# 7. TrojAI Filter (High-Res 299x299)
cp sample_trojai_models/poisoned_inception_sepia.pt $DEST/07_trojai_poisoned_filter.pt 2>/dev/null

# 8. Mystery Industry Model (VGG16 - Clean)
cp real_world_mystery_set/real_world_unknown_5722.pth $DEST/08_mystery_industry_vgg16_clean.pth 2>/dev/null

# 9. Mystery Industry Model (MobileNetV2 - Poisoned)
cp real_world_mystery_set/real_world_unknown_9315.pth $DEST/09_mystery_industry_mobilenet_poisoned.pth 2>/dev/null

echo "Selection organized in '$DEST/' directory."
ls -lh $DEST/
