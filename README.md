# Gemini Trojan Detection

This project provides a comprehensive suite of tools to simulate, detect, and mitigate **Neural Trojans** (backdoors) deeply embedded within Deep Neural Networks (DNNs). It implements the advanced attack and defense methodologies discussed in the article *"Neural Trojan Attacks and How You Can Help"*.

## Overview

A Neural Trojan is a maliciously injected behavior that causes a model to misclassify an input when a specific "trigger" (like a distinct pattern or shape) is present, while acting completely normally on clean data. This project explores both the offensive generation of these Trojans and the defensive strategies to detect and sanitize them.

## Features

### Supported Attacks
The repository can simulate various sophisticated Trojan attacks during the training phase on the CIFAR-10 dataset using a ResNet18 model:
* **Checkerboard & Square Triggers**: Standard visible geometric patches.
* **Blending Attack**: Steganography-like attacks that blend the trigger into the image using transparency (alpha blending).
* **Clean-Label Attack**: Sophisticated attacks where the poisoned images are completely imperceptible to humans and look identical to the target class, tricking the model without changing the ground-truth label during data inspection.
* **Dynamic Triggers**: Triggers whose position and rotation randomly change across the dataset, preventing simple deterministic pattern matching.
* **Weight Perturbation**: Directly altering the weights of specific convolutional filters to embed a Trojan without poisoning the actual dataset.

### Supported Defenses & Moderations
* **STRIP (Test-Time Detection)**: Data-based defense that superimposes incoming test images with varying patterns and measures the output entropy to detect anomalous, low-entropy predictions characteristic of triggers.
* **Spectral Signatures (Train-Time Detection)**: Data-based defense that evaluates the training data representations in the penultimate layer. It utilizes Singular Value Decomposition (SVD) to identify mathematical outlier "signatures" left by poisoned data and removes them.
* **Neural Cleanse (Model-Based Detection)**: Analyzes a finalized model to reverse-engineer potential triggers for every class, flagging classes that require unusually small triggers (which indicates a backdoor shortcut).
* **Fine-Pruning (Model Sanitization)**: Mitigates a suspected Trojan by profiling neuron activations on a clean dataset and iteratively pruning (zeroing out) the weights of the most "dormant" neurons, which the trigger relies on to activate.
* **Unlearning (Model Sanitization)**: Uses the reverse-engineered trigger from Neural Cleanse to briefly retrain the model on superimposed clean data with their *true* labels, effectively "unlearning" the malicious association.

## Installation

Ensure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/saitarrun/Trojan-Detection-using-Deep-Neural-Networks.git
   cd Trojan-Detection-using-Deep-Neural-Networks
   ```

2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### 1. Simulating Attacks (Training Poisoned Models)
Use the `train.py` script to generate models injected with specific triggers. Models will be saved in the `models/` directory.

Example: Train a model with a Blending attack targeting class 0:
```bash
python train.py --trigger-type blending --target-class 0 --epochs 10 --save-model models/blended_model.pth
```
*Note: A bash script `train_advanced_attacks.sh` is provided to generate multiple offensive models in sequence.*

### 2. Evaluating Defenses (Interactive UI)
The easiest way to evaluate the defenses (Neural Cleanse, STRIP, Spectral Signatures, and Fine-Pruning) is via the provided interactive Streamlit web application.

To launch the UI:
```bash
streamlit run app.py
```
This will open a dashboard in your browser where you can select a trained model checkpoint, set its attack parameters, and execute the defensive evaluations dynamically.

### 3. Evaluating Model Sanitization
To evaluate the deep sanitization techniques (Fine-Pruning metrics and Unlearning) in the terminal:
```bash
python sanitize_model.py --model-path models/poisoned_model.pth --target-class 0
```

## Structure
* `dataset.py`: CIFAR-10 data loading and runtime trigger injection (poisoning) logic.
* `models.py`: ResNet18 architecture definition.
* `train.py`: Main training loop for simulating attacks.
* `defenses.py`: Core defensive algorithms (Neural Cleanse, STRIP, Spectral Signatures, Fine-Pruning, Unlearning).
* `eval_defenses.py` / `sanitize_model.py`: Terminal evaluation scripts.
* `app.py`: Streamlit-based Interactive UI.

## Acknowledgements
Inspired by methodologies explored in advanced adversarial machine learning literature, specifically referencing implementations discussed for Neutral Cleanse, STRIP, and Spectral Signatures.
