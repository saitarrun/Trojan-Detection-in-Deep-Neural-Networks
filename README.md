# 🛡️ Trojan Detection using Deep Neural Networks

An enterprise-grade, distributed MLOps platform for simulating, detecting, and mitigating **Neural Trojans** (backdoors) embedded within Deep Neural Networks (DNNs). Inspired by advanced adversarial machine learning literature, this project provides a comprehensive suite of tools for both offensive generation of Trojans and defensive forensic audits.

[![Model Audit CI](https://github.com/saitarrun/Trojan-Detection-using-Deep-Neural-Networks/actions/workflows/model-audit.yml/badge.svg)](https://github.com/saitarrun/Trojan-Detection-using-Deep-Neural-Networks/actions)

---

## 🌟 Overview

A Neural Trojan is a maliciously injected behavior that causes a model to misclassify an input when a specific "trigger" (like a distinct pattern or shape) is present, while acting completely normally on clean data. This project explores both the **offensive** generation of these Trojans and the **defensive** strategies to detect and sanitize them using a modern, microservices-oriented architecture.

Recent updates have transformed this project into a robust, industrial-scale detection system featuring a Next.js frontend, an asynchronous FastAPI/Celery backend, dynamic audit animations, a CI/CD pipeline, and a new **RiskMetaClassifier** for identifying novel, unknown threats.

---

## ✨ Features

### 💥 Supported Attacks (Offensive)
The repository can simulate various sophisticated Trojan attacks during the training phase using a ResNet18 model (e.g., on the CIFAR-10 / GTSRB datasets):
* **Checkerboard & Square Triggers**: Standard visible geometric patches.
* **Blending Attack**: Steganography-like attacks layering the trigger using transparency (alpha blending).
* **Clean-Label Attack**: Sophisticated attacks where poisoned images are imperceptible to humans and look identical to the target class, tricking the model without changing the ground-truth label.
* **Dynamic Triggers**: Triggers with randomized positions and rotations, preventing deterministic pattern matching.
* **Weight Perturbation**: Directly altering convolutional filter weights to embed a Trojan without dataset poisoning.

### 🛡️ Supported Defenses & Moderations (Defensive)
* **STRIP (Test-Time Detection)**: Superimposes test images with varying patterns and measures output entropy to detect anomalous, low-entropy predictions.
* **Spectral Signatures (Train-Time Detection)**: Evaluates representations in the penultimate layer using SVD to identify mathematical outlier "signatures" left by poisoned data.
* **Neural Cleanse (Model-Based Detection)**: Reverse-engineers potential triggers for every class to flag suspiciously small triggers indicating a backdoor.
* **RiskMetaClassifier (Advanced Fusion)**: A newly trained ML model that ingests telemetry data from various defense algorithms to identify complex, novel Trojan threats.
* **Fine-Pruning (Model Sanitization)**: Profiles neuron activations on clean data and iteratively prunes dormant neurons relied upon by the trigger.
* **Unlearning (Model Sanitization)**: Uses reverse-engineered triggers to briefly retrain and "unlearn" the malicious association using true labels.

### 🏗️ Enterprise Infrastructure
* **Next.js Real-time Dashboard**: A highly polished React frontend with dynamic, step-by-step animations visualizing the forensic audit process (Neural Cleanse, STRIP, Risk Fusion).
* **FastAPI + Celery + Redis**: Monolith-to-microservices asynchronous backend for scaling heavy GPU-bound forensic tasks.
* **Automated CI/CD**: Fully integrated GitHub Actions pipeline (`.github/workflows/model-audit.yml`) for continuous testing and automated quality checks.
* **Nautilus Cluster Remote Dev**: Built-in support (`setup_ssh.sh`) for remote SSH development on high-performance compute clusters.
* **Dockerized Setup**: Seamless, one-click local deployment using Docker Compose.

---

## 🚀 How to Execute

You have two options to run the Enterprise MLOps dashboard and backend services: **Docker Compose** (Recommended) or **Local Development**.

### Option A: Quickstart with Docker Compose (Recommended)

Ensure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saitarrun/Trojan-Detection-using-Deep-Neural-Networks.git
   cd Trojan-Detection-using-Deep-Neural-Networks
   ```

2. **Boot up the entire microservices stack:**
   ```bash
   docker-compose up --build
   ```
   *This single command will spin up the Redis message broker, the FastAPI Python backend, the Celery GPU worker, and the Next.js UI.*

3. **Access the Dashboard:**
   Navigate to [http://localhost:3000](http://localhost:3000) in your browser. The API operates at `http://localhost:8000`.

### Option B: Local Development Setup

If you prefer to run the services natively for development, ensure you have Python 3.8+, Node.js 20+, and Redis installed.

#### **1. Setup Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### **2. Start the Message Broker (Redis)**
Celery requires Redis to queue forensic tasks.
```bash
redis-server
```

#### **3. Start the CUDA GPU Worker (Celery)**
In a **new terminal** (ensure your venv is activated), start the background worker:
```bash
celery -A celery_worker worker --loglevel=info
```

#### **4. Start the Python API (FastAPI)**
In a **third terminal** (ensure your venv is activated), start the API server:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### **5. Launch the Enterprise UI Dashboard (Next.js)**
In a **fourth terminal**, navigate to the frontend directory and start the dev server:
```bash
cd frontend
npm install
npm run dev
```
Navigate to [http://localhost:3000](http://localhost:3000) to view the MLOps dashboard.

---

## 🧪 Terminal Operations & Evaluations

For researchers and developers, you can run detailed attacks and evaluations directly from the CLI.

### Simulating Attacks (Training Poisoned Models)
Use the `train.py` script to generate models injected with triggers. Models are saved in the `models/` directory.

```bash
# Example: Train a model with a Blending attack targeting class 0
python train.py --trigger-type blending --target-class 0 --epochs 10 --save-model models/blended_model.pth

# Alternative: Generate multiple offensive models in sequence
bash train_advanced_attacks.sh
```

### Training the RiskMetaClassifier
Train the meta-classifier using forensic telemetry data generated from various detection algorithms:
```bash
python train_meta_classifier.py
```

### Evaluating Model Sanitization
Test the defense mechanisms (Fine-Pruning and Unlearning) directly in the terminal:
```bash
python sanitize_model.py --model-path models/poisoned_model.pth --target-class 0
```

---

## 📂 Project Structure

* **`api.py` & `celery_worker.py`**: The FastAPI server and Celery worker definitions.
* **`frontend/`**: The Next.js 15 React application, featuring dynamic Step-by-Step interactive animations (`page.tsx`).
* **`dataset.py` & `gtsrb_dataset.py`**: Dataset loaders and runtime trigger injection (poisoning) logic.
* **`models.py`**: Deep Learning architecture definitions (e.g., ResNet18).
* **`train.py`**: Main training loop for simulating Trojan attacks.
* **`defenses.py`**: Core mathematical defensive algorithms (Neural Cleanse, STRIP, DB-SVD, Fine-Pruning, Unlearning).
* **`train_meta_classifier.py`**: Training script for the telemetry `RiskMetaClassifier`.
* **`eval_defenses.py` / `sanitize_model.py` / `evaluate_fusion_framework.py`**: Terminal-based evaluation scripts.
* **`.github/workflows/`**: Continuous Integration pipelines.
* **`setup_ssh.sh`**: Helper utility for establishing remote SSH bindings inside Nautilus clusters.

---

## Acknowledgements
Inspired by methodologies explored in advanced adversarial machine learning literature, specifically referencing fundamental implementations for *Neural Cleanse, STRIP*, and *Spectral Signatures*.
