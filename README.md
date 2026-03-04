# 🛡️ Trojan Detection in Deep Neural Networks
**Enterprise-grade, distributed MLOps platform** to **simulate, detect, and mitigate Neural Trojans (backdoors)** in DNNs—covering both **offensive generation** and **defensive forensic audits** with an end-to-end microservices architecture.

[![Model Audit CI](https://github.com/saitarrun/Trojan-Detection-in-Deep-Neural-Networks/actions/workflows/model-audit.yml/badge.svg)](https://github.com/saitarrun/Trojan-Detection-in-Deep-Neural-Networks/actions/workflows/model-audit.yml)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Node](https://img.shields.io/badge/Node-20%2B-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-informational) <!-- update if different -->

> **What you get:** a reproducible pipeline to generate trojaned models, run multiple detectors (STRIP / Spectral / Neural Cleanse), fuse evidence with a **RiskMetaClassifier**, and optionally **sanitize** the model (fine-pruning / unlearning) with an audit trail.

---

## ⭐ Why this repo is worth starring
- **End-to-end system:** Attack simulation → Detection → Evidence fusion → Sanitization → Audit artifacts
- **Production-style stack:** Next.js dashboard + FastAPI API + Celery GPU worker + Redis broker
- **Forensic visibility:** Step-by-step UI animations + telemetry + dashboards
- **Research-ready:** CLI scripts + reproducible experiments + CI-based model audit workflow

---

## 📌 Table of Contents
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Features](#-features)
- [5-minute Quickstart (Docker Compose)](#-5-minute-quickstart-docker-compose)
- [Local Development](#-local-development)
- [CLI: Attacks, Defenses, Sanitization](#-cli-attacks-defenses-sanitization)
- [Project Structure](#-project-structure)
- [Results and Reproducibility](#-results-and-reproducibility)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)


## 🧩 Architecture
**UI (Next.js)** → **API (FastAPI)** → **Task Queue (Celery + Redis)** → **GPU/CPU Forensic Jobs** → **Telemetry + Audit Artifacts**

- Next.js dashboard visualizes detection steps, risk scores, and artifacts
- FastAPI provides orchestration endpoints for audits
- Celery workers run compute-heavy defense algorithms asynchronously
- Redis acts as message broker for task scheduling

---

## ✨ Features

### 💥 Offensive (Attack Simulation)
Simulate Trojan attacks during training (e.g., ResNet18 on CIFAR-10 / GTSRB):
- **Checkerboard & Square triggers**
- **Blending attack** (alpha blending / steganography-like)
- **Clean-label attack** (poisoned samples appear benign)
- **Dynamic triggers** (randomized positions/rotations)
- **Weight perturbation** (direct parameter manipulation)

### 🛡️ Defensive (Detection + Moderation)
- **STRIP (test-time):** entropy-based anomaly detection via superposition
- **Spectral Signatures (train-time):** SVD-based representation outliers
- **Neural Cleanse (model-based):** reverse-engineer class-wise triggers
- **RiskMetaClassifier (fusion):** ML model over multi-signal telemetry
- **Fine-Pruning (sanitization):** prune neurons tied to backdoor behavior
- **Unlearning (sanitization):** retrain to remove malicious association

### 🏗️ Platform / Infrastructure
- **Next.js real-time dashboard** with step-by-step audit animations
- **FastAPI + Celery + Redis** asynchronous microservices backend
- **GitHub Actions CI** for continuous checks and model audit workflow
- **Docker Compose** one-command local deployment
- **Remote dev support** (`setup_ssh.sh`) for Nautilus/HPC workflows

---

## 🚀 5-minute Quickstart (Docker Compose)
**Recommended** for most users.

### Prerequisites
- Docker + Docker Compose

### Run
bash
git clone https://github.com/saitarrun/Trojan-Detection-in-Deep-Neural-Networks.git
cd Trojan-Detection-in-Deep-Neural-Networks
docker-compose up --build 

Open
	•	UI: http://localhost:3000
	•	API: http://localhost:8000

Tip: add -d to run detached: docker-compose up --build -d

⸻

🧑‍💻 Local Development

Prereqs: Python 3.8+, Node 20+, Redis

1) Python environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

2) Start Redis

redis-server

3) Start Celery worker

celery -A celery_worker worker --loglevel=info

4) Start FastAPI

uvicorn api:app --host 0.0.0.0 --port 8000

5) Start Next.js dashboard

cd frontend
npm install
npm run dev

Open http://localhost:3000

⸻

🧪 CLI: Attacks, Defenses, Sanitization

Simulate attacks (train poisoned models)

# Example: Blending attack targeting class 0
python train.py --trigger-type blending --target-class 0 --epochs 10 --save-model models/blended_model.pth

# Generate multiple offensive models
bash train_advanced_attacks.sh

Train RiskMetaClassifier

python train_meta_classifier.py

Evaluate / run defenses (terminal)

python eval_defenses.py --model-path models/blended_model.pth

Sanitize model (Fine-Pruning / Unlearning)

python sanitize_model.py --model-path models/poisoned_model.pth --target-class 0


⸻

📂 Project Structure
	•	api.py, celery_worker.py — FastAPI server and Celery worker
	•	frontend/ — Next.js dashboard (animations in page.tsx)
	•	dataset.py, gtsrb_dataset.py — datasets + trigger injection/poisoning logic
	•	models.py — architectures (e.g., ResNet18)
	•	train.py — training loop for attack simulation
	•	defenses.py — Neural Cleanse, STRIP, Spectral, Fine-Pruning, Unlearning
	•	train_meta_classifier.py — telemetry fusion model training
	•	eval_defenses.py, sanitize_model.py, evaluate_fusion_framework.py — evaluation/sanitization utilities
	•	.github/workflows/ — CI pipelines
	•	setup_ssh.sh — remote SSH helper (Nautilus/HPC)

⸻

📊 Results and Reproducibility

To make results comparable and repeatable:
	•	Set random seeds (Python/NumPy/PyTorch) where applicable
	•	Document dataset versions and preprocessing
	•	Store artifacts (telemetry, recovered triggers, audit reports) under a consistent output directory

Recommended: add a results/ folder and a short table like:
	•	Attack type → ASR / Clean Acc → Detector flags → Fusion score → Sanitization outcome

⸻

🤝 Contributing

Contributions are welcome:
	•	bug fixes, docs, detectors, new attack modules, UI improvements
	•	please open an issue first for larger changes

Suggested repo hygiene (recommended for reach):
	•	CONTRIBUTING.md
	•	issue templates
	•	good first issue labels

⸻

📌 Citation

If you use this repo in academic work, please cite:

@misc{pitta_trojan_detection_2026,
  title        = {Trojan Detection in Deep Neural Networks: A Distributed MLOps Platform},
  author       = {Pitta, Sai Tarrun},
  year         = {2026},
  howpublished = {\url{https://github.com/saitarrun/Trojan-Detection-in-Deep-Neural-Networks}}
}


⸻

🙏 Acknowledgements

Inspired by adversarial ML and backdoor defense literature, including foundational methodologies for Neural Cleanse, STRIP, and Spectral Signatures.

If you want, paste your current repo file tree (or confirm whether you already have `docs/demo.gif`, `LICENSE`, `CONTRIBUTING.md`) and I’ll tailor the README links/sections exactly to what exists in your repository.
