
# 🦴 WristVision AI

**Explainable AI for Wrist Fracture Detection**

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

**WristVision AI** is a deep learning research project that detects wrist fractures and related findings from X-ray images. The system provides:

- ✅ **Multi-class detection** — Fractura (fractures), Metal (surgical hardware), Texto (text annotations)
- ✅ **Explainability** — Grad-CAM heatmaps highlight regions that influenced the prediction
- ✅ **AI-generated clinical descriptions** — Local LLM (Ollama) provides educational clinical summaries
- ✅ **Interactive dashboard** — Adjust thresholds, explore results, and understand model behavior

> ⚠️ **Disclaimer:** This is an educational research prototype. Not for clinical use.

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **🦴 Wrist X-Ray Analysis** | Upload an X-ray and get instant AI predictions |
| **🔥 Grad-CAM Heatmaps** | Visual explanations showing where the model "looks" |
| **🎚️ Adjustable Thresholds** | Control sensitivity vs specificity |
| **💬 Clinical Descriptions** | AI-generated summaries using Ollama (local LLM) |
| **📊 Model Performance** | ROC-AUC, sensitivity, specificity, and more |
| **📜 Analysis History** | Track all analyses in your session |

---
---
## 🚀 Live Demo

[![Launch App](https://img.shields.io/badge/Launch_App-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://wrist-fracture-detection-x5tspcfstp7oyvuipxzybh.streamlit.app/)
---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      WristVision AI                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   ResNet-50  │  │  Grad-CAM    │  │    Ollama LLM        │ │
│  │  Classifier  │→ │  Heatmaps    │  │  (Clinical Summary)  │ │
│  │  (3 Classes) │  │  (Explain)   │  │                      │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Outputs                                     │
│  • Prediction (Fractura/Metal/Texto)                         │
│  • Confidence Score                                           │
│  • Grad-CAM Heatmap                                           │
│  • Clinical Summary (Ollama)                                 │
│  • Downloadable Report                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Model Details

| Component | Detail |
|-----------|--------|
| **Architecture** | ResNet-50 (pre-trained on ImageNet, fine-tuned) |
| **Classes** | Fractura, Metal, Texto |
| **Input Size** | 224 × 224 pixels |
| **Framework** | PyTorch |
| **Explainability** | Grad-CAM |
| **LLM** | Ollama (llama3.2:3b) |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | DETECCION DE FRACTURAS |
| **Source** | Roboflow Universe |
| **Total Images** | 20,309 |
| **Splits** | Train: 14,216 / Valid: 4,063 / Test: 2,030 |
| **Classes** | Fractura (42.4%), Metal (1.9%), Texto (55.6%) |
| **License** | CC BY 4.0 |

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.6% |
| **ROC-AUC (Fractura)** | 0.971 |
| **Sensitivity** | 95.1% |
| **Specificity** | 92.5% |
| **Precision** | 91.3% |

### Per-Class F1-Scores

| Class | F1-Score |
|-------|----------|
| **Fractura** | 0.931 |
| **Metal** | 0.994 |
| **Texto** | 0.999 |

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) (for clinical descriptions)
- 16GB+ RAM recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/wrist-fracture-detection.git
cd wrist-fracture-detection
```

### Step 2: Set Up Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Place the DETECCION DE FRACTURAS dataset in `data/raw/`:

```
data/raw/
├── train/
├── valid/
└── test/
```

### Step 5: Download the Trained Model

Place `baseline_resnet50.pth` in `models/checkpoints/`

### Step 6: Install and Run Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

---

## 🚀 Usage

### Run the Streamlit Dashboard

```bash
source venv/bin/activate
python -m streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`

### Run the FastAPI Backend

```bash
python src/api/main.py
```

Open your browser at `http://localhost:8000`

### Run Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

---

## 📁 Project Structure

```
wrist-fracture-detection/
├── data/
│   └── raw/                        # Dataset (train, valid, test)
├── notebooks/                      # 6 Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_baseline.ipynb
│   ├── 03_gradcam_implementation.ipynb
│   ├── 04_medclip_integration.ipynb
│   ├── 05_treatment_rules.ipynb
│   └── 06_api_prototype.ipynb
├── src/
│   ├── data/
│   │   └── dataloader.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── gradcam.py
│   │   └── medclip.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── config.yaml
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── helpers.py
│   └── api/
│       ├── main.py
│       ├── endpoints/
│       │   ├── analyze.py
│       │   ├── diagnose.py
│       │   ├── explain.py
│       │   └── treatment.py
│       ├── models/
│       │   └── schemas.py
│       └── static/
│           └── index.html
├── tests/
│   ├── test_api.py
│   ├── test_classifier.py
│   └── test_utils.py
├── models/
│   └── checkpoints/
│       └── baseline_resnet50.pth
├── results/                         # Generated outputs (figures, predictions)
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
├── streamlit_app.py                 # Main dashboard
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Dataset analysis and visualization |
| `02_model_baseline.ipynb` | ResNet-50 training and evaluation |
| `03_gradcam_implementation.ipynb` | Grad-CAM heatmaps for explainability |
| `04_medclip_integration.ipynb` | Text explanations with CLIP/MedCLIP |
| `05_treatment_rules.ipynb` | Ollama clinical descriptions |
| `06_api_prototype.ipynb` | FastAPI prototype and testing |

---

## 🧪 Evaluation

### Confusion Matrix

| | Pred: Fractura | Pred: Metal | Pred: Texto |
|---|:---:|:---:|:---:|
| **True: Fractura** | 0.95 | 0.03 | 0.02 |
| **True: Metal** | 0.01 | 0.99 | 0.00 |
| **True: Texto** | 0.01 | 0.00 | 0.99 |

### ROC Curves

| Class | AUC |
|-------|:---:|
| Fractura | 0.971 |
| Metal | 1.000 |
| Texto | 0.976 |

---

## 🐳 Deployment

### Docker

```bash
docker build -t wristvision -f deployment/Dockerfile .
docker run -p 8501:8501 wristvision
```

### Docker Compose

```bash
docker-compose -f deployment/docker-compose.yml up
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Israel Murimiro**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: israel.murimiro@example.com

---

## 📚 References

- [DETECCION DE FRACTURAS Dataset](https://universe.roboflow.com/detecciondefracturasantebrazo-8tucq/deteccion-de-fracturas-bntcm)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Ollama](https://ollama.com/)

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis, treatment decisions, or patient care. Always consult qualified healthcare professionals.**

---

## 🎯 Acknowledgements

- University of Catania
- Roboflow Universe (dataset)
- Open-source community (PyTorch, Streamlit, Ollama, Grad-CAM)

---

Made with ❤️ by Israel Murimiro
```

