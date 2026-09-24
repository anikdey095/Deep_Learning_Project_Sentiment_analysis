# Moodline — Deep Learning Sentiment & Emotion Intelligence

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.0+-D00000.svg?style=flat&logo=Keras&logoColor=white)](https://keras.io)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=flat&logo=Docker&logoColor=white)](https://www.docker.com)
[![Render](https://img.shields.io/badge/Render-Live%20Demo-46E3B7.svg?style=flat&logo=Render&logoColor=white)](https://moodline-sentiment.onrender.com/)
[![Status](https://img.shields.io/website?url=https%3A%2F%2Fmoodline-sentiment.onrender.com%2Fhealth&label=Service%20Status)](https://moodline-sentiment.onrender.com/)

**Moodline** is a production-ready, end-to-end Deep Learning and Natural Language Processing (NLP) system designed to detect subtle emotional states embedded within human language. Built on a Bidirectional Gated Recurrent Unit (**BiGRU**) architecture with 300-dimensional embeddings, Moodline categorizes input text into 6 discrete emotion dimensions: **Joy**, **Sadness**, **Love**, **Anger**, **Fear**, and **Surprise**.

Predictions are served in real time via an asynchronous **FastAPI** backend coupled with a modern, glassmorphic dark-mode web console featuring dynamic ambient emotion lighting, live probability spectrum breakdowns, telemetry benchmarking, and optional Web Audio synthesis.

---

## 🚀 Live Web Application

- **Live Application:** [https://moodline-sentiment.onrender.com/](https://moodline-sentiment.onrender.com/)
- **Interactive Swagger Docs:** [https://moodline-sentiment.onrender.com/docs](https://moodline-sentiment.onrender.com/docs)
- **ReDoc Documentation:** [https://moodline-sentiment.onrender.com/redoc](https://moodline-sentiment.onrender.com/redoc)
- **Health Check Endpoint:** [https://moodline-sentiment.onrender.com/health](https://moodline-sentiment.onrender.com/health)

> [!NOTE]
> Since this project is hosted on Render's free tier, the web service may spin down after periods of inactivity. If accessing after idle time, please allow 30–50 seconds for the initial cold start and neural graph warm-up.

---

## Architecture & Neural Network Design

### Model Pipeline

```
Raw Text Input
      │
      ▼
Text Preprocessing (Lowercasing, Regex Cleaning, Contraction Normalization)
      │
      ▼
Tokenization & Sequence Padding (Max Sequence Length: 50, Vocab: ~15,000 words)
      │
      ▼
Embedding Layer (Input Dim: 10,000, Output Dim: 300)
      │
      ▼
Bidirectional GRU Layer (128 units, Return Sequences = True)
      │
      ▼
Spatial Dropout (0.5)
      │
      ▼
Bidirectional GRU Layer (64 units)
      │
      ▼
Dropout (0.5)
      │
      ▼
Dense Softmax Output (6 Emotion Classes)
```

### Emotion Classes

| Emotion | Indicator Emoji | Theme Color | Representative Phrase |
| :--- | :---: | :---: | :--- |
| **Joy** | 😄 | `#f59e0b` (Amber Gold) | *"I just got my dream job and I am overjoyed!"* |
| **Love** | ❤️ | `#f43f5e` (Passionate Rose) | *"I cherish every moment spent with you."* |
| **Sadness** | 😢 | `#3b82f6` (Cobalt Blue) | *"I feel an overwhelming emptiness and grief."* |
| **Anger** | 😠 | `#ef4444` (Fiery Crimson) | *"I am furious that they deliberately lied to me!"* |
| **Fear** | 😨 | `#a855f7` (Mystic Violet) | *"A cold dread seized me in the dark alley."* |
| **Surprise** | 😲 | `#06b6d4` (Electric Cyan) | *"I was completely speechless when everyone jumped out!"* |

### Comparative Architecture Benchmarks

During research and development (see [`sentiment_analysis.ipynb`](sentiment_analysis.ipynb)), four sequence modeling architectures were trained on the Hugging Face `dair-ai/emotion` benchmark:

| Architecture | Parameters | Strengths & Characteristics | Test Accuracy |
| :--- | :--- | :--- | :---: |
| **SimpleRNN** | ~1.3M | Baseline recurrent network; struggles with long-range dependencies | ~78.4% |
| **LSTM** | ~1.6M | Mitigates vanishing gradient via input, forget, and output gates | ~86.2% |
| **GRU** | ~1.5M | Streamlined gating mechanism; faster convergence than LSTM | ~87.8% |
| **BiGRU** *(Selected)* | ~2.1M | **Processes context in both forward and backward directions; captures rich affective nuance** | **~90.1%** |

---

## Features

- **BiGRU Inference Engine:** 300D dense word representations with bidirectional sequence context.
- **Resilient Deserialization:** Automatic compatibility layer handling Keras 3 cross-version layer serialization (such as `quantization_config`).
- **Modern Cyber-Editorial UI:** Glassmorphism, dynamic ambient glow reacting to predicted emotions, and accessible typography (Outfit & Plus Jakarta Sans).
- **Interactive Quick-Test Chips:** Instant one-click testing for each of the 6 emotional states.
- **Full Spectrum Distribution:** Interactive percentage breakdown across all classes with animated neon tracks.
- **Real-Time Telemetry:** Latency benchmark pill (`latency_ms`), word & character counters.
- **Session History:** Session memory allowing 1-click retrieval of recent predictions.
- **Audio Synthesizer:** Subtle Web Audio harmonic chimes on classification reveal (with mute switch).
- **Production Containerization:** Dockerfile with non-root security user, multi-stage caching, and built-in healthchecks.
- **Automated Test Suite:** Complete end-to-end integration tests (`test_app.py`).

---

## Project Structure

```
Deep_Learning_Project_Sentiment_analysis/
├── Artifacts/
│   ├── BiGRU_Model.keras       # Trained Bidirectional GRU model weights
│   ├── BiGRU_Modle.keras       # Fallback alias for backward compatibility
│   └── tokenizer.pkl           # Fitted Keras text tokenizer
├── static/
│   ├── index.html              # Frontend console UI
│   ├── style.css               # Design tokens, glassmorphism & ambient aura
│   └── script.js               # Reactive frontend logic & audio synthesis
├── main.py                     # FastAPI application & inference pipeline
├── test_app.py                 # Automated integration test suite
├── Dockerfile                  # Container definition for production
├── .dockerignore               # Build optimization exclusions
├── render.yaml                 # Render cloud deployment blueprint
├── Procfile                    # PaaS process file (Render / Railway / Heroku)
├── requirements.txt            # Locked Python dependencies
├── sentiment_analysis.ipynb    # Training notebook (data prep, model training)
└── README.md                   # Comprehensive documentation
```

---

## Quickstart & Local Setup

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/anikdey095/Deep_Learning_Project_Sentiment_analysis.git
cd Deep_Learning_Project_Sentiment_analysis

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python test_app.py
```
Expected output:
```
Starting Moodline test suite...
Test 1: Testing /health endpoint...
  -> Passed! Model loaded and classes verified.
Test 2: Testing /predict with joyful sentence...
  -> Passed! Predicted: joy (99.9%) in 42.1ms
Test 3: Testing /predict with sad sentence...
  -> Passed! Predicted: sadness (99.9%) in 38.4ms
Test 4: Testing validation handling for blank text...
  -> Passed! Invalid input cleanly rejected.
Test 5: Testing GET / endpoint...
  -> Passed! Static index.html served successfully.

All 5 automated tests passed with flying colors! [SUCCESS]
```

### 4. Start the Application
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
- **Web Application:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Docker Deployment

### Build the Image
```bash
docker build -t moodline-sentiment:latest .
```

### Run Container
```bash
docker run -d -p 8000:8000 --name moodline moodline-sentiment:latest
```

Verify container status:
```bash
curl http://localhost:8000/health
```

---

## Cloud Deployment (Render / Railway)

### Live Instance
- **Production URL:** [https://moodline-sentiment.onrender.com/](https://moodline-sentiment.onrender.com/)

### 1-Click Render Deployment
This repository includes a preconfigured `render.yaml` blueprint:
1. Connect your repository to [Render](https://render.com).
2. Choose **New Blueprint Instance**.
3. Select `render.yaml`. Render will automatically build the container and deploy the service.

Alternatively, create a **Web Service**:
- **Environment:** Python 3.11
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`

---

## API Reference

### 1. Health Check
```http
GET /health
```
**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 124.52,
  "model_name": "BiGRU_Model.keras",
  "classes": ["sadness", "joy", "love", "anger", "fear", "surprise"],
  "vocab_size": 15213
}
```

### 2. Predict Emotion
```http
POST /predict
Content-Type: application/json

{
  "text": "I feel so grateful and completely overjoyed about this milestone!"
}
```

**Response (`200 OK`):**
```json
{
  "text": "I feel so grateful and completely overjoyed about this milestone!",
  "predicted_emotion": "joy",
  "emoji": "😄",
  "confidence": 0.998412,
  "all_probabilities": {
    "sadness": 0.000082,
    "joy": 0.998412,
    "love": 0.000921,
    "anger": 0.000406,
    "fear": 0.000070,
    "surprise": 0.000109
  },
  "all_probabilites": { ... },
  "latency_ms": 28.4
}
```

---

## License

This project is licensed under the MIT License.
