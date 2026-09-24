import logging
import os

# Silence TensorFlow verbose logs & disable oneDNN noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("moodline.api")

# --- Keras Deserialization Compatibility Layer ---
# Ensures backwards & forward compatibility across Keras 3.x minor versions
# (e.g. models serialized with quantization_config in newer Keras)
try:
    import keras
    if hasattr(keras.layers, "Layer") and hasattr(keras.layers.Layer, "from_config"):
        orig_from_config = keras.layers.Layer.from_config.__func__

        @classmethod
        def safe_layer_from_config(cls, config):
            cfg = dict(config)
            cfg.pop("quantization_config", None)
            return orig_from_config(cls, cfg)

        keras.layers.Layer.from_config = safe_layer_from_config
        logger.info("Applied Keras Layer deserializer compatibility patch.")
except Exception as patch_err:
    logger.warning(f"Keras deserializer patch skipped or unnecessary: {patch_err}")

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Directories and File Resolution ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"
STATIC_DIR = BASE_DIR / "static"

# Handle primary and fallback model filenames
candidate_model_path = ARTIFACTS_DIR / "BiGRU_Model.keras"
if not candidate_model_path.exists():
    candidate_model_path = ARTIFACTS_DIR / "BiGRU_Modle.keras"

MODEL_PATH = candidate_model_path
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"

MAX_SEQUENCE_LENGTH = 50
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

# In-memory storage for loaded deep learning model and tokenizer
dl_resources = {}
SERVER_START_TIME = time.time()


def preprocess_text(text: str) -> str:
    """
    Cleans raw text to match the training pipeline:
    1. Lowercase
    2. Remove apostrophes (can't -> cant)
    3. Replace non-alphanumeric chars with spaces
    4. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Schemas ---
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence or text passage to classify for emotion.",
        json_schema_extra={"example": "I feel so happy and excited about this new chapter!"}
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Input text cannot be blank or whitespace-only.")
        return trimmed


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    all_probabilities: Dict[str, float]
    all_probabilites: Dict[str, float]  # Alias for backwards compatibility
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    model_name: str
    classes: List[str]
    vocab_size: int


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Moodline DL Model and Tokenizer...")
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found at {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        raise RuntimeError(f"Tokenizer artifact not found at {TOKENIZER_PATH}")

    try:
        model = load_model(str(MODEL_PATH))
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)

        dl_resources["BiGRU"] = model
        dl_resources["Tokenizer"] = tokenizer

        # Warm-up graph execution so first user request doesn't experience compilation latency
        try:
            dummy_seq = np.zeros((1, MAX_SEQUENCE_LENGTH), dtype=np.int32)
            _ = model(dummy_seq, training=False)
            logger.info("Graph warm-up execution completed.")
        except Exception as warmup_err:
            logger.warning(f"Warm-up pass skipped: {warmup_err}")

        logger.info(f"Model ({MODEL_PATH.name}) & Tokenizer loaded successfully. Vocab size: {len(tokenizer.word_index)}")
    except Exception as exc:
        logger.error(f"Failed to load model artifacts: {exc}", exc_info=True)
        raise exc

    yield

    logger.info("Shutting down and clearing DL memory resources...")
    dl_resources.clear()


# --- Application Setup ---
app = FastAPI(
    title="Moodline — Deep Learning Emotion Analysis",
    description="Production-ready FastAPI backend serving a Bidirectional GRU (BiGRU) deep neural network trained on dair-ai/emotion.",
    version="2.0.0",
    lifespan=lifespan
)

# Compression & Security / CORS
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Endpoints ---
@app.get("/", include_in_schema=False)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(index_path))


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System and model health check"
)
def health_check():
    is_ready = bool(dl_resources.get("BiGRU") and dl_resources.get("Tokenizer"))
    tokenizer = dl_resources.get("Tokenizer")
    vocab_size = len(tokenizer.word_index) if tokenizer else 0

    return HealthResponse(
        status="healthy" if is_ready else "warming_up",
        model_loaded=is_ready,
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
        model_name=MODEL_PATH.name,
        classes=EMOTION_LABELS,
        vocab_size=vocab_size
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Predict emotion sentiment with full probability distribution"
)
def predict_emotion(text_input: TextInput):
    start_time = time.perf_counter()

    model = dl_resources.get("BiGRU")
    tokenizer = dl_resources.get("Tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Neural network model is not yet loaded into memory. Please retry shortly."
        )

    # 1. Clean incoming text
    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text contains no alphanumeric words after preprocessing."
        )

    # 2. Tokenize & Pad
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_seq = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # 3. Model Inference (Direct call avoids dataset iterator compilation latency)
    try:
        raw_pred = model(padded_seq, training=False)
        probabilities = raw_pred.numpy()[0] if hasattr(raw_pred, "numpy") else np.array(raw_pred)[0]
    except Exception:
        probabilities = model.predict(padded_seq, verbose=0)[0]
    top_idx = int(np.argmax(probabilities))
    predicted_emotion = EMOTION_LABELS[top_idx]
    confidence = float(probabilities[top_idx])

    # 4. Map probabilities across all 6 classes
    prob_dict = {
        label: round(float(prob), 6)
        for label, prob in zip(EMOTION_LABELS, probabilities)
    }

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        emoji=EMOTION_EMOJIS.get(predicted_emotion, "✨"),
        confidence=round(confidence, 6),
        all_probabilities=prob_dict,
        all_probabilites=prob_dict,
        latency_ms=latency_ms
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "path": request.url.path}
    )