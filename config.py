"""Central configuration for LecViz.

Everything that controls the speech-to-text model and the benchmark
layout lives here so the CLI and the web app always behave the same way.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Model configuration (tuned for CPU-only laptops such as an i7 / 16 GB RAM)
# ---------------------------------------------------------------------------
MODEL_SIZE = "medium"          # tiny | base | small | medium | large-v3
DEVICE = "cpu"                # "cpu" or "cuda"
COMPUTE_TYPE = "int8"         # int8 is the fastest CPU quantisation

# Decoding defaults
BEAM_SIZE = 5
VAD_FILTER = True             # skip long silences -> faster + cleaner output

# ---------------------------------------------------------------------------
# Live microphone streaming (browser -> WebSocket)
# ---------------------------------------------------------------------------
LIVE_LANGUAGE = "en"          # live mode is English-only
LIVE_BEAM_SIZE = 1            # greedy decoding keeps short chunks fast
LIVE_VAD_FILTER = True        # drop silence and avoid "thank you" hallucinations
LIVE_MAX_QUEUE = 8            # buffered chunks per client before the oldest is dropped

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
RESULTS_DIR = BASE_DIR / "results"
TEMPLATES_DIR = BASE_DIR / "templates"
MODELS_DIR = BASE_DIR / "models"

BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
INDEX_HTML = TEMPLATES_DIR / "index.html"

# ---------------------------------------------------------------------------
# Upload / API limits
# ---------------------------------------------------------------------------
ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}

MAX_UPLOAD_MB = 500
BENCHMARK_PREVIEW_ROWS = 10
CORS_ORIGINS = ["*"]  # local-only app; tighten if you expose it on a network


def ensure_directories() -> None:
    """Create the runtime folders if they do not exist yet."""
    for directory in (AUDIO_DIR, RESULTS_DIR, TRANSCRIPTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
