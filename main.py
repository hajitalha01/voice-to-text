"""LecViz - FastAPI backend and web UI.

Run:
    uvicorn main:app --reload
Then open http://127.0.0.1:8000

Endpoints
---------
GET  /                       Web UI
GET  /api/config             Model + app configuration
GET  /api/health             Health / model-loaded status
GET  /api/files              Audio files available in audio/
POST /api/upload             Upload an audio file
DELETE /api/files/{name}     Delete an audio file
POST /api/sample             Generate a sample lecture WAV (Windows SAPI)
GET  /api/transcribe/stream  Server-Sent Events: live segments + final metrics
POST /api/transcribe         Blocking transcription (JSON API)
WS   /ws/live                Live microphone chunks -> English transcript
GET  /api/benchmarks         Most recent rows from results/benchmark.csv
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    AUDIO_DIR,
    BENCHMARK_CSV,
    BENCHMARK_PREVIEW_ROWS,
    COMPUTE_TYPE,
    CORS_ORIGINS,
    DEVICE,
    INDEX_HTML,
    LIVE_BEAM_SIZE,
    LIVE_LANGUAGE,
    LIVE_MAX_QUEUE,
    LIVE_VAD_FILTER,
    MAX_UPLOAD_MB,
    MODEL_SIZE,
    ensure_directories,
)
from transcribe_file import (
    TranscriptionCancelled,
    build_failure_result,
    is_model_loaded,
    model_source,
    read_benchmarks,
    run_transcription,
    save_benchmark,
    save_transcript,
    transcribe_chunk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lecviz.app")

APP_VERSION = "1.0.0"

app = FastAPI(
    title="LecViz",
    description="Local faster-whisper speech-to-text testing and benchmarking.",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_store_api_responses(request: Request, call_next):
    """Benchmark/file data changes at runtime, so never let the browser cache it."""
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


ensure_directories()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sse(event: str, payload: dict[str, Any]) -> str:
    """Encode one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop server-internal fields before sending a result to the browser."""
    return {key: value for key, value in result.items() if key != "path"}


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _list_audio_files() -> list[dict[str, Any]]:
    if not AUDIO_DIR.is_dir():
        return []
    files = [
        path
        for path in AUDIO_DIR.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [_file_info(path) for path in files]


def _resolve_audio(file: str) -> Path:
    """Resolve a user-supplied file name inside audio/ and block path traversal."""
    raw = Path(file.strip())
    candidate = (raw if raw.is_absolute() else AUDIO_DIR / raw).resolve()
    try:
        candidate.relative_to(AUDIO_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Audio file must be inside the audio/ folder."
        ) from exc
    if candidate.suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio format: {candidate.suffix or '(none)'}",
        )
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {file}")
    return candidate


def _safe_filename(filename: str) -> str:
    """Strip directories and unsafe characters from an uploaded file name."""
    name = Path(filename.replace("\\", "/")).name
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. Allowed: "
                + ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
            ),
        )
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).stem).strip(" ._")
    if not stem:
        stem = f"audio_{uuid.uuid4().hex[:8]}"
    return f"{stem[:80]}{suffix}"


def _unique_path(name: str) -> Path:
    target = AUDIO_DIR / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for index in range(1, 1000):
        candidate = AUDIO_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    return AUDIO_DIR / f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    if not INDEX_HTML.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Missing frontend file: {INDEX_HTML}",
        )
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Configuration / health / files
# ---------------------------------------------------------------------------
@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "app": "LecViz",
        "version": APP_VERSION,
        "model": MODEL_SIZE,
        "model_source": str(model_source(MODEL_SIZE)),
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "audio_dir": str(AUDIO_DIR),
        "results_dir": str(BENCHMARK_CSV.parent),
        "allowed_extensions": sorted(ALLOWED_AUDIO_EXTENSIONS),
        "max_upload_mb": MAX_UPLOAD_MB,
        "sample_name": "lecviz_sample.wav",
        "sample_available": (AUDIO_DIR / "lecviz_sample.wav").is_file(),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "app": "LecViz",
        "version": APP_VERSION,
        "model": MODEL_SIZE,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "model_loaded": is_model_loaded(MODEL_SIZE),
    }


@app.get("/api/files")
def list_files() -> dict[str, Any]:
    return {"files": _list_audio_files(), "audio_dir": str(AUDIO_DIR)}


@app.post("/api/upload", status_code=201)
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    name = _safe_filename(file.filename or "")
    target = _unique_path(name)
    limit = MAX_UPLOAD_MB * 1024 * 1024
    total = 0

    try:
        with target.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is larger than the {MAX_UPLOAD_MB} MB limit.",
                    )
                await asyncio.to_thread(handle.write, chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    logger.info("Uploaded %s (%.1f MB)", target.name, total / (1024 * 1024))
    return {"file": _file_info(target), "files": _list_audio_files()}


@app.delete("/api/files/{filename}")
def delete_file(filename: str) -> dict[str, Any]:
    path = _resolve_audio(filename)
    try:
        path.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {exc}") from exc
    return {"deleted": path.name, "files": _list_audio_files()}


# ---------------------------------------------------------------------------
# Sample audio (Windows built-in text-to-speech, no extra dependency)
# ---------------------------------------------------------------------------
SAMPLE_TEXT = (
    "Welcome to LecViz. This is a sample recording generated on your computer, "
    "so you can test the speech to text pipeline right away. "
    "Speech recognition systems convert spoken words into written text. "
    "In this short clip, we will check how quickly the model transcribes audio "
    "on a standard laptop processor. Real time factor compares processing time "
    "with the length of the recording. A value below one means the model is "
    "faster than real time. Thank you for testing LecViz."
)

_SAMPLE_PS1 = r"""param(
    [Parameter(Mandatory = $true)][string]$Text,
    [Parameter(Mandatory = $true)][string]$OutPath
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $synth.Rate = 0
    $synth.Volume = 100
    $synth.SetOutputToWaveFile($OutPath)
    $synth.Speak($Text)
} finally {
    $synth.Dispose()
}
"""


@app.post("/api/sample", status_code=201)
def create_sample() -> dict[str, Any]:
    """Create audio/lecviz_sample.wav with Windows' built-in speech synthesis."""
    target = AUDIO_DIR / "lecviz_sample.wav"
    if target.is_file() and target.stat().st_size > 1000:
        return {"file": _file_info(target), "files": _list_audio_files(), "created": False}

    script_path = Path(tempfile.gettempdir()) / "lecviz_sample_tts.ps1"
    script_path.write_text(_SAMPLE_PS1, encoding="utf-8-sig")

    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-Text",
        SAMPLE_TEXT,
        "-OutPath",
        str(target),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "Windows PowerShell was not found, so a sample could not be generated. "
                "Please upload an audio file instead."
            ),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=504, detail="Sample generation timed out. Please upload a file instead."
        ) from exc

    if (
        completed.returncode != 0
        or not target.is_file()
        or target.stat().st_size < 1000
    ):
        target.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "").strip()[:500]
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate the sample with Windows speech synthesis. "
                "Please upload an audio file instead."
                + (f" Details: {detail}" if detail else "")
            ),
        )

    logger.info("Generated sample audio: %s", target)
    return {"file": _file_info(target), "files": _list_audio_files(), "created": True}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
@app.get("/api/transcribe/stream")
async def transcribe_stream(file: str = Query(..., min_length=1)) -> StreamingResponse:
    """Stream transcription progress and segments over Server-Sent Events."""
    path = _resolve_audio(file)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[tuple[str, dict[str, Any]]]] = asyncio.Queue()
    cancel_event = threading.Event()

    def post(event: Optional[str], payload: dict[str, Any]) -> None:
        item = None if event is None else (event, payload)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # Event loop already closed (client went away) - nothing to do.
            pass

    def worker() -> None:
        try:
            result = run_transcription(
                path,
                on_status=lambda phase, message: post(
                    "status", {"phase": phase, "message": message}
                ),
                on_segment=lambda segment: post("segment", segment),
                cancel_event=cancel_event,
            )
            try:
                save_benchmark(result)
            except Exception:
                logger.exception("Could not append to the benchmark CSV")
            transcript_path = None
            try:
                transcript_path = save_transcript(result)
            except Exception:
                logger.exception("Could not save the transcript file")
            payload = _public_result(result)
            if transcript_path is not None:
                payload["transcript_file"] = transcript_path.name
            post("done", payload)
        except TranscriptionCancelled:
            logger.info("Transcription cancelled for %s", path.name)
            post("cancelled", {"message": "Transcription cancelled."})
        except Exception as exc:
            logger.exception("Transcription failed for %s", path)
            failed = build_failure_result(path, exc)
            try:
                save_benchmark(failed)
            except Exception:
                logger.exception("Could not write failure row to benchmark CSV")
            post("error", {"message": str(exc) or exc.__class__.__name__})
        finally:
            post(None, {})

    async def event_stream():
        yield _sse(
            "status",
            {"phase": "starting", "message": f"Preparing {path.name} ..."},
        )
        thread = threading.Thread(
            target=worker, name="lecviz-transcription", daemon=True
        )
        thread.start()
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    break
                event, payload = item
                yield _sse(event, payload)
        finally:
            # Client disconnected or stream finished: stop decoding at the
            # next segment boundary so we do not burn CPU for nobody.
            cancel_event.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/transcribe")
async def transcribe(file: str = Query(..., min_length=1)) -> JSONResponse:
    """Blocking transcription endpoint (useful for scripts and curl)."""
    path = _resolve_audio(file)
    try:
        result = await asyncio.to_thread(run_transcription, path)
    except TranscriptionCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Transcription failed for %s", path)
        try:
            await asyncio.to_thread(save_benchmark, build_failure_result(path, exc))
        except Exception:
            logger.exception("Could not write failure row to benchmark CSV")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    try:
        await asyncio.to_thread(save_benchmark, result)
    except Exception:
        logger.exception("Could not append to the benchmark CSV")
    transcript_path = None
    try:
        transcript_path = await asyncio.to_thread(save_transcript, result)
    except Exception:
        logger.exception("Could not save the transcript file")
    payload = _public_result(result)
    if transcript_path is not None:
        payload["transcript_file"] = transcript_path.name
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Live microphone streaming (WebSocket)
# ---------------------------------------------------------------------------
_LIVE_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
}


def _live_suffix(mime_type: str) -> str:
    """Map a MediaRecorder MIME type to a container extension PyAV can open."""
    base = (mime_type or "").split(";", 1)[0].strip().lower()
    return _LIVE_SUFFIXES.get(base, ".webm")


def _remove_quietly(path: Path) -> None:
    """Delete a temp chunk; Windows may still hold it briefly during a cancel."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Could not delete live audio chunk %s", path)


@app.websocket("/ws/live")
async def live_transcription(websocket: WebSocket) -> None:
    """Transcribe microphone chunks streamed from the browser.

    Browser -> server (text frame, then binary frame for each chunk):
        {"type": "start", "mime_type": "audio/webm;codecs=opus"}
        {"type": "chunk", "offset": 12.5}
        <binary MediaRecorder blob>
        {"type": "stop"}

    Server -> browser:
        {"type": "ready", "model": "small", "language": "en"}
        {"type": "transcript", "text": "...", "offset": 12.5, "segments": [...]}
        {"type": "error", "message": "..."}
        {"type": "done", "text": "<whole session>", "segments": 7}

    Chunks are processed one at a time per connection (and the shared model
    lock keeps them from colliding with a file transcription), so transcript
    text always arrives in recording order.
    """
    await websocket.accept()

    work_dir = Path(tempfile.mkdtemp(prefix="lecviz-live-"))
    queue: asyncio.Queue[Optional[tuple[Path, float]]] = asyncio.Queue(
        maxsize=LIVE_MAX_QUEUE
    )
    language = LIVE_LANGUAGE
    mime_type = "audio/webm"
    next_offset = 0.0
    stopping = False
    parts: list[str] = []
    segment_total = 0
    announced_load = False

    async def send(payload: dict[str, Any]) -> bool:
        """Send JSON, returning False once the browser has gone away."""
        try:
            await websocket.send_json(payload)
            return True
        except Exception:  # closed socket / client disconnect
            return False

    def enqueue(item: Optional[tuple[Path, float]]) -> None:
        """Queue work, dropping the oldest chunk if decoding falls behind."""
        if queue.full():
            try:
                dropped = queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                dropped = None
            if dropped is not None:
                _remove_quietly(dropped[0])
                logger.warning("Live audio queue full - dropped one chunk")
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            if item is not None:
                _remove_quietly(item[0])

    async def worker() -> None:
        nonlocal segment_total, announced_load
        while True:
            item = await queue.get()
            try:
                if item is None:  # stop marker: everything before it is done
                    break
                path, offset = item
                if not announced_load and not is_model_loaded(MODEL_SIZE):
                    announced_load = True
                    if not await send(
                        {
                            "type": "status",
                            "message": "Loading the speech-to-text model (first live run)…",
                        }
                    ):
                        break
                try:
                    result = await asyncio.to_thread(
                        transcribe_chunk,
                        path,
                        language=language,
                        beam_size=LIVE_BEAM_SIZE,
                        vad_filter=LIVE_VAD_FILTER,
                    )
                except Exception as exc:
                    logger.exception("Live chunk transcription failed")
                    if not await send(
                        {
                            "type": "error",
                            "message": f"Could not transcribe this audio chunk: {exc}",
                        }
                    ):
                        break
                    continue
                finally:
                    _remove_quietly(path)

                text = str(result.get("text", "")).strip()
                if not text:
                    continue

                pieces = result.get("segments") or []
                parts.append(text)
                segment_total += max(1, len(pieces))
                if not await send(
                    {
                        "type": "transcript",
                        "text": text,
                        "offset": round(offset, 2),
                        "duration": result.get("duration", 0.0),
                        "segments": pieces,
                        "language": result.get("language", language),
                    }
                ):
                    break
            finally:
                queue.task_done()

        await send(
            {
                "type": "done",
                "text": " ".join(parts).strip(),
                "segments": segment_total,
                "language": language,
            }
        )

    task: Optional[asyncio.Task[None]] = None
    try:
        await send({"type": "ready", "model": MODEL_SIZE, "language": LIVE_LANGUAGE})
        task = asyncio.create_task(worker())

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_frame = message.get("text")
            if text_frame is not None:
                try:
                    control = json.loads(text_frame)
                except (TypeError, ValueError):
                    continue
                if not isinstance(control, dict):
                    continue

                action = control.get("type")
                if action == "start":
                    mime_type = str(control.get("mime_type") or mime_type)
                    requested = str(control.get("language") or "").strip().lower()
                    if requested:
                        language = requested
                elif action == "chunk":
                    try:
                        next_offset = max(0.0, float(control.get("offset") or 0.0))
                    except (TypeError, ValueError):
                        next_offset = 0.0
                elif action == "stop":
                    stopping = True
                    enqueue(None)
                    break
                continue

            data = message.get("bytes")
            if data:
                if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                    await send(
                        {
                            "type": "error",
                            "message": f"Audio chunk exceeds the {MAX_UPLOAD_MB} MB limit.",
                        }
                    )
                    continue
                chunk_path = (
                    work_dir / f"chunk_{uuid.uuid4().hex}{_live_suffix(mime_type)}"
                )
                chunk_path.write_bytes(data)
                enqueue((chunk_path, next_offset))

    except WebSocketDisconnect:
        logger.info("Live transcription client disconnected")
    except Exception:
        logger.exception("Live transcription session failed")
    finally:
        if task is not None:
            try:
                if stopping:
                    # Let the queued chunks finish so "done" carries the text.
                    await asyncio.wait_for(task, timeout=120)
                else:
                    task.cancel()
                    await task
            except (asyncio.CancelledError, asyncio.TimeoutError):
                task.cancel()
            except Exception:
                logger.exception("Live transcription worker stopped with an error")
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/benchmarks.csv", include_in_schema=False)
def benchmark_csv() -> FileResponse:
    """Download the raw benchmark CSV (opens it directly in Excel)."""
    if not BENCHMARK_CSV.is_file():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results yet. Run a transcription first.",
        )
    return FileResponse(
        BENCHMARK_CSV,
        media_type="text/csv",
        filename="lecviz_benchmark.csv",
    )


@app.get("/api/benchmarks")
def benchmarks(limit: int = Query(BENCHMARK_PREVIEW_ROWS, ge=1, le=100)) -> dict[str, Any]:
    all_rows = read_benchmarks(100_000)
    rows = all_rows[:limit]
    numeric_fields = {
        "audio_duration_sec",
        "processing_time_sec",
        "rtf",
        "speedup",
        "load_time_sec",
        "peak_ram_mb",
        "ram_delta_mb",
        "avg_process_cpu_pct",
        "peak_process_cpu_pct",
        "avg_system_cpu_pct",
        "system_ram_pct",
        "segments",
        "words",
        "characters",
        "language_probability",
    }

    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        for field in numeric_fields:
            try:
                item[field] = float(item.get(field, 0) or 0)
            except (TypeError, ValueError):
                item[field] = 0.0
        item["success"] = str(row.get("success", "")).strip().lower() in {
            "true",
            "1",
            "yes",
        }
        parsed.append(item)

    return {
        "rows": parsed,
        "csv": str(BENCHMARK_CSV),
        "total": len(all_rows),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
