"""LecViz - transcribe one audio file and benchmark the run.

Usage:
    python transcribe_file.py audio/lecture.wav
    python transcribe_file.py audio/lecture.wav --model base --no-vad

The script prints the transcript while faster-whisper produces it and
appends one row per run to ``results/benchmark.csv``.

The helpers in this module are also imported by ``main.py`` so the CLI and
the web UI always measure things the same way.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import psutil

from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    AUDIO_DIR,
    BEAM_SIZE,
    BENCHMARK_CSV,
    COMPUTE_TYPE,
    DEVICE,
    MODEL_SIZE,
    MODELS_DIR,
    TRANSCRIPTS_DIR,
    VAD_FILTER,
    ensure_directories,
)

logger = logging.getLogger("lecviz.transcribe")

SegmentCallback = Callable[[dict[str, Any]], None]
StatusCallback = Callable[[str, str], None]

# One model instance per (size, device, compute_type) triple.
_MODELS: dict[tuple[str, str, str], Any] = {}
_MODEL_LOCK = threading.Lock()

# Only one transcription at a time: keeps CPU/memory numbers meaningful and
# avoids fighting over the same model instance.
TRANSCRIBE_LOCK = threading.Lock()

CSV_FIELDS = [
    "timestamp",
    "file",
    "model",
    "device",
    "compute_type",
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
    "language",
    "language_probability",
    "success",
    "error",
    "text_preview",
]

_CSV_LOCK = threading.Lock()


class TranscriptionCancelled(RuntimeError):
    """Raised when the caller asks to stop an in-flight transcription."""


class ResourceMonitor:
    """Sample RSS / CPU usage from a background thread."""

    MB = 1024 * 1024

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_rss = 0
        self.peak_rss = 0
        self.samples: list[tuple[float, float, float, float]] = []

    def start(self) -> None:
        # First call primes psutil's CPU counters.
        self._process.cpu_percent(None)
        psutil.cpu_percent(None)
        self.baseline_rss = self._process.memory_info().rss
        self.peak_rss = self.baseline_rss
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="lecviz-resource-monitor", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rss = self._process.memory_info().rss
                process_cpu = self._process.cpu_percent(None)
                system_cpu = psutil.cpu_percent(None)
                system_ram = psutil.virtual_memory().percent
            except psutil.Error:
                break
            self.peak_rss = max(self.peak_rss, rss)
            self.samples.append((rss, process_cpu, system_cpu, system_ram))

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        try:
            self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)
        except psutil.Error:
            pass

        if self.samples:
            count = len(self.samples)
            avg_process_cpu = sum(s[1] for s in self.samples) / count
            peak_process_cpu = max(s[1] for s in self.samples)
            avg_system_cpu = sum(s[2] for s in self.samples) / count
            avg_system_ram = sum(s[3] for s in self.samples) / count
        else:
            avg_process_cpu = peak_process_cpu = avg_system_cpu = avg_system_ram = 0.0

        return {
            "peak_ram_mb": round(self.peak_rss / self.MB, 1),
            "ram_delta_mb": round((self.peak_rss - self.baseline_rss) / self.MB, 1),
            "avg_process_cpu_pct": round(avg_process_cpu, 1),
            "peak_process_cpu_pct": round(peak_process_cpu, 1),
            "avg_system_cpu_pct": round(avg_system_cpu, 1),
            "system_ram_pct": round(avg_system_ram, 1),
        }


def model_source(model_size: str = MODEL_SIZE) -> str | Path:
    """Prefer models/faster-whisper-<size>/ when it exists (offline-friendly).

    Otherwise return the model name and let faster-whisper download it from
    Hugging Face into the user's cache.
    """
    local_dir = MODELS_DIR / f"faster-whisper-{model_size}"
    if (local_dir / "model.bin").is_file():
        return local_dir
    return model_size


def get_model(
    model_size: str = MODEL_SIZE,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
) -> Any:
    """Load (and cache) a faster-whisper model."""
    source = model_source(model_size)
    key = (str(source), device, compute_type)
    with _MODEL_LOCK:
        if key not in _MODELS:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model from '%s' on %s (%s) ...",
                source,
                device,
                compute_type,
            )
            _MODELS[key] = WhisperModel(
                str(source), device=device, compute_type=compute_type
            )
        return _MODELS[key]


def is_model_loaded(model_size: str = MODEL_SIZE) -> bool:
    return any(key[0] == str(model_source(model_size)) for key in _MODELS)


def _status(callback: Optional[StatusCallback], phase: str, message: str) -> None:
    if callback is not None:
        try:
            callback(phase, message)
        except Exception:  # a broken UI callback must not kill transcription
            logger.exception("Status callback failed")


def _emit_segment(
    callback: Optional[SegmentCallback], payload: dict[str, Any]
) -> None:
    if callback is not None:
        try:
            callback(payload)
        except Exception:
            logger.exception("Segment callback failed")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _format_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def run_transcription(
    audio_path: str | os.PathLike[str],
    *,
    model_size: str = MODEL_SIZE,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    beam_size: int = BEAM_SIZE,
    vad_filter: bool = VAD_FILTER,
    on_segment: Optional[SegmentCallback] = None,
    on_status: Optional[StatusCallback] = None,
    cancel_event: Optional[threading.Event] = None,
) -> dict[str, Any]:
    """Transcribe ``audio_path`` and return a result dict with metrics.

    Raises:
        FileNotFoundError: if the audio file does not exist.
        TranscriptionCancelled: if ``cancel_event`` was set while running.
        Exception: any faster-whisper / decoding error is propagated as-is.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with TRANSCRIBE_LOCK:
        return _run_transcription_locked(
            path,
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
            on_segment=on_segment,
            on_status=on_status,
            cancel_event=cancel_event,
        )


def _run_transcription_locked(
    path: Path,
    *,
    model_size: str,
    device: str,
    compute_type: str,
    beam_size: int,
    vad_filter: bool,
    on_segment: Optional[SegmentCallback],
    on_status: Optional[StatusCallback],
    cancel_event: Optional[threading.Event],
) -> dict[str, Any]:
    monitor = ResourceMonitor()
    monitor.start()
    load_started = time.perf_counter()
    try:
        _status(on_status, "loading", "Loading the speech-to-text model ...")
        model = get_model(model_size, device, compute_type)
        load_time = time.perf_counter() - load_started

        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelled("Transcription cancelled before it started")

        _status(on_status, "transcribing", "Listening and transcribing ...")
        transcription_started = time.perf_counter()
        segments, info = model.transcribe(
            str(path),
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if not math.isfinite(duration):
            duration = 0.0
        language = str(getattr(info, "language", "") or "unknown")
        language_probability = float(
            getattr(info, "language_probability", 0.0) or 0.0
        )

        pieces: list[str] = []
        words = 0
        characters = 0
        segment_count = 0

        for index, segment in enumerate(segments):
            if cancel_event is not None and cancel_event.is_set():
                raise TranscriptionCancelled("Transcription cancelled by the client")

            text = (segment.text or "").strip()
            if not text:
                continue

            payload = {
                "id": index,
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
            }
            pieces.append(text)
            words += len(text.split())
            characters += len(text)
            segment_count += 1
            _emit_segment(on_segment, payload)

        processing_time = time.perf_counter() - transcription_started
    finally:
        metrics = monitor.stop()

    if duration <= 0:
        # Some containers do not expose a duration; fall back to the last
        # segment end so RTF stays meaningful.
        duration = float(payload["end"]) if segment_count else 0.0

    rtf = processing_time / duration if duration > 0 else 0.0
    speedup = 1.0 / rtf if rtf > 0 else 0.0
    full_text = " ".join(pieces).strip()

    result: dict[str, Any] = {
        "timestamp": _now_iso(),
        "file": path.name,
        "path": str(path),
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "audio_duration_sec": round(duration, 3),
        "processing_time_sec": round(processing_time, 3),
        "rtf": round(rtf, 4),
        "speedup": round(speedup, 2),
        "load_time_sec": round(load_time, 3),
        "segments": segment_count,
        "words": words,
        "characters": characters,
        "language": language,
        "language_probability": round(language_probability, 4),
        "text": full_text,
        "success": True,
        "error": "",
    }
    result.update(metrics)
    return result


def transcribe_chunk(
    audio_path: str | os.PathLike[str],
    *,
    model_size: str = MODEL_SIZE,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
    language: str = "en",
    beam_size: int = 1,
    vad_filter: bool = True,
) -> dict[str, Any]:
    """Transcribe a short live-microphone chunk (no benchmark row, no CSV).

    Used by the ``/ws/live`` WebSocket. The model instance is shared with the
    file transcription path, so the first live chunk may spend a few seconds
    loading the model when nothing has been transcribed yet.

    Returns a dict with the chunk ``text``, its ``segments`` (relative to the
    chunk start), the detected ``language``, and the decoded ``duration``.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio chunk not found: {path}")
    if path.stat().st_size == 0:
        return {"text": "", "segments": [], "language": language, "duration": 0.0}

    with TRANSCRIBE_LOCK:
        model = get_model(model_size, device, compute_type)
        segments, info = model.transcribe(
            str(path),
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100},
            condition_on_previous_text=False,
        )

        pieces: list[str] = []
        collected: list[dict[str, Any]] = []
        for segment in segments:
            text = (segment.text or "").strip()
            if not text:
                continue
            pieces.append(text)
            collected.append(
                {
                    "start": round(float(segment.start), 3),
                    "end": round(float(segment.end), 3),
                    "text": text,
                }
            )

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if not math.isfinite(duration):
            duration = 0.0
        detected = str(getattr(info, "language", language) or language)

    return {
        "text": " ".join(pieces).strip(),
        "segments": collected,
        "language": detected,
        "duration": round(duration, 3),
    }


def build_failure_result(
    audio_path: str | os.PathLike[str],
    error: BaseException | str,
    *,
    model_size: str = MODEL_SIZE,
    device: str = DEVICE,
    compute_type: str = COMPUTE_TYPE,
) -> dict[str, Any]:
    """Build a benchmark row for a failed run."""
    path = Path(audio_path)
    message = str(error) or error.__class__.__name__
    return {
        "timestamp": _now_iso(),
        "file": path.name,
        "path": str(path),
        "model": model_size,
        "device": device,
        "compute_type": compute_type,
        "audio_duration_sec": 0.0,
        "processing_time_sec": 0.0,
        "rtf": 0.0,
        "speedup": 0.0,
        "load_time_sec": 0.0,
        "peak_ram_mb": 0.0,
        "ram_delta_mb": 0.0,
        "avg_process_cpu_pct": 0.0,
        "peak_process_cpu_pct": 0.0,
        "avg_system_cpu_pct": 0.0,
        "system_ram_pct": 0.0,
        "segments": 0,
        "words": 0,
        "characters": 0,
        "language": "",
        "language_probability": 0.0,
        "text": "",
        "success": False,
        "error": message,
    }


def save_benchmark(result: dict[str, Any]) -> Path:
    """Append one result row to results/benchmark.csv (creating it if needed)."""
    ensure_directories()
    text = str(result.get("text", "") or "")
    row = {
        field: result.get(field, "")
        for field in CSV_FIELDS
        if field != "text_preview"
    }
    row["text_preview"] = " ".join(text.split())[:160]

    with _CSV_LOCK:
        file_exists = BENCHMARK_CSV.is_file() and BENCHMARK_CSV.stat().st_size > 0
        if file_exists:
            # Plain UTF-8 on append: utf-8-sig would insert a BOM before
            # every row and corrupt the CSV.
            with BENCHMARK_CSV.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writerow(row)
        else:
            with BENCHMARK_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(row)
    return BENCHMARK_CSV


def save_transcript(result: dict[str, Any]) -> Optional[Path]:
    """Write the full transcript to results/transcripts/ (returns None if empty)."""
    text = str(result.get("text", "") or "").strip()
    if not text:
        return None

    ensure_directories()
    stem = Path(str(result.get("file", "audio"))).stem or "audio"
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = TRANSCRIPTS_DIR / f"{safe_stem}_{stamp}.txt"
    if target.exists():
        target = TRANSCRIPTS_DIR / f"{safe_stem}_{stamp}_{uuid.uuid4().hex[:4]}.txt"

    header = [
        "# LecViz transcript",
        f"# file: {result.get('file', '')}",
        f"# model: {result.get('model', MODEL_SIZE)} "
        f"({result.get('device', DEVICE)}/{result.get('compute_type', COMPUTE_TYPE)})",
        f"# transcribed: {result.get('timestamp', '')}",
        f"# audio duration: {result.get('audio_duration_sec', 0)} s",
        f"# processing time: {result.get('processing_time_sec', 0)} s "
        f"(RTF {result.get('rtf', 0)})",
        "",
    ]
    target.write_text("\n".join(header) + text + "\n", encoding="utf-8")
    return target


def read_benchmarks(limit: int = 10) -> list[dict[str, str]]:
    """Return the most recent benchmark rows (newest first)."""
    if not BENCHMARK_CSV.is_file():
        return []
    try:
        with BENCHMARK_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        logger.exception("Could not read %s", BENCHMARK_CSV)
        return []
    return rows[-limit:][::-1]


def _print_summary(result: dict[str, Any], transcript_path: Optional[Path]) -> None:
    print()
    print("-" * 66)
    print(f"  File            : {result['file']}")
    print(f"  Model           : {result['model']} "
          f"({result['device']}/{result['compute_type']})")
    print(f"  Audio duration  : {_format_clock(result['audio_duration_sec'])}")
    print(f"  Processing time : {result['processing_time_sec']:.2f} s")
    print(f"  RTF             : {result['rtf']:.4f}  "
          f"({result['speedup']:.2f}x real-time)")
    print(f"  Peak RAM        : {result['peak_ram_mb']:.1f} MB "
          f"(+{result['ram_delta_mb']:.1f} MB)")
    print(f"  CPU (process)   : {result['avg_process_cpu_pct']:.1f}% avg / "
          f"{result['peak_process_cpu_pct']:.1f}% peak")
    print(f"  CPU (system)    : {result['avg_system_cpu_pct']:.1f}% avg")
    print(f"  Language        : {result['language']} "
          f"({result['language_probability'] * 100:.1f}%)")
    print(f"  Segments/words  : {result['segments']} / {result['words']}")
    print(f"  Benchmark       : {BENCHMARK_CSV}")
    if transcript_path is not None:
        print(f"  Transcript      : {transcript_path}")
    print("-" * 66)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with faster-whisper and log a benchmark row."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Path to an audio/video file (defaults to the first file in audio/).",
    )
    parser.add_argument("--model", default=MODEL_SIZE, help="tiny|base|small|medium|large-v3")
    parser.add_argument("--device", default=DEVICE, help="cpu or cuda")
    parser.add_argument("--compute-type", default=COMPUTE_TYPE, help="int8, int8_float16, float16, float32")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE)
    parser.add_argument("--no-vad", action="store_true", help="Disable voice-activity filtering.")
    parser.add_argument("--no-transcript", action="store_true", help="Do not save a .txt transcript.")
    parser.add_argument("--quiet", action="store_true", help="Do not print segments live.")
    return parser


def _default_audio_file() -> Optional[Path]:
    if not AUDIO_DIR.is_dir():
        return None
    candidates = sorted(
        (
            p
            for p in AUDIO_DIR.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main(argv: Optional[list[str]] = None) -> int:
    # Make Unicode transcript text safe on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    audio_file = Path(args.audio_file) if args.audio_file else _default_audio_file()
    if audio_file is None:
        print(
            "No audio file given and the audio/ folder is empty.\n"
            "Usage: python transcribe_file.py path\\to\\audio.wav",
            file=sys.stderr,
        )
        return 2
    if not audio_file.is_file():
        print(f"Audio file not found: {audio_file}", file=sys.stderr)
        return 2

    def on_status(phase: str, message: str) -> None:
        print(f"[{phase}] {message}", flush=True)

    def on_segment(segment: dict[str, Any]) -> None:
        if args.quiet:
            return
        stamp = (
            f"[{_format_clock(segment['start'])} --> "
            f"{_format_clock(segment['end'])}]"
        )
        print(f"{stamp} {segment['text']}", flush=True)

    print(f"Transcribing: {audio_file}")
    try:
        result = run_transcription(
            audio_file,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
            on_segment=on_segment,
            on_status=on_status,
        )
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:  # log the failure so the CSV reflects all attempts
        logger.error("Transcription failed: %s", exc)
        failed = build_failure_result(
            audio_file,
            exc,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        save_benchmark(failed)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Failure logged to {BENCHMARK_CSV}", file=sys.stderr)
        return 1

    save_benchmark(result)
    transcript_path = None if args.no_transcript else save_transcript(result)
    print()
    print("=== Transcript ===")
    print(result["text"] or "(no speech detected)")
    _print_summary(result, transcript_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
