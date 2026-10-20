# LecViz

Local speech-to-text testing and benchmarking for lecture audio, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It runs entirely on
your machine — no API keys, no audio leaves the laptop.

Tuned for a CPU-only laptop (Intel i7 10th Gen, 16 GB RAM, no GPU):

| Setting | Value |
| --- | --- |
| Model | `small` |
| Device | `cpu` |
| Compute type | `int8` |

## Features

- Clean web UI with file picker, drag-and-drop upload, and a generated sample clip.
- **Live microphone mode** — record straight from the browser and watch English
  speech become text as you talk (MediaRecorder + WebSocket chunks).
- **Live transcript** that streams segment by segment while the model is running
  (Server-Sent Events, no polling).
- Per-run benchmark: processing time, **Real-Time Factor (RTF)**, speed-up,
  peak RAM, process/system CPU, audio duration, language confidence.
- Every run is appended to `results/benchmark.csv`; full transcripts are saved to
  `results/transcripts/`.
- CLI for batch/offline benchmarking: `python transcribe_file.py audio/file.mp3`.
- Recent-runs table in the UI, plus a one-click CSV download.

## Project layout

```
LecViz/
├── audio/                  # put test audio here (uploads land here too)
├── results/                # benchmark.csv + transcripts/
├── models/                 # optional local model (auto-detected, gitignored)
├── templates/
│   └── index.html          # single-file frontend (no build step, no CDN)
├── config.py               # model + paths + upload limits
├── transcribe_file.py      # CLI transcription + shared benchmark engine
├── main.py                 # FastAPI app / API / UI server
└── requirements.txt
```

## Setup

**Python version:** Python 3.10 or newer. This project was installed and tested
on Windows 10 with **Python 3.14.8** and faster-whisper 1.2.1. If a future
`ctranslate2`/`av` release lacks a wheel for your Python version, use Python
3.11 or 3.12 instead.

```powershell
cd C:\Users\Haji\LecViz

# Optional but recommended on a fresh machine: create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (already installed in this environment)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks activation, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call
> `.\.venv\Scripts\python.exe` directly.

No separate FFmpeg installation is needed: faster-whisper decodes audio through
PyAV, whose wheels bundle the required libraries.

## Run the web app

```powershell
cd C:\Users\Haji\LecViz
uvicorn main:app --reload
```

If PowerShell says `uvicorn` is not recognized, open a new terminal (the
Python `Scripts` folder was added to your user PATH during setup) or run:

```powershell
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. API docs: <http://127.0.0.1:8000/docs>.

First run: faster-whisper downloads the `small` model (~485 MB) from Hugging
Face into `%USERPROFILE%\.cache\huggingface`. The status line shows
"Loading the speech-to-text model…" while this happens; later runs load from
disk in a couple of seconds.

If a folder `models/faster-whisper-small/` containing `model.bin`,
`config.json`, `tokenizer.json`, and `vocabulary.txt` exists, LecViz uses it
directly and skips the download entirely. Such a folder is provided on this
machine, so the app works offline. Set `HF_HUB_DISABLE_XET=1` if a Hugging Face
download keeps restarting instead of resuming.

### Using the UI

1. Pick a file from `audio/`, drag one onto the drop zone, or click
   **Generate a sample recording** (uses the built-in Windows voice).
2. Press **Start transcription** (or `Ctrl+Enter`).
3. Text streams into the live panel as each segment is decoded.
4. When it finishes, the Performance cards and Recent runs table update.
5. Use **Copy** / **Save** for the transcript, or the **CSV** button in the
   header to download benchmark results.

**Live microphone:** click **Microphone** in the Audio source panel, allow
microphone access, and start speaking. A pulsing red indicator with a timer
shows the session is active. The browser sends a fresh 4-second audio chunk
every 4 seconds; each chunk is transcribed in English and appended to the Live
transcript panel. Click **Stop recording** (or press `Esc`) to flush the last
chunk. Live sessions are not written to `benchmark.csv`; use **Copy** / **Save**
to keep the text. Microphone capture needs a secure context, which includes
`http://127.0.0.1` / `http://localhost`.

A file transcription can be cancelled with the **Cancel** button or `Esc`; the
server stops at the next segment boundary. Cancelled runs are not written to
the CSV.

## CLI usage

Transcribe a single file and print a benchmark summary:

```powershell
python transcribe_file.py audio\lecture.mp3
```

With no path, the most recently added file in `audio/` is used. Options:

```text
--model small|base|tiny|medium|large-v3
--device cpu|cuda
--compute-type int8|int8_float16|float16|float32
--beam-size 5
--no-vad              # disable voice-activity filtering
--no-transcript       # don't save results/transcripts/<name>_<time>.txt
--quiet               # don't print segments live
```

Examples:

```powershell
python transcribe_file.py audio\lecture.mp3 --model base
python transcribe_file.py audio\lecture.mp3 --model small --beam-size 1
```

## Benchmark CSV

`results/benchmark.csv` is written with a UTF-8 BOM so Excel opens it correctly.

| Column | Meaning |
| --- | --- |
| `timestamp` | Local time when the run finished |
| `file`, `model`, `device`, `compute_type` | What was tested |
| `audio_duration_sec` | Length reported by the decoder |
| `processing_time_sec` | Wall-clock transcription time (excludes model load) |
| `rtf` | `processing_time_sec / audio_duration_sec` |
| `speedup` | `audio_duration_sec / processing_time_sec` (higher is better) |
| `load_time_sec` | Time spent loading the model for this run (0 if cached) |
| `peak_ram_mb`, `ram_delta_mb` | Peak process RSS and increase over baseline |
| `avg_process_cpu_pct` | Average CPU used by the Python process (can exceed 100% across cores) |
| `peak_process_cpu_pct` | Highest process CPU sample |
| `avg_system_cpu_pct` | Average whole-machine CPU usage (0–100%) |
| `system_ram_pct` | Average whole-machine RAM usage |
| `segments`, `words`, `characters`, `language`, `language_probability` | Transcript stats |
| `success`, `error` | Failed runs are logged too |
| `text_preview` | First 160 characters of the transcript |

**Reading RTF:** `RTF < 1` is faster than real time. `RTF = 0.4` means 10 minutes
of audio takes about 4 minutes on this machine. A "speed-up" of `2.5×` is the
same measurement from the other side.

## API quick reference

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/api/config` | Model, paths, allowed formats, upload limit |
| `GET` | `/api/health` | App status + whether the model is loaded |
| `GET` | `/api/files` | Audio files in `audio/` |
| `POST` | `/api/upload` | Multipart upload (`file` field) |
| `DELETE` | `/api/files/{name}` | Delete a file from `audio/` |
| `POST` | `/api/sample` | Create the Windows TTS sample clip |
| `GET` | `/api/transcribe/stream?file=name.wav` | SSE stream: `status`, `segment`, `done`, `error` |
| `POST` | `/api/transcribe?file=name.wav` | Blocking transcription, returns JSON |
| `WS` | `/ws/live` | Live microphone chunks → English transcript |
| `GET` | `/api/benchmarks?limit=10` | Recent benchmark rows as JSON |
| `GET` | `/api/benchmarks.csv` | Download the raw CSV |

The `/ws/live` socket speaks a tiny JSON protocol: send
`{"type":"start","mime_type":"audio/webm;codecs=opus"}`, then
`{"type":"chunk","offset":<seconds>}` followed by one binary audio blob per
chunk, and finally `{"type":"stop"}`. The server replies with `ready`,
`transcript`, `error`, and a final `done` carrying the whole session text.
Chunks are decoded in recording order and each blob must be a complete audio
container (the bundled UI restarts MediaRecorder per chunk to guarantee this).

Example with `curl`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/transcribe?file=lecviz_sample.wav"
```

## Troubleshooting

**`pip install` fails on `ctranslate2` / `av`**
Use Python 3.11 or 3.12. Check with `python --version` inside the activated venv.

**`FileNotFoundError: Could not find module 'ctranslate2.dll' (or one of its dependencies)`**
Windows is missing the Microsoft Visual C++ runtime. Install the official
**Microsoft Visual C++ Redistributable for Visual Studio 2015–2022 (x64)**:

```powershell
winget install --id Microsoft.VCRedist.2015+.x64
```

(or download `vc_redist.x64.exe` from <https://aka.ms/vs/17/release/vc_redist.x64.exe>).
The same missing runtime can make `onnxruntime` fail with
`DLL load failed while importing onnxruntime_pybind11_state`.

If you cannot install it because you do not have administrator rights, a
non-admin workaround is to copy `msvcp140.dll`, `msvcp140_1.dll`,
`msvcp140_2.dll`, `vcruntime140_1.dll`, and `concrt140.dll` from an application
that bundles them (for example Edge or OneDrive) next to `python.exe`. This was
done on this machine; the official redistributable is the better long-term fix.

**`open() got an unexpected keyword argument 'metadata_errors'`**
PyAV 19 removed the `metadata_errors` argument that faster-whisper 1.2.x still
passes. `requirements.txt` pins `av<19` (18.1.0 was tested); reinstall with:

```powershell
pip install "av>=14,<19"
```

**First transcription seems stuck**
The model download can take several minutes on a slow connection. Watch the
status line; the terminal running uvicorn also logs model loading.

**`Generate a sample recording` fails**
It uses Windows PowerShell's `System.Speech` API. On stripped-down Windows
installs that component may be missing — upload a WAV/MP3 instead. Any audio
format supported by PyAV works.

**`Microphone` does nothing or shows a permission error**
The browser needs microphone permission and a secure context; `127.0.0.1` and
`localhost` count as secure, but a plain-HTTP LAN address does not. Allow the
microphone in the site settings, make sure no other app holds the device
exclusively, and use a recent Chrome, Edge, or Firefox. Live mode sends audio
only to your local FastAPI server.

**Transcription is slower than expected**
- Close other CPU-heavy apps; laptops throttle on battery. Plug in the charger
  and use the "Balanced" or "Best performance" power plan.
- Try `--model base` (faster, less accurate) or `--model small --beam-size 1`.
- `int8` is already the fastest CPU compute type.
- Disable VAD only for testing — it normally speeds things up by skipping silence.

**Memory**
In testing, `small` + int8 peaked at roughly **0.56–0.62 GB RSS** for the whole
process, including Python and the loaded model. `medium` needs noticeably more;
on a 16 GB machine it still fits, but expect slower runs.

**Port already in use**

```powershell
uvicorn main:app --reload --port 8001
```

**Windows Firewall prompt**
Allow access for private networks if you want to open the UI from another
device. Otherwise the default `127.0.0.1` binding is local-only.

## Notes on accuracy

- `small` is a good speed/quality trade-off for English lectures. For noisy
  recordings or heavy accents, try `medium` (slower) or `small.en`.
- The sample clip is synthesized speech — real lecture audio will produce
  different RTFs and word counts.
- Language detection runs automatically; the detected language and confidence
  are shown under the transcript.

## License

MIT — do whatever you like with it.
