#!/usr/bin/env python3
"""
understudy transcription benchmark harness.

Standalone. Copy this one file anywhere and run it -- it does not import
the understudy package.

What it does
------------
1. Probes the machine (CPU, RAM, disk, GPU, ffmpeg/av availability).
2. Picks a ladder of Whisper model sizes that plausibly fit this machine.
3. For each model, in an isolated subprocess: times the load (including
   first-run download), transcribes a short clip taken from the middle of
   your real audio, samples peak memory, and saves the transcript text.
4. Writes bench-results.json and prints a comparison table with the
   projected wall-clock time for a full-length session.

Each model runs in its own subprocess so that an out-of-memory kill, a
native crash, or a blocked model download takes down only that one trial.

Usage
-----
    python bench_transcribe.py --probe-only
    python bench_transcribe.py --audio PATH_TO_AUDIO
    python bench_transcribe.py --audio PATH --models small.en,distil-large-v3
    python bench_transcribe.py --session PATH_TO_SESSION_FOLDER

Requires: pip install faster-whisper
Optional: pip install psutil   (better RAM/core detection; degrades gracefully)
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

RESULT_PREFIX = "##RESULT##"
SAMPLE_RATE = 16000

AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".mka", ".webm")

# name, min_ram_gb, approx_disk_mb, english_only, gpu_only
MODEL_LADDER = [
    ("tiny.en",          2,    75,  True,  False),
    ("base.en",          2,   145,  True,  False),
    ("small.en",         4,   480,  True,  False),
    ("distil-small.en",  4,   400,  True,  False),
    ("distil-large-v3",  8,  1500,  False, False),
    ("large-v3-turbo",   8,  1600,  False, False),
    ("large-v3",        12,  3090,  False, False),
]


# ----------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------

def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def probe():
    ps = _psutil()
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "psutil": bool(ps),
    }

    if ps:
        info["cores_physical"] = ps.cpu_count(logical=False) or 1
        info["cores_logical"] = ps.cpu_count(logical=True) or 1
        vm = ps.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["ram_available_gb"] = round(vm.available / 1e9, 1)
    else:
        info["cores_physical"] = None
        info["cores_logical"] = os.cpu_count() or 1
        info["ram_total_gb"] = None
        info["ram_available_gb"] = None

    try:
        home = os.path.expanduser("~")
        info["disk_free_gb"] = round(shutil.disk_usage(home).free / 1e9, 1)
    except Exception:
        info["disk_free_gb"] = None

    # CUDA via ctranslate2 is authoritative for faster-whisper; nvidia-smi is
    # a fallback that also tells us the card name.
    info["cuda_devices"] = 0
    try:
        import ctranslate2
        info["cuda_devices"] = ctranslate2.get_cuda_device_count()
        info["ctranslate2"] = ctranslate2.__version__
    except Exception as e:
        info["ctranslate2"] = "not installed ({})".format(type(e).__name__)

    info["gpu_name"] = None
    info["gpu_vram_mb"] = None
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0 and out.stdout.strip():
                first = out.stdout.strip().splitlines()[0]
                name, _, vram = first.partition(",")
                info["gpu_name"] = name.strip()
                try:
                    info["gpu_vram_mb"] = int(vram.strip())
                except ValueError:
                    pass
        except Exception:
            pass

    try:
        import faster_whisper
        info["faster_whisper"] = getattr(faster_whisper, "__version__", "unknown")
    except ImportError:
        info["faster_whisper"] = None

    info["ffmpeg_on_path"] = bool(shutil.which("ffmpeg"))
    try:
        import av
        info["pyav"] = av.__version__
    except Exception:
        info["pyav"] = None

    # Corporate networks frequently block or TLS-inspect huggingface.co,
    # which is where model weights come from on first run.
    info["hf_home"] = os.environ.get("HF_HOME") or os.environ.get(
        "HUGGINGFACE_HUB_CACHE") or "(default)"
    info["hf_offline"] = os.environ.get("HF_HUB_OFFLINE", "0")

    return info


def print_probe(info):
    print("=" * 68)
    print("MACHINE PROBE")
    print("=" * 68)
    rows = [
        ("Platform", info["platform"]),
        ("Processor", info["processor"]),
        ("Cores (physical/logical)",
         "{} / {}".format(info["cores_physical"] or "?", info["cores_logical"])),
        ("RAM total", _gb(info["ram_total_gb"])),
        ("RAM available", _gb(info["ram_available_gb"])),
        ("Disk free (home)", _gb(info["disk_free_gb"])),
        ("CUDA devices", info["cuda_devices"]),
        ("GPU", info["gpu_name"] or "none detected"),
        ("GPU VRAM", "{} MB".format(info["gpu_vram_mb"]) if info["gpu_vram_mb"] else "-"),
        ("Python", info["python"]),
        ("faster-whisper", info["faster_whisper"] or "NOT INSTALLED"),
        ("ctranslate2", info["ctranslate2"]),
        ("PyAV (audio decode)", info["pyav"] or "NOT INSTALLED"),
        ("ffmpeg on PATH", "yes" if info["ffmpeg_on_path"] else "no (not required)"),
        ("HF cache dir", info["hf_home"]),
        ("HF offline mode", info["hf_offline"]),
    ]
    for k, v in rows:
        print("  {:<26} {}".format(k, v))
    print()

    if not info["psutil"]:
        print("  NOTE: psutil not installed -- RAM detection unavailable, so the")
        print("        model ladder cannot be gated by memory. pip install psutil")
        print()


def _gb(v):
    return "{} GB".format(v) if v is not None else "unknown"


# ----------------------------------------------------------------------
# model selection
# ----------------------------------------------------------------------

def choose_models(info, requested=None):
    if requested:
        return [m.strip() for m in requested.split(",") if m.strip()]

    ram = info["ram_total_gb"]
    gpu = info["cuda_devices"] > 0

    picked = []
    for name, min_ram, _disk, _en, _gpu_only in MODEL_LADDER:
        if ram is not None and ram < min_ram:
            continue
        if name == "large-v3" and not gpu:
            # ~1x realtime on CPU: a 30 min session takes 30+ min. Skip by
            # default; --models large-v3 still forces it.
            continue
        picked.append(name)

    if not picked:
        picked = ["tiny.en", "base.en"]
    return picked


def device_and_compute(info):
    if info["cuda_devices"] > 0:
        return "cuda", "float16"
    return "cpu", "int8"


def worker_threads(info):
    phys = info.get("cores_physical")
    if not phys:
        phys = max(1, (info.get("cores_logical") or 2) // 2)
    # Leave one physical core free so the machine stays usable while this runs.
    return max(1, phys - 1)


# ----------------------------------------------------------------------
# audio
# ----------------------------------------------------------------------

def find_audio(session_dir):
    hits = []
    for root, _dirs, files in os.walk(session_dir):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                p = os.path.join(root, f)
                hits.append((os.path.getsize(p), p))
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


# ----------------------------------------------------------------------
# child mode: run exactly one model, print JSON
# ----------------------------------------------------------------------

def run_one(args):
    result = {
        "model": args.run_one,
        "ok": False,
        "error": None,
        "load_s": None,
        "transcribe_s": None,
        "rtf": None,
        "peak_rss_mb": None,
        "clip_s": args.clip,
        "chars": 0,
    }

    ps = _psutil()
    stop = threading.Event()
    peak = [0]

    def sample():
        if not ps:
            return
        proc = ps.Process()
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], proc.memory_info().rss)
            except Exception:
                return
            time.sleep(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio

        audio = decode_audio(args.audio, sampling_rate=SAMPLE_RATE)
        total_s = len(audio) / SAMPLE_RATE
        result["audio_total_s"] = round(total_s, 1)

        # Sample from the middle: the opening of a meeting recording is
        # usually silence and "can you hear me", which is not representative.
        start = int(total_s * args.offset) * SAMPLE_RATE
        end = start + int(args.clip * SAMPLE_RATE)
        if end > len(audio):
            start = max(0, len(audio) - int(args.clip * SAMPLE_RATE))
            end = len(audio)
        clip = audio[start:end]
        result["clip_s"] = round(len(clip) / SAMPLE_RATE, 1)

        t0 = time.perf_counter()
        model = WhisperModel(
            args.run_one,
            device=args.device,
            compute_type=args.compute,
            cpu_threads=args.threads,
        )
        result["load_s"] = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        segments, _info = model.transcribe(clip, beam_size=args.beam, language="en")
        text = " ".join(s.text for s in segments).strip()
        elapsed = time.perf_counter() - t1

        result["transcribe_s"] = round(elapsed, 2)
        result["rtf"] = round(result["clip_s"] / elapsed, 2) if elapsed > 0 else None
        result["chars"] = len(text)
        result["ok"] = True

        if args.outdir:
            os.makedirs(args.outdir, exist_ok=True)
            safe = args.run_one.replace("/", "_")
            with open(os.path.join(args.outdir, "transcript-{}.txt".format(safe)),
                      "w", encoding="utf-8") as fh:
                fh.write(text + "\n")

    except Exception as e:
        result["error"] = "{}: {}".format(type(e).__name__, str(e)[:400])
    finally:
        stop.set()
        sampler.join(timeout=2)
        if peak[0]:
            result["peak_rss_mb"] = round(peak[0] / 1e6, 1)

    print(RESULT_PREFIX + json.dumps(result))
    return 0


# ----------------------------------------------------------------------
# parent mode
# ----------------------------------------------------------------------

def run_bench(args, info):
    device, compute = device_and_compute(info)
    threads = worker_threads(info)
    models = choose_models(info, args.models)

    print("=" * 68)
    print("BENCHMARK PLAN")
    print("=" * 68)
    print("  Audio        {}".format(args.audio))
    print("  Device       {} ({})".format(device, compute))
    print("  CPU threads  {}".format(threads))
    print("  Beam size    {}".format(args.beam))
    print("  Clip length  {} s, taken {:.0%} into the file".format(args.clip, args.offset))
    print("  Models       {}".format(", ".join(models)))
    print("  Timeout      {} s per model".format(args.timeout))
    print()
    print("  First run downloads weights from huggingface.co. On a locked-down")
    print("  corporate network that download may fail -- that shows up below as")
    print("  a per-model error, not a crash.")
    print()

    results = []
    for name in models:
        print("-" * 68)
        print("[{}] starting...".format(name))
        sys.stdout.flush()

        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--run-one", name,
            "--audio", args.audio,
            "--clip", str(args.clip),
            "--offset", str(args.offset),
            "--beam", str(args.beam),
            "--device", device,
            "--compute", compute,
            "--threads", str(threads),
        ]
        if args.outdir:
            cmd += ["--outdir", args.outdir]

        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout)
            payload = None
            for line in proc.stdout.splitlines():
                if line.startswith(RESULT_PREFIX):
                    payload = json.loads(line[len(RESULT_PREFIX):])
            if payload is None:
                payload = {
                    "model": name, "ok": False,
                    "error": "no result returned (exit {}); stderr: {}".format(
                        proc.returncode, (proc.stderr or "")[-300:]),
                }
        except subprocess.TimeoutExpired:
            payload = {"model": name, "ok": False,
                       "error": "timed out after {}s".format(args.timeout)}
        except Exception as e:
            payload = {"model": name, "ok": False,
                       "error": "{}: {}".format(type(e).__name__, e)}

        payload["wall_s"] = round(time.perf_counter() - started, 1)
        results.append(payload)

        if payload.get("ok"):
            print("[{}] ok  rtf={}x  load={}s  transcribe={}s  peak={} MB".format(
                name, payload["rtf"], payload["load_s"],
                payload["transcribe_s"], payload["peak_rss_mb"]))
        else:
            print("[{}] FAILED  {}".format(name, payload.get("error")))
        sys.stdout.flush()

    return results, {"device": device, "compute": compute, "threads": threads}


def print_table(results, session_minutes):
    print()
    print("=" * 68)
    print("RESULTS  (projected for a {:.0f} min session, beam-5 adjusted)".format(
        session_minutes))
    print("=" * 68)
    header = "{:<18} {:>7} {:>8} {:>9} {:>11} {:>7}".format(
        "model", "rtf", "load s", "peak MB", "projected", "chars")
    print(header)
    print("-" * len(header))

    ok = [r for r in results if r.get("ok")]
    for r in results:
        if not r.get("ok"):
            print("{:<18} {:>7} {:>8} {:>9} {:>11} {:>7}".format(
                r["model"], "-", "-", "-", "FAILED", "-"))
            continue
        rtf = r["rtf"] or 0
        # Benchmarks run at the requested beam size; production uses beam 5,
        # which costs roughly 30% more time.
        proj_min = (session_minutes / rtf) * 1.3 if rtf else 0
        print("{:<18} {:>6.2f}x {:>8.1f} {:>9} {:>9.1f}m {:>7}".format(
            r["model"], rtf, r["load_s"] or 0,
            r["peak_rss_mb"] if r["peak_rss_mb"] else "?",
            proj_min, r["chars"]))

    print()
    for r in results:
        if not r.get("ok") and r.get("error"):
            print("  {} -> {}".format(r["model"], r["error"]))

    if ok:
        print()
        print("Compare the saved transcript-*.txt files side by side. Pick the")
        print("smallest model whose text you would actually trust, not the one")
        print("with the best number.")


def main():
    p = argparse.ArgumentParser(description="Benchmark local Whisper transcription.")
    p.add_argument("--audio", help="path to an audio file")
    p.add_argument("--session", help="path to a session folder; largest audio file is used")
    p.add_argument("--models", help="comma-separated override, e.g. small.en,large-v3")
    p.add_argument("--clip", type=float, default=60.0, help="seconds to transcribe (default 60)")
    p.add_argument("--offset", type=float, default=0.25,
                   help="fraction into the file to sample from (default 0.25)")
    p.add_argument("--beam", type=int, default=1, help="beam size (default 1, fast)")
    p.add_argument("--timeout", type=int, default=1800, help="per-model timeout seconds")
    p.add_argument("--outdir", default="bench-out", help="where to write transcripts and JSON")
    p.add_argument("--probe-only", action="store_true", help="system report only, no downloads")

    p.add_argument("--run-one", help=argparse.SUPPRESS)
    p.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    p.add_argument("--compute", default="int8", help=argparse.SUPPRESS)
    p.add_argument("--threads", type=int, default=4, help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.run_one:
        return run_one(args)

    info = probe()
    print_probe(info)

    if args.probe_only:
        print("Recommended ladder for this machine: {}".format(
            ", ".join(choose_models(info))))
        dev, comp = device_and_compute(info)
        print("Device: {} / {}   threads: {}".format(dev, comp, worker_threads(info)))
        return 0

    if info["faster_whisper"] is None:
        print("faster-whisper is not installed. Run:")
        print("    pip install faster-whisper psutil")
        return 1

    audio = args.audio
    if not audio and args.session:
        audio = find_audio(args.session)
        if not audio:
            print("No audio file found under {}".format(args.session))
            return 1
        print("Using audio: {}".format(audio))
    if not audio:
        print("Give me --audio PATH or --session FOLDER (or use --probe-only).")
        return 1
    if not os.path.isfile(audio):
        print("Not a file: {}".format(audio))
        return 1
    args.audio = os.path.abspath(audio)

    results, cfg = run_bench(args, info)

    session_minutes = 29.0
    for r in results:
        if r.get("audio_total_s"):
            session_minutes = r["audio_total_s"] / 60.0
            break

    print_table(results, session_minutes)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "bench-results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"probe": info, "config": cfg, "results": results,
                   "session_minutes": round(session_minutes, 2)}, fh, indent=2)
    print()
    print("Wrote {}".format(os.path.abspath(out)))
    print("Send me that file and I can pick the default tier for your machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
