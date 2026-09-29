# Plan: live analysis of demos, student sessions, and meetings

## Goal

Generalize Understudy from "an expert records one workflow" to any session
where application content is on screen: a teacher demoing, a student working,
or a video meeting with shared screens. It keeps running on the user's own
machine (installed from GitHub) and adds three things:

1. **Live analysis.** A fixed 2–4 fps frame stream plus audio. Audio goes to
   local streaming transcription. Frames go to a vision-capable LLM that
   describes each frame, or what changed since the last one.
2. **Instrumentation.** Record CPU, memory, GPU, disk, queue depth and model
   latency throughout the session, so we learn what real hardware needs.
3. **Diagnostic upload.** With explicit opt-in, send a sample bundle (raw
   frames, audio, transcript, LLM outputs, metrics) to an endpoint we control
   so we can inspect inputs against results.

## Pipeline

```
 screen ─► capture (fixed N fps + existing dedup flags) ─► frame queue ─► vision worker ─► frames.jsonl (descriptions/diffs)
 mic + system audio ─► audio ring buffer ─► streaming whisper ─► transcript.jsonl
                                  │
 metrics sampler (1 Hz: cpu, rss, gpu, queue lag, model latency) ─► metrics.jsonl
                                  │
 session folder ─► `understudy bundle` (redact/trim) ─► `understudy upload` ─► endpoint (Worker + R2)
```

Every stage writes append-only JSONL keyed on the existing session clock, so
live output and post-hoc `pack` stay aligned.

## Workstreams (each one can go to its own Sonnet agent)

File ownership is split so agents don't collide. Only the integrator edits
`record.py`, `__main__.py` and `gui.py`.

| # | Workstream | Owns | Notes |
|---|---|---|---|
| A | **Capture modes** | `capture.py`, new `profiles.py` | Add a `fixed` mode that emits every frame at 2–4 fps while keeping the dedup reason as a tag. Add profiles `teacher`, `student`, `meeting` (meeting assumes window-switch signals are weak and the shared screen is the content). |
| B | **Meeting audio** | `audio.py`, new `loopback.py` | Capture system/loopback audio as well as the mic (WASAPI loopback on Windows, BlackHole/ScreenCaptureKit on macOS, PulseAudio monitor on Linux). Two channels or two files, both clock-aligned. |
| C | **Streaming transcription** | new `live_transcribe.py` | faster-whisper over a sliding window (about 5 s chunks with overlap) that emits partial and final segments. The existing batch `transcribe.py` stays as the accurate pass. |
| D | **Vision analysis** | new `vision.py`, `vision_backends/` | Pluggable backend: local (Ollama or llama.cpp running Qwen2.5-VL / Gemma 3) and optionally a cloud API. Modes: `describe` and `diff` (previous frame + current frame + bounding box from capture). Bounded queue that drops or skips frames under load and logs every drop. |
| E | **Instrumentation** | new `metrics.py` | psutil sampler thread plus optional GPU sampling (NVML / `powermetrics`), plus a timing context manager other modules can import. Writes `metrics.jsonl` and a machine profile `system.json` (CPU model, cores, RAM, GPU, OS, model names, versions). |
| F | **Bundle + upload client** | new `bundle.py`, `upload.py` | Package a session or a time slice, optionally downsample, add a manifest with hashes, and upload in chunks with resume. Consent prompt and a `--dry-run` that lists what would be sent. |
| G | **Ingest endpoint** | new `server/` (Cloudflare Worker + R2 + D1 index) | Authenticated upload (per-install token), multipart to R2, one D1 row per bundle, and a small listing page for us to review. |
| H | **Integrator** | `record.py`, `__main__.py`, `gui.py`, `install.*`, `requirements-*.txt`, README | Wire A–F into `understudy live`, add GUI toggles and extend the installers. Runs last, or continuously once interfaces are stubbed. |

**Order:** first I write the shared interfaces (the JSONL schemas, a
`Stage` protocol and the metrics hook), about half an hour of work. Then A–G
run in parallel and H follows. Each agent works in its own worktree and
branch, adds tests for its module, and opens a PR.

## Decisions needed before I launch agents

1. **Vision model:** local only (Ollama with Qwen2.5-VL 7B needs a GPU or
   Apple Silicon to keep up with 2 fps; CPU-only will not) or a cloud API
   allowed too? Realistically, local plus CPU means analysing roughly one
   frame every few seconds, not 2–4 per second.
2. **"Audio-capable" LLM:** is local whisper enough for audio, or do you
   also want audio clips sent to a multimodal model?
3. **Meetings and consent:** recording other participants' audio and screens
   is subject to all-party consent laws in some places. I propose a
   visible recording indicator, a consent checkbox, and uploads that are
   opt-in per session, with an option to strip audio or faces.
4. **Endpoint hosting:** a Cloudflare Worker + R2 on your account (the
   connector is already attached here)? And how should the upload token
   reach testers?
