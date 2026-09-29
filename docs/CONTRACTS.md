# Live pipeline contracts

All workstreams build against these contracts. If you need to change one,
say so in your PR description; don't change it silently.

## Time
Every record has `t`: the number of seconds since the session's `Clock` started
(`understudy/clock.py`). Nothing uses wall-clock time, except `system.json`
and the manifest.

## Stage base
`understudy/live/bus.py`: `Stage` (a thread with a bounded inbox that drops
the oldest item when full and counts every drop) and the metrics interface
`timing(name, s)`, `gauge(name, v)`, `count(name, n)`. Stages never raise into
the recorder.

## Session-folder files (all append-only JSONL unless noted)

| file | writer | record |
|---|---|---|
| `frames.jsonl` | capture (existing) | existing fields, plus `mode: "fixed"\|"dedup"`, `reasons: [..]` |
| `audio.wav` | mic, 16 kHz mono (existing) | |
| `audio_system.wav` | loopback, 16 kHz mono | start offset stored in the manifest as `audio_system_offset` |
| `live_transcript.jsonl` | live_transcribe | `{t0, t1, text, source: "mic"\|"system", final: bool, speaker?: str, words?: [{w,t0,t1}]}` |
| `vision.jsonl` | vision | `{t, frame, mode: "describe"\|"diff", prev_frame?, backend, model, latency_s, text, skipped?: reason}` |
| `speakers.jsonl` | diarize | `{t0, t1, source, speaker_id, name?: str, confidence}` |
| `participants.json` (JSON) | user/GUI | `{"speaker_id": "Display Name", ...}`, optional |
| `metrics.jsonl` | metrics | `{t, cpu_pct, cpu_per_core, rss_mb, sys_mem_pct, gpu?: {...}, disk_write_mb_s, counters, gauges, timings: {name: {n, p50, p95, max}}}` at 1 Hz |
| `system.json` (JSON) | metrics | OS, CPU model, cores, RAM, GPU, python, package versions, model names |
| `bundle-manifest.json` | bundle | files, sizes, sha256, time slice, redactions |

## Backends
Both the vision and transcription models go behind a `Backend` with
`name`, `model`, and `run(payload) -> dict`. Only local backends ship now
(Ollama or llama.cpp for vision, faster-whisper for audio). A `remote`
backend slot exists but is not implemented yet, so the stream can later be
pointed at another model without touching the stages.

## Uploads
Uploads go directly to a Cloudflare R2 bucket over its S3-compatible API
(boto3). The upload needs four values: `account_id`, `bucket`,
`access_key_id` and `secret_access_key`. They're read from
`~/.understudy/upload.toml`, or from the environment variables
`UNDERSTUDY_R2_*`. Object key: `bundles/<hostname>/<session>/<slice>/<file>`.

## Privacy
Recording in `meeting` mode shows an always-on-top indicator window. There
is no consent checkbox. Uploads are always an explicit command and support
`--dry-run`, `--no-audio` and `--no-frames`.
