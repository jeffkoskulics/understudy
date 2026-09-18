# tools/bench_transcribe.py

Standalone harness that answers one question: **which local Whisper model
should understudy default to on this machine?**

It is deliberately a single file with no imports from the `understudy`
package, so it can be downloaded on its own and run anywhere.

## Quick start

Safe first step -- reports hardware only, downloads nothing:

```
python tools/bench_transcribe.py --probe-only
```

Then the real run against a recorded session:

```
pip install faster-whisper psutil
python tools/bench_transcribe.py --session "C:\Users\jeffk\Recordings\session-2026-09-18-093421"
```

Outputs land in `bench-out/`:

- `bench-results.json` -- full machine probe plus per-model timings
- `transcript-<model>.txt` -- one per model, for side-by-side quality comparison

## Useful flags

| Flag | Purpose |
|---|---|
| `--probe-only` | Hardware report, no model downloads |
| `--audio PATH` | Point at one audio file directly |
| `--session DIR` | Auto-pick the largest audio file in a session folder |
| `--models a,b` | Override the ladder, bypassing the RAM gate |
| `--clip 60` | Seconds of audio to transcribe per model |
| `--offset 0.25` | How far into the file to sample from |
| `--beam 1` | Beam size; production uses 5 |
| `--timeout 1800` | Per-model timeout in seconds |

## Design notes

**Measured, not predicted.** Spec-sheet estimates fail on laptops because of
thermal throttling, background AV scans, and whatever else is running. The
harness transcribes a real clip and extrapolates from the observed rate.

**Sampled from the middle.** The opening minute of a meeting recording is
usually silence and microphone checks. Default is to sample 25% into the
file.

**Isolated subprocesses.** Each model runs in its own process with a
timeout. An OOM kill, a native crash, or a blocked download takes out one
trial and leaves the rest of the run intact.

**Load time is measured separately** from transcribe time. It is a one-time
per-run cost that includes the first-run weight download, and folding it
into the rate would badly distort short-clip measurements.

**One core left free.** `cpu_threads` defaults to physical cores minus one,
so the machine stays usable while a benchmark runs.

**RAM-gated ladder.** Models whose memory footprint exceeds available RAM
are skipped rather than attempted. `large-v3` is skipped on CPU by default
-- it runs near 1x realtime, so a 30-minute session takes 30+ minutes.
`--models` overrides both gates.

## Known failure mode: blocked model weights

First run fetches weights from `huggingface.co`. Corporate networks often
block or TLS-inspect that host, which surfaces as a per-model
`LocalEntryNotFoundError` / 403. That is a network policy problem, not a
bug in the harness.

Workarounds: pre-seed a model cache on an unrestricted machine and point
`HF_HOME` at it, then set `HF_HUB_OFFLINE=1`.

## Status

Experimental, on the `bench/transcription-harness` branch. Either folds
into install as a capability check with a "test my machine" button, or gets
deleted once the default tier is settled.
