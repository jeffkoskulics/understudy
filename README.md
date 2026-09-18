# wfcap — GUI workflow capture for AI teaching materials

Records someone demonstrating a workflow in a GUI application — screen,
clicks, keystrokes and narration, all on one time base — then compresses the
result into something you can paste into a chat session with an LLM.

The compression is the point. A 30-minute recording holds roughly 200,000
tokens of on-screen text. This gets it to about 30,000: one paste, no API key
required.

Built for a specific shape of problem: an application structured like an
operating system, with features spread across many windows, driven by a lot of
manual pointing, clicking and file manipulation — the kind of workflow that is
tedious to document by hand and poorly served by a screen recording nobody
watches.

## How it works

    record  ->  handoff  ->  merge  ->  pack  ->  workflow.md

1. **record** — samples the screen at 2–4 fps and keeps a frame only when
   something meaningful changed. Clicks, typing runs and window switches are
   recorded alongside, sharing a monotonic clock with the audio.
2. **handoff** — compresses the audio and writes a prompt to paste beside it,
   for any chat model that accepts audio uploads.
3. **merge** — takes the reply back, aligned to speech boundaries found
   locally, so narration lines up with what was on screen.
4. **pack** — OCRs only the frames that matter and emits one entry per user
   action, grouped by window.

## Install

Requires Python 3.9+ and, on macOS, the Xcode command line tools.

    python3 -m venv .venv
    ./.venv/bin/pip install -r requirements.txt
    swiftc -O helpers/ocr_mac.swift -o helpers/ocr_mac   # macOS OCR helper

## Status

| Stage | State |
|---|---|
| 1. Recorder (frames, dedup, events, audio) | **working** |
| 2a. OCR (macOS Vision, parallel) | **working** |
| 2b. Transcription (chat hand-off + merge) | **working** |
| 3. Packer (steps, delta text) | **working** |
| 3b. Contact sheets (cropped, annotated) | not started |
| 4. Handoff CLI + paste-back merge | **working** |
| 4b. Handoff GUI | not started |

Tested on macOS (Intel). The Windows backends for window tracking are written
but untested, and the Windows OCR helper is not yet written.

## Run

    ./record.command --out ~/Recordings

Options: `--fps 2` (2–4), `--min-change 0.004`, `--monitor 1`,
`--keys metadata|full`, `--no-audio`, `--duration N`.

## Permissions

macOS needs two grants in System Settings > Privacy & Security, both one-time:

- **Screen Recording** — for frames, and for window *titles*
- **Accessibility** — for clicks and keystrokes

Without Accessibility the recorder still works, but loses click coordinates,
which the packer needs to name the control the user pressed.

Windows needs neither.

## Session layout

    session-YYYY-MM-DD-HHMMSS/
      manifest.json    config, geometry, clock start, audio offset
      frames/000123.jpg
      frames.jsonl     idx, t, duration, reason, change, bbox, app, window,
                       window_id, window_bounds
      events.jsonl     clicks, scrolls, typing runs, structural keys
      audio.wav        16 kHz mono, continuous

Each frame carries a `reason`:

| reason | meaning |
|---|---|
| `window-switch` | the frontmost window changed identity |
| `click` / `post-click` / `input` | the user acted; kept even if nothing visibly changed |
| `window-move` | a drag or resize, after the bounds settled |
| `diff` | enough pixels changed on their own |
| `heartbeat` | nothing changed for a long time; anchors idle stretches |

Window drags and resizes are *suppressed* while in flight and recorded as a
single frame once the bounds hold still, so shuffling windows around does not
flood the session with near-duplicates. `window_id` is recorded so a later
stage can diff a window against the last time that same window was in front,
rather than against whatever unrelated window happened to precede it.

`t` is seconds from session start on a monotonic clock, shared by all three
streams. Every dropped frame extends the previous kept frame's `duration`, so
idle time is represented rather than missing.

## Full pipeline

    python -m wfcap record --out ~/Recordings
    python -m wfcap handoff ~/Recordings/session-...   # audio + prompt
    python -m wfcap merge   ~/Recordings/session-...   # paste the reply
    python -m wfcap pack    ~/Recordings/session-...   # -> workflow.md

### Transcription without an API

`handoff` compresses the audio (`afconvert` on macOS, `ffmpeg` elsewhere --
2.2 MB to 215 KB on the reference session, so 30 minutes lands near 5 MB) and
writes a prompt to paste alongside it.

A chat reply has no timestamps, but narration is only useful lined up against
the steps it describes. So speech boundaries are found locally by energy
(`speech.py`) and the model is asked for exactly one line per segment:

    1| Okay so I'm going to start by opening the terminal.
    2| And then we check whether gh is on the path.

Alignment is then exact by construction. `merge` ignores the prose and code
fences a chat reply comes wrapped in, and reports any segment the model
dropped rather than letting the remaining lines shift onto the wrong steps.

## Pack a session

    ./.venv/bin/python -m wfcap.pack ~/Recordings/session-YYYY-MM-DD-HHMMSS

Writes two files beside the recording:

- `workflow.md` — one entry per user action, grouped by window. This is the
  file to paste into a chat session.
- `steps.json` — the same steps with ids, so an answer citing `S007` can be
  merged back against the frame it came from.

Steps are chosen *before* OCR runs, so only the frames a step cites are
recognised. On the reference session that is 21 frames instead of 52, and
cuts packing from 171 s to 51 s — a bigger saving than parallelism gave.

## Measured on a real 79 s session

52 frames kept from 174 sampled (70% deduplicated), 6 distinct windows.
OCR output, and what each packing stage removes:

| encoding | ~tokens | vs raw | 30-min estimate |
|---|---|---|---|
| raw OCR, every frame | 26,396 | 1.0x | ~203,000 |
| naive frame-to-frame delta | 11,941 | 2.2x | ~92,000 |
| window-keyed delta | 10,315 | 2.6x | ~79,000 |
| + step segmentation | 1,443 | 18x | ~33,000 |
| + crop to change bbox | **1,142** | **23x** | **~26,000** |

Two findings worth keeping:

*Naive delta encoding is unreliable.* It measured 1.0x (no compression at
all) on an earlier session that switched apps constantly, and 2.2x here.
Keying the diff on `window_id` -- comparing a window against the last time
that same window was in front -- is steadier but only ~16% better than naive.
Neither is where the compression comes from.

*The compression comes from step segmentation.* Emitting one entry per user
action rather than per frame is worth ~7x on its own, and it is what makes a
session pasteable. It depends entirely on the input event stream, and so on
the macOS Accessibility grant.

*OCR is the bottleneck and parallelism barely helps.* 4.5 s/frame with one
worker, 3.3 s/frame with four -- Vision appears to be GPU-bound, not CPU-bound.
The real fix is ordering: segment into steps first, then OCR only the frames
a step actually cites (~18 of 52 here), rather than OCRing everything.

## Privacy

Keystrokes are recorded as metadata only (field, duration, character count)
plus structural keys (Tab, Enter, Backspace). `--keys full` records typed
characters and should be treated as sensitive.
