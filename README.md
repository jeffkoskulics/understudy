# Understudy

**Your expert performs the workflow once. Your trainee gets the script.**

Understudy watches someone do a job in a GUI application — every click, every
keystroke, every window, and what they say while they do it — and turns it into
a written walkthrough an LLM can read.

No API key. No cloud service. No procurement ticket. It runs on the machine,
and nothing leaves it until you decide to paste something.

---

## The problem it solves

Somewhere in your organisation is a person who knows how to do a thing. They
open six windows in some particular order, they know which of the four
identical-looking dialogs is the right one, and they know the one field that
must be filled before the other or it silently fails.

That knowledge is not written down. It probably cannot be written down by the
person who has it, because they stopped noticing the hard parts years ago.

The usual answers do not work:

- **Screen recordings** capture everything and explain nothing. Nobody watches
  a 40-minute video to learn one procedure.
- **Writing documentation** costs the expert a day per workflow, and they
  will not do it.
- **Feeding video to an LLM** is not practical for most people, and a
  30-minute session is roughly 200,000 tokens of on-screen text.

Understudy records the session properly, then compresses it by about 7x until
it fits in a single chat message. The expert does the job once, narrating as
they go. Twenty minutes later there is a draft procedure.

---

## What the output looks like

`workflow.md`, ready to paste into any chat session:

```markdown
## Vault Manager — Batch Import

**S014** [4:12] click "Import from Staging…"
  - said: Always check the staging folder date first, it's the one thing
          that bites people.
  - new: Import from Staging | Source folder | Validate on import | Cancel | OK

**S015** [4:19] type 24 characters over 6.2s
  - said: So that's the batch ID, and it has to match the folder name exactly.
  - new: /vault/staging/2026-Q1-B

**S016** [4:31] click "Validate on import"
  - said: People skip this and then spend an afternoon working out why the
          totals are off.

**S017** [4:36] click "OK"
  - new: Importing 1,284 records… | Elapsed 0:03 | Cancel
```

*(Illustrative — the format is real, the application is not.)*

Every step is one user action, tagged with an id, the control that was
clicked, what appeared on screen as a result, and what the expert said while
doing it. Hand that to an LLM and ask for a training guide, a checklist, an
SOP, or a list of the steps most likely to trip up a new hire.

---

## Quickstart

### macOS / Linux

```bash
curl -fsSL https://raw.githubusercontent.com/jeffkoskulics/understudy/main/install.sh | sh
```

That clones the repo into `./understudy`, creates a virtualenv, installs the
dependencies, and on macOS builds the Vision OCR helper. From an existing
clone, run `sh install.sh` instead.

### Windows

Open PowerShell and run:

```powershell
iwr -useb https://raw.githubusercontent.com/jeffkoskulics/understudy/main/install.ps1 | iex
```

Or download [install.ps1](https://raw.githubusercontent.com/jeffkoskulics/understudy/main/install.ps1),
right-click it and choose **Run with PowerShell**. If PowerShell refuses to run
the script, allow it for that one session:

```powershell
Set-ExecutionPolicy -Scope Process -Bypass
```

You need [Python 3.9+](https://www.python.org/downloads/windows/) (tick *Add
python.exe to PATH* in its installer) and [Git](https://git-scm.com/download/win).

The commands below are written for a Unix shell. On Windows use
`bin\understudy.cmd` in place of `./bin/understudy` — the `./.venv/bin/...`
paths in older instructions do not exist on Windows, where the venv puts its
executables in `.venv\Scripts\`, and `swiftc` is macOS-only.

**Windows is partly supported today:** recording, window tracking and audio
work, but the OCR helper is macOS-only, so `pack` cannot yet read text off the
frames on Windows.

Then open the control panel:

```bash
./bin/understudy gui
```

It asks the two questions that matter — **which monitor** to record and **where
to put the files** — and shows a thumbnail of the chosen screen, because two
monitors of the same size are indistinguishable from their geometry alone and
picking the wrong one is only discovered after the expert has finished
narrating.

After that it is one button per step, in the order they happen:

| button | what it does |
|---|---|
| **Start / Stop recording** | the same capture the CLI runs, with a live frame count |
| **Transcribe on this machine** | runs whisper locally, if it is installed |
| **Prepare audio + prompt** | finds the speech segments and compresses the audio |
| **Copy prompt** | puts the transcription prompt on the clipboard |
| **Show audio file** | reveals the audio in the file manager, to drag into the chat |
| **Merge reply** | takes the pasted reply back |
| **Build workflow.md** | runs the OCR and packing |
| **Copy workflow.md** | puts the finished walkthrough on the clipboard |

The hand-off buttons copy to the clipboard rather than opening anything,
because the destination is a chat window in a browser: there is nothing to
open, only something to paste. Audio is the exception — it has to be dragged
in — so that button reveals the file instead.

Buttons that cannot work yet are greyed out, so the order is never in doubt.
The GUI is Tkinter, which ships with Python on all three platforms, so it adds
nothing to `requirements.txt`.

---

### Or from the command line

Record a workflow, narrating as you go:

```bash
./bin/understudy record --out ~/Recordings
```

```
Recording -> ~/Recordings/session-2026-09-18-1432
Press Ctrl-C to stop.
  3:41  62 frames kept / 441 sampled
```

Turn it into something pasteable:

```bash
./bin/understudy transcribe ~/Recordings/session-2026-09-18-1432  # whisper, here
./bin/understudy pack       ~/Recordings/session-2026-09-18-1432  # -> workflow.md
```

Or, without installing a model, hand the audio to a chat window instead:

```bash
./bin/understudy handoff ~/Recordings/session-2026-09-18-1432   # audio + prompt
./bin/understudy merge   ~/Recordings/session-2026-09-18-1432   # paste the reply
./bin/understudy pack    ~/Recordings/session-2026-09-18-1432   # -> workflow.md
```

`transcribe` runs whisper on this machine and writes the transcript directly.
`handoff` gives you a compressed audio file to drag into a chat session and a
prompt to paste beside it, and `merge` takes the reply back. Both write the
same `transcript.json`, so `pack` cannot tell which route produced it. If you
have API access, skip all of them and write that file yourself.

Both paths write the same session folder, so you can start a recording in the
GUI and pack it from a script, or the other way round.

---

### Classifying what the user was doing

`activity` labels the session without looking at the frames at all:

```bash
./bin/understudy activity ~/Recordings/session-2026-09-18-1432
```

```
A001  0:00-0:08  navigating             Mail - Inbox - Mail
A002  0:08-0:25  reading                Mail - Inbox - Mail
A003  0:25-0:44  form-filling           Excel - budget.xlsx - Excel
A004  0:44-1:20  idle                   Excel - budget.xlsx - Excel
A005  1:20-1:49  typing                 Editor - notes.md - Editor
```

The unit is a stretch of work, not a frame -- a user is not doing a different
thing every 500 ms, and hundreds of images an hour is the wrong granularity to
label. A stretch ends when the user arrives in a different window, stops for
long enough that the activity has clearly ended, or switches the kind of input
they are giving. Each one is then labelled from the event stream and the frame
metadata: `typing`, `form-filling`, `editing`, `reading`, `navigating`,
`app-switching`, `arranging-windows`, `watching`, `waiting`, `idle`.

This is rules over numbers that were already recorded, not a model, and that
is deliberate: a screenshot of a spreadsheet looks identical whether it is
being read or filled in, and the click and keystroke record is what tells them
apart. Every segment carries a `why` saying which numbers produced its label,
so a label that looks wrong names the rule to argue with. Results go to
`activity.jsonl`, and `pack` folds them into `workflow.md` if they are there.

That says what *kind* of work a stretch was, not what the work was *about*.
For that, `--embed` runs a small sentence model (about 90 MB, CPU, local) over
the text the session already contains -- window titles, the OCR delta, the
narration -- and names each segment:

```bash
./bin/understudy activity --install                     # one-time, ~a few hundred MB
./bin/understudy activity <session> --embed             # discover the activities
./bin/understudy activity <session> --embed \
    --labels "email triage,data entry,web research"     # or match a fixed set
```

Without `--labels` it clusters the segments and names each cluster by the
terms that distinguish it, which is how you find out what your label set
should be. With `--labels` it matches each segment against labels you have
settled on, by cosine similarity, so adding one costs nothing and needs no
training data. A segment that matches nothing well keeps its rule-based label
rather than being given a confident wrong name.

Both write into the same `activity.jsonl`: `label` is what the input stream
proves, `name` is what the text suggests, and the two disagreeing is worth
looking at. Run `transcribe` and `pack` first if you can -- the embedding
stage reads their output, and window titles alone are thin.

No image model is involved at any point. Embedding text that OCR has already
extracted is far cheaper than captioning frames, and on screenshots it is also
more accurate: CLIP-class models are weak on fine UI detail and on reading
text in an image, which is exactly what the Vision helper is good at.

---

## Why not just record the screen

Because the interesting thing is not the pixels.

**It keeps frames that matter.** Sampling runs at 2–4 fps, but a frame is only
kept when something meaningful changed — measured against the last *kept*
frame, so a slow redraw cannot creep past the threshold. A blinking cursor
scores 0.0005 and is dropped. A dialog opening scores 0.088 and is kept.
Typical sessions discard 70–90% of samples.

**It knows a window move from a window change.** Dragging a window redraws most
of the screen and means nothing. Understudy suppresses the drag and records one
frame where the window landed.

**It reads the clicks, not just the screen.** A click on a button that does not
visibly change is invisible to pixel diffing. Understudy records the
coordinate, then finds the OCR text under it — so the step says
`click "Validate on import"`, not `click at (412, 388)`.

**It uses the window title as structure.** In applications built like an
operating system — features scattered across dozens of windows — the title bar
announces each step for free, with no OCR at all.

---

## The compression, measured

From a real session, per 30 minutes of recording:

| stage | ~tokens |
|---|---|
| raw OCR of every frame | 203,000 |
| naive frame-to-frame delta | 92,000 |
| window-keyed delta | 79,000 |
| **+ step segmentation** | **33,000** |
| **+ crop to changed region** | **26,000** |

Most of the win is step segmentation: one entry per user action instead of one
per frame. Delta encoding measured between 1.0x and 2.6x depending on the
session and is not something to rely on.

Steps are chosen *before* OCR runs, so only frames a step actually cites get
recognised — on the reference session, 21 frames instead of 52, cutting
processing from 171s to 51s.

---

## Privacy

This matters if you are recording a colleague at work, and it is deliberate:

- **Keystrokes are not logged.** A typing run is recorded as *what field, how
  long, how many characters* — not the characters. Typed values usually come
  back from OCR of the resulting screen anyway. `--keys full` records the
  characters, and should be treated as sensitive.
- **Nothing is uploaded.** Capture, OCR and packing are entirely local. The
  only data that leaves the machine is what you choose to paste.
- **The output is reviewable text.** Before sharing a `workflow.md`, you can
  read every line of it. That is not true of a video.
- **Recordings stay put.** Sessions are written to a folder you name and are
  never touched again.

---

## Requirements

Python 3.9+, and on macOS the Xcode command line tools for the OCR helper.

On-device transcription is optional and installed separately, because
CTranslate2 and its wheels are a few hundred MB that the chat hand-off path
does not need:

```bash
bin/understudy transcribe --install
```

That installs it into the interpreter Understudy actually runs -- the project
venv -- which a bare `pip install -r requirements-whisper.txt` from another
shell may not. The GUI offers the same thing the first time you press
**Transcribe on this machine**, and `install.sh --with-whisper` (PowerShell:
`.\install.ps1 -WithWhisper`) pulls it in at install time.

Semantic activity labels are optional in the same way, and separate again
because they pull in torch:

```bash
understudy activity --install
```

The models are downloaded on first use and cached in
`~/.cache/huggingface`; after that they run offline like everything else.
`base` is the default and is roughly realtime on a laptop CPU; `small.en` is
noticeably better on English narration and about three times slower.

The GUI needs Tkinter, which is part of the standard library on macOS and
Windows. Some Linux distributions package it separately (`apt install
python3-tk`); the command line works without it.

macOS needs two one-time grants in System Settings → Privacy & Security:

| grant | needed for |
|---|---|
| **Screen Recording** | frames, and window titles |
| **Accessibility** | clicks and keystrokes |

Without Accessibility, recording still works but loses click coordinates — and
with them the ability to name the control that was pressed, which is where most
of the value is.

---

## Status

| stage | state |
|---|---|
| Recorder — frames, dedup, events, audio | working |
| OCR — macOS Vision | working |
| Transcription — chat hand-off and merge | working |
| Transcription — on-device, faster-whisper | working |
| Activity labels — rules over events and frame metadata | working |
| Activity labels — local text embeddings, clustered or zero-shot | working |
| Activity labels — local vision model over the frames | not started |
| Packer — steps, delta text | working |
| Contact sheets — cropped, annotated screenshots | not started |
| Windows OCR helper | not started |
| GUI — monitor picker, destination, copy buttons | working |

Tested on macOS (Intel); the GUI was additionally exercised end to end on
Linux. The Windows backends for window tracking are written but untested.

This is early. It works end to end and the numbers above are real measurements,
not projections, but it has been exercised on a handful of sessions rather than
a hundred. Issues and recordings that break it are welcome.

---

## Licence

MIT. See [LICENSE](LICENSE).
