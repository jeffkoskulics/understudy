"""Prepare a session for hand-off to a chat LLM, and merge the answer back.

Most users reach a model by pasting into a chat window, so the session has to
leave here as things a person can drag and paste, and come back as text this
program can trust.

The awkward part is timing. A chat session returns prose with no timestamps,
but narration is only useful if it lines up with the steps it describes. So
the speech boundaries are found locally (see speech.py) and the model is asked
to fill in exactly one line per segment. Alignment is then exact by
construction, and a miscount shows up as a validation error instead of a
transcript that silently drifts out of sync with the screen.
"""
import json
import os
import re
import shutil
import subprocess
import sys

from . import speech

SEGMENT_RE = re.compile(r"^\s*(\d+)\s*\|\s*(.*)$")


def export_audio(session_dir):
    """Compress the session audio to something a chat window will accept.

    16 kHz mono WAV is about 2 MB a minute, which is past most upload limits
    for a long session and is wasteful regardless. Both encoders used here ship
    with the OS or are already present; nothing is downloaded.
    """
    src = os.path.join(session_dir, "audio.wav")
    if not os.path.exists(src):
        return None
    dst = os.path.join(session_dir, "audio.m4a")
    if sys.platform == "darwin" and shutil.which("afconvert"):
        cmd = ["afconvert", "-f", "m4af", "-d", "aac", "-b", "24000", src, dst]
    elif shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
               "-c:a", "aac", "-b:a", "24k", "-ac", "1", dst]
    else:
        return src  # no encoder available; hand over the WAV and say so
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return src
    return dst


def transcription_prompt(segs, session_name=""):
    lines = []
    lines.append("You are transcribing narration recorded while someone "
                 "demonstrated a software workflow.")
    lines.append("")
    lines.append("The audio contains %d separate stretches of speech, already "
                 "located by timing. Transcribe each one." % len(segs))
    lines.append("")
    lines.append("Reply with EXACTLY %d lines, in this format, and nothing "
                 "else before or after them:" % len(segs))
    lines.append("")
    lines.append("```")
    lines.append("1| <what is said in the first stretch of speech>")
    lines.append("2| <the second>")
    lines.append("```")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Exactly %d numbered lines. If a stretch is inaudible, "
                 "write `1| [inaudible]` rather than skipping it, because the "
                 "line numbers are matched to timings." % len(segs))
    lines.append("- Keep each line to what is said in that stretch; do not "
                 "merge or reorder them.")
    lines.append("- Transcribe verbatim. Keep filenames, menu names and "
                 "technical terms exactly as spoken; do not tidy them up.")
    lines.append("- No commentary, no summary, no timestamps.")
    lines.append("")
    lines.append("For reference, the stretches last (in seconds):")
    lines.append(", ".join("%d:%.1fs" % (s["i"], s["duration"]) for s in segs))
    return "\n".join(lines) + "\n"


def parse_transcript(text, expected):
    """Parse a pasted reply into {segment index: text}.

    Chat replies arrive wrapped in prose and code fences no matter how firmly
    the prompt asks otherwise, so anything that is not a numbered line is
    ignored rather than treated as an error.
    """
    found = {}
    for raw in text.splitlines():
        m = SEGMENT_RE.match(raw.strip().lstrip("`"))
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= expected:
            found[n] = m.group(2).strip()
    missing = [n for n in range(1, expected + 1) if n not in found]
    return found, missing


def prepare(session_dir):
    """Write everything the user needs to hand off, and say what to do."""
    segs = speech.analyse(session_dir)
    audio = export_audio(session_dir)
    prompt_path = os.path.join(session_dir, "transcribe-prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        fh.write(transcription_prompt(segs, os.path.basename(session_dir)))
    return {"segments": len(segs), "audio": audio, "prompt": prompt_path,
            "speech_seconds": round(sum(s["duration"] for s in segs), 1)}


def merge(session_dir, text):
    """Merge a pasted reply into the session. Returns (written, missing)."""
    segs = json.load(open(os.path.join(session_dir, "speech.json")))
    found, missing = parse_transcript(text, len(segs))
    out = []
    for s in segs:
        if s["i"] in found and found[s["i"]]:
            out.append({"start": s["start"], "end": s["end"],
                        "text": found[s["i"]]})
    path = os.path.join(session_dir, "transcript.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    return len(out), missing
