"""Transcribe the session narration on this machine, with faster-whisper.

The hand-off path in `handoff.py` asks a chat LLM for one line per speech
segment, because alignment has to be exact and a chat reply carries no
timings. Running a model locally removes the paste step, but the alignment
problem does not go away -- it just moves. Whisper returns its own segment
boundaries, and they do not agree with the ones `speech.py` found by energy:
whisper splits on grammar, the energy pass splits on silence.

So the boundaries stay authoritative and whisper is asked for word timings
instead. The audio is decoded in one pass -- not clipped per segment, which
would strip the surrounding context the model needs to get boundary words
right -- and each word is then dropped into the span it overlaps. Alignment
stays exact by construction, exactly as it is on the hand-off path, and both
routes write the same `transcript.json`.

The model runs on the machine and nothing leaves it, which is the point.
"""
import json
import os
import subprocess
import sys

from . import speech

DEFAULT_MODEL = "base"
NEAR = 0.75       # a stray word this close to a span still belongs to it

REQUIREMENTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "requirements-whisper.txt")


class TranscribeError(RuntimeError):
    pass


def is_installed():
    """Is faster-whisper importable in the interpreter running us?"""
    try:
        import importlib.util
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def install_command():
    """The pip command that installs it into *this* interpreter. -> argv list

    `sys.executable` matters more than it looks. The launcher in `bin/` runs
    the project venv, so a bare `pip` on PATH is usually a different Python
    altogether -- installing there leaves the import failing exactly as
    before, which is the confusing half of the usual report.
    """
    if os.path.exists(REQUIREMENTS):
        return [sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS]
    # Installed away from the clone: the pin lives in the file, so name the
    # package directly rather than pointing at a path that is not there.
    return [sys.executable, "-m", "pip", "install", "faster-whisper>=1.1"]


def install(progress=None):
    """Install faster-whisper into this interpreter. -> None"""
    cmd = install_command()
    if progress:
        progress("Installing faster-whisper (a few hundred MB)...")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = (proc.stdout or b"").decode("utf-8", "replace").strip()
        tail = "\n".join(tail.splitlines()[-10:])
        raise TranscribeError(
            "Installing faster-whisper failed (pip exited %d). Run it by hand "
            "to see the whole log:\n    %s\n\n%s"
            % (proc.returncode, " ".join(cmd), tail))
    if not is_installed():
        raise TranscribeError(
            "pip reported success but faster-whisper still does not import. "
            "Try again by hand:\n    %s" % " ".join(install_command()))


def missing_message():
    return ("faster-whisper is not installed. It is optional, because it "
            "pulls in a few hundred MB that the hand-off path does not "
            "need. Install it with:\n"
            "    understudy transcribe --install\n"
            "which is the same as:\n"
            "    %s\n"
            "Or keep using `understudy handoff` instead."
            % " ".join(install_command()))


def _load_model(name, device, compute_type):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscribeError(missing_message())
    try:
        return WhisperModel(name, device=device, compute_type=compute_type)
    except Exception as exc:
        raise TranscribeError(
            "Could not load the '%s' model (%s).\n"
            "The first run downloads it, so this needs network access once; "
            "afterwards it is cached in ~/.cache/huggingface and runs "
            "offline." % (name, exc))


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _distance(word, seg):
    mid = (word["start"] + word["end"]) / 2.0
    if seg["start"] <= mid <= seg["end"]:
        return 0.0
    return min(abs(mid - seg["start"]), abs(mid - seg["end"]))


def assign(segs, words, near=NEAR):
    """Bucket timed words into the speech spans. -> ({span index: text}, dropped)

    Most words land inside a span and the overlap test settles it. The
    interesting case is a word that lands in none: the energy pass discards
    anything under `MIN_SPEECH`, so whisper does sometimes hear a short word
    where `speech.py` heard a cough. Attaching it to the nearest span keeps it
    in the narration; only words stranded well clear of any span are dropped,
    and those are reported rather than silently lost.
    """
    buckets = {s["i"]: [] for s in segs}
    dropped = []
    for word in words:
        best, best_overlap = None, 0.0
        for seg in segs:
            ov = _overlap(word["start"], word["end"], seg["start"], seg["end"])
            if ov > best_overlap:
                best, best_overlap = seg, ov
        if best is None and segs:
            nearest = min(segs, key=lambda s: _distance(word, s))
            if _distance(word, nearest) <= near:
                best = nearest
        if best is None:
            dropped.append(word)
            continue
        buckets[best["i"]].append(word["word"])
    text = {i: "".join(parts).strip() for i, parts in buckets.items()}
    return text, dropped


def words_from(model, path, language=None):
    """Decode the whole file once and flatten it to timed words.

    `vad_filter` stays off deliberately. Whisper's own voice detection would
    re-cut the audio and shift every timestamp relative to the frames;
    `speech.py` has already done that job against the session clock.
    """
    segments, info = model.transcribe(path, word_timestamps=True,
                                      vad_filter=False, language=language)
    words = []
    for seg in segments:            # a generator: this is where decoding happens
        for w in (seg.words or []):
            words.append({"start": w.start, "end": w.end, "word": w.word})
    return words, info


def transcribe(session_dir, model=DEFAULT_MODEL, device="auto",
               compute_type="int8", language=None, progress=None):
    """Transcribe a recorded session in place. -> info dict

    Writes `transcript.json` in the same shape `handoff.merge` writes, so
    `pack` cannot tell which route produced it.
    """
    manifest_path = os.path.join(session_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise TranscribeError("No manifest.json in %s -- is that a session "
                              "directory?" % session_dir)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    audio = manifest.get("audio") or {}
    path = os.path.join(session_dir, audio.get("path", "audio.wav"))
    if not os.path.exists(path):
        raise TranscribeError("No audio in this session (%s). Nothing to "
                              "transcribe." % os.path.basename(path))

    segs = speech.analyse(session_dir)
    if not segs:
        return {"segments": 0, "written": 0, "empty": [], "dropped": 0,
                "language": None, "speech_seconds": 0.0}

    if progress:
        progress("Loading the %s model..." % model)
    whisper = _load_model(model, device, compute_type)
    if progress:
        progress("Transcribing %.0fs of narration..."
                 % sum(s["duration"] for s in segs))

    words, info = words_from(whisper, path, language)

    # speech.py reports span times on the session clock; whisper's are
    # relative to the start of the file. Without this the whole transcript
    # slides against the frames by however long the microphone took to open.
    offset = audio.get("start_offset") or 0.0
    for w in words:
        w["start"] += offset
        w["end"] += offset

    text, dropped = assign(segs, words)
    out = [{"start": s["start"], "end": s["end"], "text": text[s["i"]]}
           for s in segs if text[s["i"]]]
    with open(os.path.join(session_dir, "transcript.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    return {"segments": len(segs), "written": len(out),
            "empty": [s["i"] for s in segs if not text[s["i"]]],
            "dropped": len(dropped),
            "language": getattr(info, "language", None),
            "speech_seconds": round(sum(s["duration"] for s in segs), 1)}


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="understudy transcribe",
                                 description="Transcribe session narration "
                                             "locally with faster-whisper.")
    ap.add_argument("session", nargs="?",
                    help="the session directory to transcribe; optional with "
                         "--install, which can be run on its own")
    ap.add_argument("--install", action="store_true",
                    help="install faster-whisper into this interpreter first")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="whisper model: tiny, base, small, medium, large-v3, "
                         "or an English-only variant such as base.en, which is "
                         "more accurate on English at the smaller sizes "
                         "(default: %(default)s)")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--compute-type", default="int8",
                    help="int8 is the safe default; try float16 on a GPU")
    ap.add_argument("--language", default=None,
                    help="force a language code instead of detecting it")
    args = ap.parse_args(argv)

    if args.install and not is_installed():
        try:
            install(progress=lambda m: print(m))
        except TranscribeError as exc:
            print(str(exc))
            return 1
        print("faster-whisper is installed.")
    elif args.install:
        print("faster-whisper is already installed.")

    if not args.session:
        if args.install:
            return 0
        ap.error("a session directory is required")

    try:
        info = transcribe(args.session, model=args.model, device=args.device,
                          compute_type=args.compute_type,
                          language=args.language,
                          progress=lambda m: print(m))
    except TranscribeError as exc:
        print(str(exc))
        return 1

    if not info["segments"]:
        print("No narration found in this session.")
        return 0
    print("Transcribed %d of %d segments (%.0fs of narration%s)."
          % (info["written"], info["segments"], info["speech_seconds"],
             ", " + info["language"] if info["language"] else ""))
    if info["empty"]:
        # Same contract as the hand-off path: a gap is narration that is
        # absent, not narration shifted onto a neighbouring step.
        print("Silent segments: %s" % ", ".join(map(str, info["empty"])))
    if info["dropped"]:
        print("%d words fell outside every segment and were dropped."
              % info["dropped"])
    print("Now run:  understudy pack %s" % args.session)
    return 0
