"""Find speech segments in the session audio, locally.

The transcript comes back from a chat session as plain prose with no timings,
but the packer has to interleave narration with steps that are timestamped to
the tenth of a second. Rather than guess where words fall, the boundaries are
found here -- where speech starts and stops is an energy question, not a
language one, and needs no model to answer.

The LLM is then asked for exactly one line per segment. Alignment becomes
exact by construction instead of an estimate, and a miscount is detectable
rather than silent.
"""
import json
import os
import wave

import numpy as np

FRAME_MS = 30
MIN_SPEECH = 0.35     # shorter than this is a cough, a click, a chair creak
MAX_GAP = 0.6         # pauses shorter than this stay inside one segment
MAX_SEGMENT = 30.0    # long monologues are split; a wall of text aligns poorly
NOISE_PERCENTILE = 20
SPEECH_FACTOR = 3.0   # speech is this much louder than the room


def read_wav(path):
    with wave.open(path) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data.astype(np.float32), sr


def frame_rms(data, sr):
    n = max(1, int(sr * FRAME_MS / 1000))
    usable = len(data) // n * n
    if usable == 0:
        return np.zeros(0), n / sr
    frames = data[:usable].reshape(-1, n)
    return np.sqrt((frames ** 2).mean(axis=1)), n / sr


def segments(path, offset=0.0):
    """Return [{start, end, duration}] in session time.

    `offset` is the session time of the first audio sample: the recorder starts
    the microphone a moment after the clock, and ignoring that would slide the
    whole transcript relative to the frames.
    """
    data, sr = read_wav(path)
    rms, step = frame_rms(data, sr)
    if rms.size == 0:
        return []

    floor = np.percentile(rms, NOISE_PERCENTILE)
    # A silent room has a floor near zero, where a multiplicative threshold
    # collapses; the absolute term keeps it meaningful.
    threshold = max(floor * SPEECH_FACTOR, 60.0)
    loud = rms > threshold

    out = []
    start = None
    gap = 0.0
    for i, is_loud in enumerate(loud):
        t = i * step
        if is_loud:
            if start is None:
                start = t
            gap = 0.0
        elif start is not None:
            gap += step
            if gap >= MAX_GAP:
                out.append((start, t - gap))
                start = None
                gap = 0.0
    if start is not None:
        out.append((start, len(loud) * step))

    result = []
    for s, e in out:
        if e - s < MIN_SPEECH:
            continue
        while e - s > MAX_SEGMENT:
            result.append((s, s + MAX_SEGMENT))
            s += MAX_SEGMENT
        result.append((s, e))
    return [{"i": n + 1, "start": round(s + offset, 2), "end": round(e + offset, 2),
             "duration": round(e - s, 2)}
            for n, (s, e) in enumerate(result)]


def analyse(session_dir):
    manifest = json.load(open(os.path.join(session_dir, "manifest.json")))
    audio = manifest.get("audio") or {}
    path = os.path.join(session_dir, audio.get("path", "audio.wav"))
    if not os.path.exists(path):
        return []
    segs = segments(path, audio.get("start_offset") or 0.0)
    with open(os.path.join(session_dir, "speech.json"), "w") as fh:
        json.dump(segs, fh, indent=1)
    return segs
