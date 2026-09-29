"""Streaming transcription (workstream C).

Audio arrives via `on_chunk(source, t0, pcm_int16_bytes)` (16 kHz mono; t0 is
the session-clock time of the first sample). One buffer per source. Every
WINDOW seconds of new audio a window (with OVERLAP seconds carried over from
the previous one) goes to the backend. Words in the tail of a window are
written as `final: false` records; words before the tail are final. The next
window re-transcribes the tail, and words already finalized are dropped by
timestamp, so the overlap never duplicates text.

`session` may be a directory path, or an object with `.dir` or `.path`.
Records go to `<dir>/live_transcript.jsonl`.
"""
import array
import json
import os
import threading
import time

from .bus import NullMetrics, Stage

RATE = 16000
WINDOW = 5.0
OVERLAP = 1.0
ENERGY_GATE = 150.0     # RMS of int16 samples below which a window is silence


def rms(pcm):
    a = array.array("h")
    a.frombytes(pcm[:len(pcm) // 2 * 2])
    if not a:
        return 0.0
    # subsample: this is a cheap gate, not a measurement
    a = a[::4] if len(a) > 4000 else a
    return (sum(s * s for s in a) / len(a)) ** 0.5


class _Buf:
    def __init__(self):
        self.data = bytearray()
        self.t0 = None           # session time of data[0]
        self.emitted = 0.0       # session time up to which windows were queued
        self.final_until = 0.0   # words centred before this are already final


class LiveTranscriber(Stage):
    name = "transcribe"

    def __init__(self, clock, session, backend, metrics=None, speaker=None,
                 window=WINDOW, overlap=OVERLAP, gate=ENERGY_GATE,
                 inbox_size=4):
        super().__init__(clock, session, metrics or NullMetrics(), inbox_size)
        self.backend = backend
        self.speaker = speaker
        self.window = window
        self.overlap = overlap
        self.gate = gate
        self._bufs = {}
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        d = session if isinstance(session, str) else (
            getattr(session, "dir", None) or getattr(session, "path", None))
        self.path = os.path.join(d, "live_transcript.jsonl") if d else None

    # -- audio thread: must never block or raise ---------------------------
    def on_chunk(self, source, t0, pcm):
        try:
            with self._lock:
                b = self._bufs.setdefault(source, _Buf())
                if b.t0 is None:
                    b.t0 = t0
                    b.emitted = t0
                b.data += pcm
                end = b.t0 + len(b.data) / 2 / RATE
                if end - b.emitted >= self.window:
                    self._queue(source, b, end)
        except Exception as exc:
            self.error = repr(exc)

    def _queue(self, source, b, end):
        start = max(b.t0, b.emitted - self.overlap)
        i = int(round((start - b.t0) * RATE)) * 2
        j = int(round((end - b.t0) * RATE)) * 2
        pcm = bytes(b.data[i:j])
        b.emitted = end
        # keep only the overlap so memory stays bounded
        keep = int(round((end - self.overlap - b.t0) * RATE)) * 2
        if keep > 0:
            del b.data[:keep]
            b.t0 += keep / 2 / RATE
        self.offer({"source": source, "t0": start, "pcm": pcm, "last": False})

    # -- worker ------------------------------------------------------------
    def step(self, item):
        self._process(item)

    def _process(self, item):
        source, t0, pcm = item["source"], item["t0"], item["pcm"]
        dur = len(pcm) / 2 / RATE
        end = t0 + dur
        b = self._bufs.setdefault(source, _Buf())
        final_to = end if item.get("last") else end - self.overlap
        if rms(pcm) < self.gate:
            b.final_until = max(b.final_until, final_to)
            return
        started = time.perf_counter()
        res = self.backend.run({"pcm": pcm, "sample_rate": RATE, "t0": t0,
                                "source": source})
        took = time.perf_counter() - started
        self.metrics.timing("transcribe.run", took)
        if took > 0:
            self.metrics.gauge("transcribe.rtf", dur / took)
        words = [{"w": w["w"], "t0": round(t0 + w["t0"], 3),
                  "t1": round(t0 + w["t1"], 3)} for w in res.get("words", [])]
        mid = lambda w: (w["t0"] + w["t1"]) / 2
        # dedupe the overlap: drop anything already finalized
        words = [w for w in words if mid(w) > b.final_until]
        fin = [w for w in words if mid(w) <= final_to]
        part = [w for w in words if mid(w) > final_to]
        # final first, so the file reads in time order; a partial is superseded
        # by the next window's final covering the same span
        self._write(source, fin, True)
        self._write(source, part, False)
        b.final_until = max(b.final_until, final_to)

    def _write(self, source, words, final):
        if not words or not self.path:
            return
        t0, t1 = words[0]["t0"], words[-1]["t1"]
        rec = {"t0": t0, "t1": t1, "source": source, "final": final,
               "text": " ".join(w["w"] for w in words).strip(),
               "words": words}
        if self.speaker:
            try:
                name = self.speaker(t0, t1, source)
            except Exception:
                name = None
            if name:
                rec["speaker"] = name
        with self._wlock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def stop(self):
        """Flush each source's remaining audio as final, then stop."""
        super().stop()
        try:
            with self._lock:
                pending = []
                for src, b in self._bufs.items():
                    if b.t0 is None:
                        continue
                    end = b.t0 + len(b.data) / 2 / RATE
                    if end - b.emitted > 0.3:
                        start = max(b.t0, b.emitted - self.overlap)
                        i = int(round((start - b.t0) * RATE)) * 2
                        pending.append((src, start, bytes(b.data[i:])))
            for src, start, pcm in pending:
                self._process({"source": src, "t0": start, "pcm": pcm,
                               "last": True})
        except Exception as exc:
            self.error = repr(exc)
