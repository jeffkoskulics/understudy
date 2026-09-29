"""Speaker attribution for live sessions.

Tier 1 (always): by channel. mic -> "local", system -> "remote".
Tier 2 (optional): system-channel voices are embedded per ~1.5 s voiced window
and clustered online by cosine similarity into S1, S2, ... The mic stays
"local". Embedding backends: sherpa-onnx (set UNDERSTUDY_SPEAKER_MODEL to an
.onnx speaker model, e.g. 3dspeaker / wespeaker) or resemblyzer.

Records carry ids only; names come from participants.json at read time.
"""
import os
import threading

import numpy as np

from understudy import participants
from understudy.live.bus import Stage
from understudy.session import JsonlWriter

SR = 16000
WINDOW_S = 1.5
VOICED_RMS = 300.0        # int16 RMS below this counts as silence
DEFAULT_THRESHOLD = 0.6   # cosine similarity below this starts a new speaker
MAX_RECORDS = 5000


def default_embed_fn():
    """Best available embedding function (float32 mono 16 kHz -> vector), or None."""
    model = os.environ.get("UNDERSTUDY_SPEAKER_MODEL")
    if model:
        try:
            import sherpa_onnx
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=1)
            ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

            def fn(x):
                s = ex.create_stream()
                s.accept_waveform(sample_rate=SR, waveform=x)
                s.input_finished()
                return np.asarray(ex.compute(s), dtype=np.float32)
            return fn
        except Exception:
            pass
    try:
        from resemblyzer import VoiceEncoder
        enc = VoiceEncoder()
        return lambda x: enc.embed_utterance(x)
    except Exception:
        return None


class Diarizer(Stage):
    name = "diarize"

    def __init__(self, clock, session, metrics=None, embed_fn="auto",
                 threshold=DEFAULT_THRESHOLD, inbox_size=64):
        super().__init__(clock, session, metrics, inbox_size)
        self.embed_fn = default_embed_fn() if embed_fn == "auto" else embed_fn
        self.threshold = threshold
        self._out = (session.writer("speakers.jsonl") if hasattr(session, "writer")
                     else JsonlWriter(os.path.join(session.dir, "speakers.jsonl")))
        self._lock = threading.Lock()
        self._records = []
        self._centroids = []   # [sum_vector, count]
        self._buf = None       # [t0, [arrays], n_samples] pending system audio
        self._names = ({}, None)

    @property
    def tier(self):
        return 2 if self.embed_fn else 1

    # -- stage --
    def step(self, item):
        source, t0, pcm = item
        x = np.frombuffer(pcm, dtype=np.int16)
        if x.size == 0:
            return
        if source == "system" and self.embed_fn:
            self._system(t0, x)
        else:
            self._emit(t0, t0 + x.size / SR, source, x,
                       "local" if source == "mic" else "remote", 1.0)

    def _system(self, t0, x):
        b = self._buf
        # a gap in the stream restarts the window so timing stays honest
        if b and abs(b[0] + b[2] / SR - t0) > 0.25:
            self._flush_window()
            b = None
        if not b:
            b = self._buf = [t0, [], 0]
        b[1].append(x)
        b[2] += x.size
        if b[2] >= WINDOW_S * SR:
            self._flush_window()

    def _flush_window(self):
        b, self._buf = self._buf, None
        if not b or not b[1]:
            return
        x = np.concatenate(b[1])
        t0, t1 = b[0], b[0] + x.size / SR
        if self._rms(x) < VOICED_RMS:
            return
        try:
            sid, conf = self._assign(np.asarray(
                self.embed_fn(x.astype(np.float32) / 32768.0), dtype=np.float32))
        except Exception as exc:
            self.error = repr(exc)
            sid, conf = "remote", 0.0
        self._emit(t0, t1, "system", x, sid, conf)

    @staticmethod
    def _rms(x):
        return float(np.sqrt((x.astype(np.float64) ** 2).mean())) if x.size else 0.0

    def _assign(self, v):
        n = np.linalg.norm(v)
        v = v / n if n else v
        best, best_sim = -1, -1.0
        for i, (s, c) in enumerate(self._centroids):
            m = s / c
            sim = float(np.dot(v, m) / (np.linalg.norm(m) or 1.0))
            if sim > best_sim:
                best, best_sim = i, sim
        if best < 0 or best_sim < self.threshold:
            self._centroids.append([v.copy(), 1])
            return "S%d" % len(self._centroids), 1.0 if best < 0 else round(1 - best_sim, 3)
        self._centroids[best][0] += v
        self._centroids[best][1] += 1
        return "S%d" % (best + 1), round(best_sim, 3)

    def _emit(self, t0, t1, source, x, sid, conf):
        if self._rms(x) < VOICED_RMS:
            return
        rec = {"t0": round(t0, 3), "t1": round(t1, 3), "source": source,
               "speaker_id": sid, "confidence": round(float(conf), 3)}
        with self._lock:
            self._records.append(rec)
            del self._records[:-MAX_RECORDS]
        self._out.write(rec)

    # -- lookup --
    def _name_map(self):
        p = os.path.join(self.session.dir, participants.FILE)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            mt = None
        with self._lock:
            if self._names[1] != mt:
                self._names = (participants.load(self.session.dir), mt)
            return self._names[0]

    def speaker_at(self, t0, t1, source):
        """Display name (or id) of whoever spoke most in [t0, t1] on `source`."""
        with self._lock:
            recs = list(self._records)
        over = {}
        for r in recs:
            if r["source"] != source:
                continue
            o = min(t1, r["t1"]) - max(t0, r["t0"])
            if o > 0:
                over[r["speaker_id"]] = over.get(r["speaker_id"], 0.0) + o
        if not over:
            return None
        sid = max(over, key=over.get)
        return self._name_map().get(sid, sid)

    def stop(self):
        super().stop()
        try:
            if self._buf and self.embed_fn:
                self._flush_window()
        except Exception:
            pass
        self._out.close()
