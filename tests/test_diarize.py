import json
import os
import types

import numpy as np

from understudy import participants
from understudy.live.diarize import Diarizer, SR


def pcm(freq, secs, amp=8000):
    t = np.arange(int(SR * secs)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16).tobytes()


def fake_embed(x):
    # dominant-frequency "voice": low tone vs high tone
    f = np.abs(np.fft.rfft(x))
    peak_hz = np.argmax(f) * SR / len(x)
    v = np.zeros(8, dtype=np.float32)
    v[0 if peak_hz < 500 else 1] = 1.0
    return v


def make(tmp_path, embed):
    s = types.SimpleNamespace(dir=str(tmp_path))
    return Diarizer(None, s, embed_fn=embed)


def recs(tmp_path):
    with open(os.path.join(str(tmp_path), "speakers.jsonl")) as fh:
        return [json.loads(l) for l in fh]


def test_tier1_channels(tmp_path):
    d = make(tmp_path, None)
    d.step(("mic", 0.0, pcm(200, 1)))
    d.step(("system", 1.0, pcm(200, 1)))
    d.step(("system", 2.0, b"\0\0" * SR))  # silence is skipped
    d.stop()
    assert [x["speaker_id"] for x in recs(tmp_path)] == ["local", "remote"]
    assert d.speaker_at(1.2, 1.8, "system") == "remote"
    assert d.speaker_at(5, 6, "system") is None


def test_tier2_clusters_and_names(tmp_path):
    d = make(tmp_path, fake_embed)
    t = 0.0
    for f in (200, 200, 900, 200):     # each 1.5 s window is one turn
        for _ in range(3):
            d.step(("system", t, pcm(f, 0.5)))
            t += 0.5
    d.step(("mic", 0.0, pcm(300, 1)))
    d.stop()
    ids = [r["speaker_id"] for r in recs(tmp_path) if r["source"] == "system"]
    assert ids == ["S1", "S1", "S2", "S1"]
    assert d.speaker_at(3.1, 4.4, "system") == "S2"
    assert d.speaker_at(0, 1, "mic") == "local"
    participants.name(str(tmp_path), "S2", "Maria")   # retroactive
    assert d.speaker_at(3.1, 4.4, "system") == "Maria"
