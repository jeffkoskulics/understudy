"""End-to-end smoke test: LiveRecorder with fake screen, audio and models."""
import json
import os
import sys
import threading
import time
import types
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

try:                                  # no input hooks needed (or wanted) in tests
    import pynput  # noqa: F401
except Exception:
    _p = types.ModuleType("pynput")
    _p.mouse = _p.keyboard = types.SimpleNamespace()
    sys.modules["pynput"] = _p
    sys.modules["pynput.mouse"] = _p.mouse
    sys.modules["pynput.keyboard"] = _p.keyboard
try:
    import mss  # noqa: F401
except Exception:
    sys.modules["mss"] = types.SimpleNamespace(mss=None)

from understudy import capture, deps, loopback, record
from understudy.live import run
from understudy.live.backends import audio_faster_whisper


class FakeSct:
    monitors = [{}, {"left": 0, "top": 0, "width": 64, "height": 48}]

    def __init__(self):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def grab(self, mon):
        self.n += 1
        bgra = bytes([self.n * 7 % 256, 80, 160, 255]) * (64 * 48)
        return types.SimpleNamespace(size=(64, 48), bgra=bgra)


class FakeInputStream:
    def __init__(self, samplerate, channels, dtype, device, callback):
        self.cb, self.halt = callback, threading.Event()

    def __enter__(self):
        t = np.arange(3200) / 16000.0
        pcm = (3000 * np.sin(2 * np.pi * 220 * t)).astype("int16").tobytes()

        def pump():
            while not self.halt.is_set():
                self.cb(pcm, 3200, None, None)
                time.sleep(0.2)
        self.th = threading.Thread(target=pump, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.halt.set()
        self.th.join(1)
        return False


class FakeEvents:
    def __init__(self, clock, writer, frontmost, key_mode="metadata", on_activity=None):
        writer.write({"t": 0.0, "type": "start"})

    def start(self):
        pass

    def stop(self):
        pass


class FakeReader:
    def read(self, frames):
        time.sleep(0.2)
        t = np.arange(frames) / 16000.0
        return (3000 * np.sin(2 * np.pi * 330 * t)).astype("int16")

    def close(self):
        pass


class FakeBackend:
    name, model = "fake-whisper", "fake"

    def __init__(self, model=None, **kw):
        pass

    def run(self, payload):
        return {"text": "hello world", "words": [
            {"w": "hello", "t0": 0.0, "t1": 0.4}, {"w": "world", "t0": 0.5, "t1": 0.9}]}


class VisionServer(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5vl:7b"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"response": "A test screen."})


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(capture, "mss", types.SimpleNamespace(mss=FakeSct))
    monkeypatch.setitem(sys.modules, "sounddevice",
                        types.SimpleNamespace(InputStream=FakeInputStream))
    monkeypatch.setattr(loopback, "open_loopback", lambda d, r: (FakeReader(), "fake-monitor"))
    monkeypatch.setattr(record, "EventRecorder", FakeEvents)
    monkeypatch.setattr(deps, "is_installed", lambda m: True)
    monkeypatch.setattr(audio_faster_whisper, "FasterWhisperBackend", FakeBackend)
    shown = []

    def fake_indicator(master=None):
        shown.append(1)
        return types.SimpleNamespace(pump=lambda: None, close=lambda: None)

    monkeypatch.setattr(run, "Indicator", fake_indicator)
    srv = HTTPServer(("127.0.0.1", 0), VisionServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": "http://127.0.0.1:%d" % srv.server_port, "indicator": shown}
    srv.shutdown()


def jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_live_recorder_end_to_end(tmp_path, fakes):
    rec = run.LiveRecorder(out=str(tmp_path), name="s", profile="meeting",
                           vision_url=fakes["url"], vision_max_rate=10)
    rec.start()
    time.sleep(2.2)
    assert "frames" in rec.status_line()
    d = rec.stop()

    assert fakes["indicator"]                       # meeting profile shows the REC window
    for name in ("frames.jsonl", "vision.jsonl", "live_transcript.jsonl",
                 "speakers.jsonl", "metrics.jsonl"):
        assert os.path.getsize(os.path.join(d, name)) > 0, name
    for name in ("audio.wav", "audio_system.wav"):
        with wave.open(os.path.join(d, name)) as w:
            assert w.getframerate() == 16000
    frames = jsonl(os.path.join(d, "frames.jsonl"))
    assert frames and frames[0]["mode"] == "fixed"
    vis = jsonl(os.path.join(d, "vision.jsonl"))
    assert any(v.get("text") == "A test screen." for v in vis)
    tr = jsonl(os.path.join(d, "live_transcript.jsonl"))
    assert {r["source"] for r in tr} == {"mic", "system"}
    assert any(r.get("speaker") in ("local", "remote") for r in tr)
    sp = jsonl(os.path.join(d, "speakers.jsonl"))
    assert {r["speaker_id"] for r in sp} <= {"local", "remote"} | {"S%d" % i for i in range(1, 9)}
    m = jsonl(os.path.join(d, "metrics.jsonl"))
    assert m and "cpu_pct" in m[0]
    with open(os.path.join(d, "system.json")) as fh:
        assert json.load(fh)["models"]["vision"] == "qwen2.5vl:7b"
    with open(os.path.join(d, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["profile"] == "meeting" and man["live"]["vision"] is True
    assert man["audio_system_offset"] is not None
    assert man["audio_system"]["device"] == "fake-monitor"
    assert man["capture"]["mode"] == "fixed"


def test_vision_probe_failure_continues(tmp_path, fakes):
    rec = run.LiveRecorder(out=str(tmp_path), name="s2", profile="student",
                           vision_url="http://127.0.0.1:9", system_audio=False)
    assert rec.vision is None and rec.notes
    rec.start()
    time.sleep(0.8)
    d = rec.stop()
    assert not fakes["indicator"]                    # student profile: no indicator
    with open(os.path.join(d, "manifest.json")) as fh:
        man = json.load(fh)
    assert man["live"]["vision"] is False and "audio_system_offset" not in man
    assert man["frames"]["kept"] >= 1


def test_session_writer_cached(tmp_path):
    from understudy.session import Session
    s = Session(str(tmp_path), "w")
    assert s.writer("a.jsonl") is s.writer("a.jsonl")
    s.writer("a.jsonl").write({"t": 1})
    s.close()
    assert jsonl(os.path.join(s.dir, "a.jsonl")) == [{"t": 1}]
