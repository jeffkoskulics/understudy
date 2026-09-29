import sys
import time
import types
import wave

import numpy as np

from understudy import audio, loopback


class Clock:
    def t(self):
        return 10.0


class FakeReader:
    def __init__(self):
        self.n = 0

    def read(self, frames):
        self.n += 1
        if self.n > 3:
            time.sleep(0.01)
            return None
        return np.ones(frames, dtype="int16")

    def close(self):
        pass


def test_system_recorder_writes_and_taps(tmp_path, monkeypatch):
    rd = FakeReader()
    monkeypatch.setattr(loopback, "open_loopback", lambda d, r: (rd, "mon"))
    got = []
    p = str(tmp_path / "audio_system.wav")
    rec = audio.SystemAudioRecorder(Clock(), p, on_chunk=lambda *a: got.append(a))
    rec.start()
    while rd.n < 4:
        time.sleep(0.01)
    rec.stop()
    rec.join(2)
    assert rec.error is None and rec.frames_written == 3 * rec.BLOCK
    assert len(got) == 3 and got[0][0] == "system"
    assert abs(got[0][1] - (10.0 - rec.BLOCK / 16000)) < 1e-6
    assert len(got[0][2]) == rec.BLOCK * 2
    with wave.open(p) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
    assert rec.info()["start_offset"] is not None


def test_failure_swallowed(tmp_path, monkeypatch):
    def boom(d, r):
        raise loopback.LoopbackUnavailable("nope")
    monkeypatch.setattr(loopback, "open_loopback", boom)
    rec = audio.SystemAudioRecorder(Clock(), str(tmp_path / "x.wav"))
    rec.run()
    assert rec.error == "nope"


def test_mac_without_blackhole(monkeypatch):
    fake = types.SimpleNamespace(query_devices=lambda: [
        {"name": "MacBook Mic", "max_input_channels": 1}])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(loopback, "_platform", lambda: "darwin")
    try:
        loopback.open_loopback()
        assert False
    except loopback.LoopbackUnavailable as e:
        assert "BlackHole" in str(e)
    assert loopback.list_devices()[0]["loopback"] is False


def test_mac_blackhole_listed(monkeypatch):
    fake = types.SimpleNamespace(query_devices=lambda: [
        {"name": "BlackHole 2ch", "max_input_channels": 2}])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    monkeypatch.setattr(loopback, "_platform", lambda: "darwin")
    assert loopback.list_devices()[0]["loopback"] is True


def test_list_devices_soundcard(monkeypatch):
    mics = [types.SimpleNamespace(name="Mic", isloopback=False),
            types.SimpleNamespace(name="Spk", isloopback=True)]
    fake = types.SimpleNamespace(all_microphones=lambda include_loopback: mics)
    monkeypatch.setitem(sys.modules, "soundcard", fake)
    monkeypatch.setattr(loopback, "_platform", lambda: "linux")
    assert [d["loopback"] for d in audio.list_devices()] == [False, True]


def test_mic_tap_and_bad_consumer():
    got = []
    r = audio.AudioRecorder(Clock(), "x", on_chunk=lambda *a: got.append(a))
    r._callback(np.zeros(1600, dtype="int16"), 1600, None, None)
    assert got[0][0] == "mic" and abs(got[0][1] - 9.9) < 1e-6
    r2 = audio.AudioRecorder(Clock(), "x", on_chunk=lambda *a: 1 / 0)
    r2._callback(np.zeros(160, dtype="int16"), 160, None, None)
    assert r2.error is None
