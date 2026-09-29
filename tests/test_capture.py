import sys
import types

import pytest

try:                       # mss needs no display to import, but may be absent
    import mss  # noqa: F401
except ImportError:
    sys.modules["mss"] = types.SimpleNamespace(mss=None)

from understudy import capture
from understudy.capture import CaptureLoop, decide_reasons
from understudy.profiles import PROFILES, get_profile

D = dict(min_change=0.004, heartbeat=120.0)


def dr(forced=None, wreason=None, frac=0.0, idle=0.0, first=False):
    return decide_reasons(forced, wreason, frac, idle, first, **D)


def test_decide_basic():
    assert dr() == []
    assert dr(frac=0.5) == ["diff"]
    assert dr(idle=200) == ["heartbeat"]
    assert dr(first=True, frac=1.0)[0] == "first"
    assert dr(forced="click", wreason="window-switch", frac=1)[:2] == ["click", "window-switch"]


def test_decide_suppress():
    assert dr(wreason="suppress", frac=0.9) == []
    assert dr(wreason="suppress", forced="click", frac=0.9) == ["click"]
    assert "heartbeat" in dr(wreason="suppress", idle=500)


def test_profiles():
    assert set(PROFILES) == {"teacher", "student", "meeting"}
    m = get_profile("meeting")
    assert m["mode"] == "fixed" and m["fps"] == 2.0 and m["indicator"] and m["audio_system"]
    assert not get_profile("teacher")["indicator"]
    for p in PROFILES.values():
        kw = {k: v for k, v in p.items() if k not in ("indicator", "audio_system")}
        CaptureLoop(None, None, **kw)
    with pytest.raises(KeyError):
        get_profile("x")


def test_validation():
    with pytest.raises(ValueError):
        CaptureLoop(None, None, mode="bogus")
    with pytest.raises(ValueError):
        CaptureLoop(None, None, mode="fixed", fps=1)
    with pytest.raises(ValueError):
        CaptureLoop(None, None, mode="fixed", fps=5)
    assert CaptureLoop(None, None).mode == "dedup"


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def t(self):
        return self.now


class FakeSession:
    def __init__(self, tmp):
        self.tmp = tmp
        self.rows = []
        self.frames = types.SimpleNamespace(write=self.rows.append)

    def frame_path(self, i):
        return str(self.tmp / f"{i}.jpg")

    def frame_rel(self, i):
        return f"frames/{i}.jpg"


class FakeSct:
    """Yields scripted grey levels; stops the loop when the script runs out."""
    def __init__(self, colours, box):
        self.colours, self.box = iter(colours), box
        self.monitors = [None, {"left": 0, "top": 0, "width": 64, "height": 64}]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def grab(self, mon):
        loop = self.box[0]
        try:
            c = next(self.colours)
        except StopIteration:
            loop.stop()
            c = 0
        loop.clock.now += loop.interval   # one sample interval per grab
        return types.SimpleNamespace(size=(64, 64), bgra=bytes([c, c, c, 0]) * 64 * 64)


def run_loop(monkeypatch, tmp_path, mode, colours, on_frame=None):
    box = [None]
    sess = FakeSession(tmp_path)
    loop = CaptureLoop(FakeClock(), sess, mode=mode, fps=2.0, on_frame=on_frame)
    box[0] = loop
    monkeypatch.setattr(capture.mss, "mss", lambda: FakeSct(colours, box))
    loop.run()
    return sess.rows


def test_dedup_mode_drops_static(monkeypatch, tmp_path):
    got = []
    rows = run_loop(monkeypatch, tmp_path, "dedup", [10, 10, 10, 200, 200],
                    on_frame=lambda r, p: got.append((r, p)))
    assert [r["reasons"] for r in rows[:2]] == [["first"], ["diff"]]  # rows[2:]: sentinel frame after script end
    assert all(r["mode"] == "dedup" for r in rows)
    assert len(got) == len(rows)


def test_fixed_mode_writes_every_frame(monkeypatch, tmp_path):
    got = []
    rows = run_loop(monkeypatch, tmp_path, "fixed", [10, 10, 10, 200, 200],
                    on_frame=lambda r, p: got.append((r, p)))
    assert [r["reasons"] for r in rows[:5]] == [["first"], [], [], ["diff"], []]
    assert rows[1]["reason"] == "fixed"
    assert all(r["mode"] == "fixed" for r in rows)
    assert len(got) == len(rows)
    assert got[0][0]["idx"] == 0 and got[0][1].endswith("0.jpg")


def test_on_frame_errors_do_not_kill_capture(monkeypatch, tmp_path):
    def boom(r, p):
        raise RuntimeError("x")
    rows = run_loop(monkeypatch, tmp_path, "fixed", [1, 2], on_frame=boom)
    assert len(rows) >= 2
