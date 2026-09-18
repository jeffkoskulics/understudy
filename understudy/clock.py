"""Single time base for every stream in a session.

Everything downstream (frames, events, audio) is stamped with `t`, seconds
since `Clock.start`, taken from a monotonic source so that NTP steps and
sleep/wake cannot make timestamps go backwards. The wall-clock start time is
recorded once, in the manifest, purely so a human can locate the session.
"""
import time
import datetime


class Clock:
    def __init__(self):
        self._t0 = time.monotonic()
        self.started_at = datetime.datetime.now().astimezone()

    def t(self) -> float:
        return time.monotonic() - self._t0

    @property
    def start_iso(self) -> str:
        return self.started_at.isoformat(timespec="seconds")
