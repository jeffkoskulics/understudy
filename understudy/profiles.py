"""Capture presets for the three ways Understudy is used.

Each profile is a dict of Recorder/CaptureLoop keyword arguments plus two
flags the integrator acts on: `indicator` (show the always-on-top recording
window) and `audio_system` (also capture loopback audio). Strip those two
before passing the rest to CaptureLoop.

  teacher  an expert demonstrating a workflow. The existing dedup behaviour:
           window switches and clicks are the step boundaries.
  student  someone working through a task. Dedup as well, but sampled a little
           faster and with a lower heartbeat, since pauses (being stuck) are
           informative here.
  meeting  a video call with shared screens. Window switches are a weak signal
           -- the shared content is the screen, and the call window is often
           frontmost -- so the pixel diff does the work: sampled at 2 fps, a
           frame is written when the shared screen changes, with a 30 s
           heartbeat. The indicator is on and system audio is captured so
           remote voices are heard. (A first real meeting session in fixed
           mode kept 645 frames of which 92 had changed; `--mode fixed` is
           still there for uniform sampling.)
"""

PROFILES = {
    "teacher": {"mode": "dedup", "fps": 2.0, "min_change": 0.004,
                "heartbeat": 120.0, "indicator": False, "audio_system": False},
    "student": {"mode": "dedup", "fps": 3.0, "min_change": 0.004,
                "heartbeat": 30.0, "indicator": False, "audio_system": False},
    "meeting": {"mode": "dedup", "fps": 2.0, "min_change": 0.004,
                "heartbeat": 30.0, "indicator": True, "audio_system": True},
}


def get_profile(name):
    """Return a copy of the named profile; raises KeyError listing the options."""
    try:
        return dict(PROFILES[name])
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(PROFILES)}") from None
