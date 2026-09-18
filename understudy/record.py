"""Recorder entry point: wires the streams together and owns the session.

The wiring lives in `Recorder` rather than in `main()` so that the GUI can
drive exactly the same recording path as the command line. A recording is
three independent streams -- frames, events, audio -- sharing one clock and one
session directory; starting and stopping them in the right order, and writing
the manifest that ties them together, is the whole job.
"""
import argparse
import platform
import signal
import sys
import time

from .audio import AudioRecorder
from .capture import CaptureLoop
from .clock import Clock
from .events import EventRecorder
from .frontmost import get_frontmost
from .session import Session

DEFAULT_ROOT = "~/Recordings"


class Recorder:
    """One recording session. Start it, poll it, stop it.

    Stopping is idempotent and always writes the manifest, because a session
    without one cannot be handed off: `handoff` reads the audio offset from it
    to align narration against the frames.
    """

    def __init__(self, out=DEFAULT_ROOT, name=None, fps=2.0, monitor=1,
                 min_change=0.004, quality=92, heartbeat=120.0,
                 audio=True, audio_device=None, keys="metadata"):
        self.clock = Clock()
        self.session = Session(out, name)
        self.fps = fps
        self.monitor = monitor
        self.keys = keys
        frontmost = get_frontmost()
        self.cap = CaptureLoop(self.clock, self.session, monitor=monitor, fps=fps,
                               min_change=min_change, heartbeat=heartbeat,
                               jpeg_quality=quality, frontmost=frontmost)
        self.ev = EventRecorder(self.clock, self.session.events, frontmost,
                                key_mode=keys, on_activity=self.cap.on_activity)
        self.audio = (AudioRecorder(self.clock, self.session.audio_path, audio_device)
                      if audio else None)
        self._settings = {"min_change": min_change, "heartbeat": heartbeat,
                          "jpeg_quality": quality}
        self._stopped = False

    @property
    def dir(self):
        return self.session.dir

    def start(self):
        self.cap.start()
        self.ev.start()
        if self.audio:
            self.audio.start()

    def status(self):
        """(elapsed seconds, frames kept, frames sampled)."""
        return self.clock.t(), self.cap.kept, self.cap.sampled

    def stop(self):
        """Stop every stream, write the manifest, return the session dir."""
        if self._stopped:
            return self.session.dir
        self._stopped = True
        self.ev.stop()
        self.cap.stop()
        if self.audio:
            self.audio.stop()
            self.audio.join(timeout=5)
        self.cap.join(timeout=5)

        self.session.write_manifest({
            "name": self.session.name,
            "started_at": self.clock.start_iso,
            "duration": round(self.clock.t(), 3),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine()},
            "capture": {"fps": self.fps, "monitor": self.monitor,
                        "geometry": self.cap.geometry, **self._settings},
            "frames": {"kept": self.cap.kept, "sampled": self.cap.sampled,
                       "moves_suppressed": self.cap.moves_suppressed},
            "events": {"key_mode": self.keys},
            "audio": self.audio.info() if self.audio else None,
        })
        self.session.close()
        return self.session.dir


def build_parser():
    p = argparse.ArgumentParser(prog="understudy", description="Capture a GUI workflow for AI analysis.")
    p.add_argument("--out", default=DEFAULT_ROOT, help="folder to save sessions into")
    p.add_argument("--name", default=None, help="session name (default: timestamp)")
    p.add_argument("--fps", type=float, default=2.0, help="capture rate, 2-4")
    p.add_argument("--monitor", type=int, default=1, help="monitor index, 1 = primary")
    p.add_argument("--min-change", type=float, default=0.004,
                   help="fraction of the screen that must change to keep a frame")
    p.add_argument("--quality", type=int, default=92, help="JPEG quality")
    p.add_argument("--heartbeat", type=float, default=120.0,
                   help="force a frame after this many idle seconds")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--audio-device", default=None)
    p.add_argument("--keys", choices=["metadata", "full"], default="metadata",
                   help="'full' also records typed characters")
    p.add_argument("--duration", type=float, default=None,
                   help="stop automatically after N seconds")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0.5 <= args.fps <= 10:
        sys.exit("--fps must be between 0.5 and 10")

    rec = Recorder(out=args.out, name=args.name, fps=args.fps, monitor=args.monitor,
                   min_change=args.min_change, quality=args.quality,
                   heartbeat=args.heartbeat, audio=not args.no_audio,
                   audio_device=args.audio_device, keys=args.keys)

    stopping = {"flag": False}

    def request_stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    rec.start()
    print("Recording -> %s" % rec.dir)
    print("Press Ctrl-C to stop.")
    try:
        while not stopping["flag"]:
            time.sleep(0.25)
            t, kept, sampled = rec.status()
            if args.duration is not None and t >= args.duration:
                break
            print("\r  %5.1fs  %d frames kept / %d sampled   "
                  % (t, kept, sampled), end="", flush=True)
    finally:
        print()
        rec.stop()

    _, kept, sampled = rec.status()
    saved = (1 - kept / sampled) * 100 if sampled else 0
    print("Saved %d frames (%.0f%% of samples deduplicated) to %s"
          % (kept, saved, rec.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
