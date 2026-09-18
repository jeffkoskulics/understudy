"""Recorder entry point: wires the streams together and owns the session."""
import argparse
import os
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


def build_parser():
    p = argparse.ArgumentParser(prog="wfcap", description="Capture a GUI workflow for AI analysis.")
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

    clock = Clock()
    session = Session(args.out, args.name)
    frontmost = get_frontmost()

    cap = CaptureLoop(clock, session, monitor=args.monitor, fps=args.fps,
                      min_change=args.min_change, heartbeat=args.heartbeat,
                      jpeg_quality=args.quality, frontmost=frontmost)
    ev = EventRecorder(clock, session.events, frontmost,
                       key_mode=args.keys, on_activity=cap.on_activity)
    audio = None if args.no_audio else AudioRecorder(clock, session.audio_path,
                                                     args.audio_device)

    stopping = {"flag": False}

    def request_stop(*_):
        stopping["flag"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cap.start()
    ev.start()
    if audio:
        audio.start()

    print("Recording -> %s" % session.dir)
    print("Press Ctrl-C to stop.")
    t_end = args.duration
    try:
        while not stopping["flag"]:
            time.sleep(0.25)
            if t_end is not None and clock.t() >= t_end:
                break
            print("\r  %5.1fs  %d frames kept / %d sampled   "
                  % (clock.t(), cap.kept, cap.sampled), end="", flush=True)
    finally:
        print()
        ev.stop()
        cap.stop()
        if audio:
            audio.stop()
            audio.join(timeout=5)
        cap.join(timeout=5)

        session.write_manifest({
            "name": session.name,
            "started_at": clock.start_iso,
            "duration": round(clock.t(), 3),
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine()},
            "capture": {"fps": args.fps, "monitor": args.monitor,
                        "min_change": args.min_change, "heartbeat": args.heartbeat,
                        "jpeg_quality": args.quality, "geometry": cap.geometry},
            "frames": {"kept": cap.kept, "sampled": cap.sampled,
                       "moves_suppressed": cap.moves_suppressed},
            "events": {"key_mode": args.keys},
            "audio": audio.info() if audio else None,
        })
        session.close()

    kept, sampled = cap.kept, cap.sampled
    saved = (1 - kept / sampled) * 100 if sampled else 0
    print("Saved %d frames (%.0f%% of samples deduplicated) to %s"
          % (kept, saved, session.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
