"""`understudy live`: a Recorder plus the live analysis stages.

Wiring (see docs/PLAN-live-analysis.md):

    capture --on_frame--> VisionAnalyzer            -> vision.jsonl
    mic / system audio --on_chunk--> LiveTranscriber -> live_transcript.jsonl
                                 \\-> Diarizer        -> speakers.jsonl
    Metrics (started first, stopped last)           -> metrics.jsonl, system.json

Every optional stage degrades instead of aborting: a missing vision server, a
missing whisper install or an unavailable loopback device is reported and the
recording carries on without it.
"""
import argparse
import os
import signal
import sys
import threading
import time

from ..audio import SystemAudioRecorder
from ..profiles import PROFILES, get_profile
from ..record import DEFAULT_ROOT, Recorder
from .metrics import Metrics, write_system_json

try:
    import psutil
except ImportError:
    psutil = None


class Indicator:
    """Small always-on-top "REC" window; console banner if Tk is unavailable."""

    def __init__(self, master=None):
        self.win = None
        self.owns_root = False
        try:
            import tkinter as tk
            if master is not None:
                self.win = tk.Toplevel(master)
            else:
                self.win = tk.Tk()
                self.owns_root = True
            self.win.title("Understudy")
            self.win.attributes("-topmost", True)
            try:
                self.win.resizable(False, False)
            except Exception:
                pass
            tk.Label(self.win, text="● REC", fg="#d00000",
                     font=("Helvetica", 14, "bold"), padx=14, pady=6).pack()
            self.win.update()
        except Exception:
            self.win = None
            self.owns_root = False
            print("=" * 40 + "\n  ● RECORDING (screen + audio)\n" + "=" * 40)

    def pump(self):
        """Service the window from a polling loop (only needed when we own the root)."""
        if self.win is not None and self.owns_root:
            try:
                self.win.update()
            except Exception:
                self.win = None

    def close(self):
        if self.win is not None:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None


class _Tail:
    """Incrementally count lines (and lines containing `needle`) of a JSONL file."""

    def __init__(self, path, needle=None):
        self.path, self.needle = path, needle
        self.pos = 0
        self.lines = 0
        self.matched = 0

    def poll(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                data = fh.read()
        except OSError:
            return self.lines, self.matched
        end = data.rfind(b"\n") + 1
        for line in data[:end].splitlines():
            self.lines += 1
            if self.needle and self.needle in line:
                self.matched += 1
        self.pos += end
        return self.lines, self.matched


class LiveRecorder(Recorder):
    def __init__(self, out=DEFAULT_ROOT, name=None, profile="meeting", mode=None,
                 fps=None, monitor=1, quality=80, audio=True, audio_device=None,
                 keys="metadata", system_audio=None, system_audio_device=None,
                 vision=True, transcribe=True, diarize=True,
                 vision_backend="ollama", vision_model=None, vision_url=None,
                 vision_mode="describe", vision_max_rate=1.0,
                 vision_timeout=120.0, vision_max_side=1280,
                 vision_on_change=False, vision_context=None,
                 whisper_model=None, indicator=None, tk_master=None):
        prof = get_profile(profile)
        want_indicator = prof.pop("indicator")
        want_system = prof.pop("audio_system")
        if mode:
            prof["mode"] = mode
        if fps:
            prof["fps"] = fps
        self.profile = profile
        self.want_indicator = want_indicator if indicator is None else indicator
        self.want_system = want_system if system_audio is None else system_audio
        self._tk_master = tk_master
        self.indicator = None
        self.vision = None
        self.transcriber = None
        self.diarizer = None
        self.system = None
        self.metrics = None
        self.notes = []          # human-readable degradations, shown and stored
        self._models = {}

        super().__init__(out=out, name=name, fps=prof["fps"], monitor=monitor,
                         min_change=prof["min_change"], quality=quality,
                         heartbeat=prof["heartbeat"], audio=audio,
                         audio_device=audio_device, keys=keys, mode=prof["mode"],
                         on_frame=self._on_frame, on_chunk=self._on_chunk)

        self.metrics = Metrics(self.session.dir, self.clock)
        self._build_diarizer(diarize)
        self._build_transcriber(transcribe, whisper_model)
        self._build_vision(vision, vision_backend, vision_model, vision_url,
                           vision_mode, vision_max_rate, vision_timeout, vision_max_side,
                           vision_on_change, vision_context)
        if self.want_system:
            self.system = SystemAudioRecorder(
                self.clock, os.path.join(self.session.dir, "audio_system.wav"),
                system_audio_device, on_chunk=self._on_chunk)
        self._live = {"profile": profile, "mode": prof["mode"], "fps": prof["fps"],
                      "vision": self.vision is not None,
                      "transcribe": self.transcriber is not None,
                      "diarize": self.diarizer is not None,
                      "system_audio": bool(self.want_system),
                      "vision_mode": vision_mode, "vision_max_rate": vision_max_rate,
                      "vision_backend": vision_backend}
        self._vtail = _Tail(os.path.join(self.session.dir, "vision.jsonl"), b'"skipped"')
        self._ttail = _Tail(os.path.join(self.session.dir, "live_transcript.jsonl"))
        self._proc = psutil.Process() if psutil else None
        if self._proc:
            self._proc.cpu_percent(None)

    # -- construction helpers ---------------------------------------------
    def _build_diarizer(self, on):
        if not on:
            return
        try:
            from .diarize import Diarizer
            self.diarizer = Diarizer(self.clock, self.session, self.metrics)
            self._models["speakers"] = ("embedding (tier 2)" if self.diarizer.tier == 2
                                        else "channel only (tier 1)")
        except Exception as exc:
            self._note("speaker attribution unavailable: %s" % exc)

    def _build_transcriber(self, on, model):
        if not on:
            return
        from .. import deps, transcribe as batch
        if not deps.is_installed("faster_whisper"):
            self._note("live transcription off: faster-whisper not installed "
                       "(pip install -r requirements-whisper.txt)")
            return
        from .backends.audio_faster_whisper import FasterWhisperBackend
        from .transcribe_live import LiveTranscriber
        backend = FasterWhisperBackend(model or batch.DEFAULT_MODEL)
        self._models["whisper"] = backend.model
        self.transcriber = LiveTranscriber(
            self.clock, self.session, backend, self.metrics,
            speaker=self.diarizer.speaker_at if self.diarizer else None)

    def _build_vision(self, on, kind, model, url, mode, max_rate,
                      timeout=120.0, max_side=1280, on_change=False,
                      context=None):
        if not on:
            return
        from .backends.vision_local import OllamaBackend, OpenAICompatBackend
        from .vision import VisionAnalyzer
        cls = OllamaBackend if kind == "ollama" else OpenAICompatBackend
        kw = {"timeout": timeout}
        if model:
            kw["model"] = model
        if url:
            kw["host"] = url
        backend = cls(**kw)
        analyzer = VisionAnalyzer(self.clock, self.session, backend, mode=mode,
                                  max_rate=max_rate, max_side=max_side,
                                  on_change=on_change, context=context,
                                  metrics=self.metrics)
        try:
            analyzer.probe()
            if hasattr(backend, "warm_up"):
                print("Loading vision model %s ..." % backend.model, file=sys.stderr)
                backend.warm_up()
        except Exception as exc:
            self._note("Vision analysis disabled: %s" % exc)
            return
        self.vision = analyzer
        self._models["vision"] = backend.model
        self._models["vision_backend"] = backend.name

    def _note(self, msg):
        self.notes.append(msg)
        print(msg, file=sys.stderr)

    # -- fan-out callbacks (capture/audio threads: must never raise) --------
    def _on_frame(self, record, path):
        v = self.vision
        if v is not None:
            v.on_frame(record, path)

    def _on_chunk(self, source, t0, data):
        t = self.transcriber
        if t is not None:
            try:
                t.on_chunk(source, t0, data)
            except Exception:
                pass
        d = self.diarizer
        if d is not None:
            try:
                d.offer((source, t0, data))
            except Exception:
                pass

    # -- lifecycle -----------------------------------------------------------
    @property
    def stages(self):
        return [s for s in (self.transcriber, self.diarizer, self.vision) if s]

    def start(self):
        self.metrics.start()
        try:
            write_system_json(self.session.dir, extra={
                "models": self._models, "live": dict(self._live)})
        except Exception:
            pass
        for s in self.stages:
            s.start()
        super().start()
        if self.system:
            self.system.start()
        if self.want_indicator:
            self.indicator = Indicator(self._tk_master)

    def pump(self):
        if self.indicator:
            self.indicator.pump()

    def status_line(self):
        t, kept, _ = self.status()
        analysed = skipped = 0
        if self.vision:
            n, skipped = self._vtail.poll()
            analysed = n - skipped
        lines = self._ttail.poll()[0] if self.transcriber else 0
        cpu = rss = "n/a"
        if self._proc:
            try:
                cpu = "%.0f%%" % self._proc.cpu_percent(None)
                rss = "%.0fMB" % (self._proc.memory_info().rss / 2**20)
            except Exception:
                pass
        return ("%5.1fs  frames %d  vision %d done/%d skipped  transcript %d  cpu %s  rss %s"
                % (t, kept, analysed, skipped, lines, cpu, rss))

    def _stop_sources(self):
        if self.system:
            self.system.stop()

    def _stop_stages(self):
        if self.indicator:
            self.indicator.close()
        if self.system:
            self.system.join(timeout=5)
        for s in self.stages:
            try:
                s.stop()
            except Exception as exc:
                s.error = repr(exc)
        for s in self.stages:
            if s.is_alive():
                s.join(timeout=3)
        if self.metrics:
            self.metrics.stop()

    def _manifest_extra(self):
        live = dict(self._live)
        live["models"] = self._models
        errors = {s.name: s.error for s in self.stages if s.error}
        if errors:
            live["stage_errors"] = errors
        if self.notes:
            live["notes"] = self.notes
        extra = {"profile": self.profile, "live": live}
        if self.system:
            info = self.system.info()
            extra["audio_system"] = info
            extra["audio_system_offset"] = info.get("start_offset")
        return extra


def build_parser():
    p = argparse.ArgumentParser(
        prog="understudy live",
        description="Record with live vision, transcription and speaker attribution.")
    p.add_argument("--profile", choices=sorted(PROFILES), default="meeting")
    p.add_argument("--mode", choices=["fixed", "dedup"], default=None,
                   help="override the profile's capture mode")
    p.add_argument("--fps", type=float, default=None, help="override the profile's rate")
    p.add_argument("--out", default=DEFAULT_ROOT)
    p.add_argument("--name", default=None)
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument("--quality", type=int, default=80,
                   help="JPEG quality of saved frames (80 is about half the size of 92)")
    p.add_argument("--keys", choices=["metadata", "full"], default="metadata")
    p.add_argument("--duration", type=float, default=None,
                   help="stop automatically after N seconds")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--audio-device", default=None)
    p.add_argument("--system-audio", dest="system_audio", action="store_true",
                   default=None, help="also record what the computer plays")
    p.add_argument("--no-system-audio", dest="system_audio", action="store_false")
    p.add_argument("--system-audio-device", default=None)
    p.add_argument("--no-indicator", action="store_true",
                   help="skip the REC window (meeting profile)")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--no-transcribe", action="store_true")
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--vision-backend", choices=["ollama", "openai"], default="ollama")
    p.add_argument("--vision-model", default=None)
    p.add_argument("--vision-url", default=None)
    p.add_argument("--vision-mode", choices=["describe", "diff"], default="describe")
    p.add_argument("--vision-max-rate", type=float, default=1.0,
                   help="max vision analyses per second")
    p.add_argument("--vision-timeout", type=float, default=120.0,
                   help="seconds to wait for one frame's analysis")
    p.add_argument("--vision-max-side", type=int, default=1280,
                   help="downscale frames to this many pixels on the long side")
    p.add_argument("--vision-context", default=None,
                   help='one line of domain context, e.g. "semiconductor wafer prober software"')
    p.add_argument("--vision-on-change", action="store_true",
                   help="only analyse frames where the screen changed (logged 'unchanged' otherwise)")
    p.add_argument("--whisper-model", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        rec = LiveRecorder(
            out=a.out, name=a.name, profile=a.profile, mode=a.mode, fps=a.fps,
            monitor=a.monitor, quality=a.quality, audio=not a.no_audio,
            audio_device=a.audio_device, keys=a.keys, system_audio=a.system_audio,
            system_audio_device=a.system_audio_device, vision=not a.no_vision,
            transcribe=not a.no_transcribe, diarize=not a.no_diarize,
            vision_backend=a.vision_backend, vision_model=a.vision_model,
            vision_url=a.vision_url, vision_mode=a.vision_mode,
            vision_max_rate=a.vision_max_rate,
            vision_timeout=a.vision_timeout, vision_max_side=a.vision_max_side,
            vision_on_change=a.vision_on_change, vision_context=a.vision_context,
            whisper_model=a.whisper_model,
            indicator=False if a.no_indicator else None)
    except ValueError as exc:
        sys.exit(str(exc))

    stopping = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, lambda *_: stopping.set())
        signal.signal(signal.SIGTERM, lambda *_: stopping.set())

    rec.start()
    print("Live recording (%s) -> %s" % (a.profile, rec.dir))
    print("Press Ctrl-C to stop.")
    last = 0.0
    try:
        while not stopping.is_set():
            time.sleep(0.1)
            rec.pump()
            t = rec.clock.t()
            if a.duration is not None and t >= a.duration:
                break
            if t - last >= 1.0:
                last = t
                print("\r" + rec.status_line() + "   ", end="", flush=True)
    finally:
        print()
        rec.stop()
    _, kept, _ = rec.status()
    print("Saved %d frames to %s" % (kept, rec.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
