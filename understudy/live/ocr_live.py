"""Live OCR: every saved frame's on-screen text, as a set.

The vision model says what the screen *is*; OCR says exactly what it *reads*.
On a 1B model the two diverge fast -- the second real session had the model
repeating one generic sentence while the screen showed alarm names, port
numbers and job IDs. OCR recovers those verbatim, cheaply, on the CPU.

Order is deliberately thrown away. Screen text is a layout, not prose; reading
order across panels is guesswork, and what downstream consumers need is "which
strings were visible, and when did each first appear". So each frame yields a
sorted, de-duplicated set of lines, and a record of what was added and removed
since the previous frame -- a dialog opening shows up as a burst of `added`.

At stop, `ocr_text.json` holds the union for the whole session: every distinct
line with the time it was first and last seen and how many frames showed it.

Uses the same long-lived platform helper as the batch `understudy ocr`
(Windows.Media.Ocr on Windows, Vision on macOS): one process for the whole
session, paths in on stdin, one JSON line out per image.
"""
import json
import os
import subprocess
import threading
import time

from .bus import Stage
from .. import ocr as batch_ocr

MIN_LEN = 2          # single characters are almost always icon noise


def normalize(text):
    return " ".join(text.split())


class LiveOCR(Stage):
    name = "ocr"

    def __init__(self, clock, session, metrics=None, inbox_size=4, command=None):
        super().__init__(clock, session, metrics=metrics, inbox_size=inbox_size)
        self.dir = session.dir
        self.out = session.writer("live_ocr.jsonl")
        if command is None:
            command = batch_ocr._command(batch_ocr.helper_path())
        self._cmd = command
        self._proc = None
        self._prev = set()
        self._seen = {}             # text -> {"first": t, "last": t, "frames": n}
        self._lock = threading.Lock()

    def probe(self):
        """Start the helper now so a missing dependency fails before recording."""
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(1.0)     # an import error (e.g. winrt missing) exits at once
        if self._proc.poll() is not None:
            err = self._proc.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError("OCR helper failed to start: " + (err[-300:] or "no output"))

    def on_frame(self, record, path):
        self.offer((record, path))

    def _ask(self, path):
        if self._proc is None or self._proc.poll() is not None:
            self.probe()
        self._proc.stdin.write((path + "\n").encode("utf-8"))
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read().decode("utf-8", "replace")[-300:]
            raise RuntimeError("OCR helper exited: " + err.strip())
        return json.loads(line)

    def step(self, item):
        record, path = item
        t = record.get("t")
        t0 = time.perf_counter()
        res = self._ask(path)
        latency = round(time.perf_counter() - t0, 3)
        if res.get("error"):
            self.out.write({"t": t, "frame": record.get("file"), "error": res["error"],
                            "latency_s": latency})
            return
        lines = {normalize(l.get("text", "")) for l in res.get("lines", [])}
        lines = {l for l in lines if len(l) >= MIN_LEN}
        added, removed = lines - self._prev, self._prev - lines
        self._prev = lines
        with self._lock:
            for s in lines:
                e = self._seen.setdefault(s, {"first": t, "last": t, "frames": 0})
                e["last"] = t
                e["frames"] += 1
        self.out.write({"t": t, "frame": record.get("file"), "latency_s": latency,
                        "text": sorted(lines), "added": sorted(added),
                        "removed": sorted(removed)})
        if self.metrics:
            self.metrics.timing("ocr.latency", latency)

    def stop(self):
        super().stop()

    def finish(self):
        """After the worker has joined: close the helper and write the union."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        with self._lock:
            seen = dict(sorted(self._seen.items()))
        with open(os.path.join(self.dir, "ocr_text.json"), "w", encoding="utf-8") as fh:
            json.dump({"count": len(seen), "text": seen}, fh, indent=1, ensure_ascii=False)
