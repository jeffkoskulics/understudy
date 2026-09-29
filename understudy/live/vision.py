"""Live vision analysis stage. Writes vision.jsonl (see docs/CONTRACTS.md).

Items offered are `(record, path)` frames from capture (`on_frame` callback).
"""
import base64
import io
import os
import queue
import threading
import time

from PIL import Image

from ..session import JsonlWriter
from .bus import Stage
from .backends.vision_local import (BackendError, OllamaBackend,  # noqa: F401
                                    OpenAICompatBackend, RemoteBackend)

DESCRIBE_PROMPT = (
    "This is a screen share of an application. In 2-4 sentences describe: "
    "the application and view shown, key UI elements (menus, dialogs, "
    "selected items), the important visible text (quote it exactly), and "
    "what the user appears to be doing. Be concise and literal; do not guess "
    "beyond what is visible."
)

DIFF_PROMPT = (
    "Two screenshots of the same screen: first the PREVIOUS frame, then the "
    "CURRENT frame.{hint} Describe only what changed: UI elements that "
    "appeared, disappeared or moved, text that was typed or edited (quote "
    "it), and the action the user just took. If nothing meaningful changed, "
    "say 'no meaningful change'. Be concise (1-3 sentences)."
)


def _hint(bbox):
    if not bbox:
        return ""
    return (" The change is concentrated in the region "
            f"(x0,y0,x1,y1) = {list(bbox)} of the current frame.")


def encode_image(path, max_side=1280, quality=80):
    """Downscale so the longest side <= max_side; return base64 JPEG."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max_side / max(im.size)
        if scale < 1:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


class VisionAnalyzer(Stage):
    name = "vision"

    def __init__(self, clock, session, backend=None, mode="describe",
                 max_rate=1.0, max_side=1280, metrics=None, inbox_size=2):
        super().__init__(clock, session, metrics=metrics, inbox_size=inbox_size)
        if mode not in ("describe", "diff"):
            raise ValueError("mode must be 'describe' or 'diff'")
        self.backend = backend or OllamaBackend()
        self.mode, self.max_rate, self.max_side = mode, max_rate, max_side
        self.out = JsonlWriter(os.path.join(session.dir, "vision.jsonl"))
        self._lock = threading.Lock()
        self._last_start = None      # monotonic start of last analysis
        self._prev = None            # (record, path) last analysed frame

    def probe(self):
        return self.backend.probe()

    def on_frame(self, record, path):
        self.offer((record, path))

    def offer(self, item):
        """Like Stage.offer, but the evicted frame gets a skipped record."""
        try:
            self.inbox.put_nowait(item)
        except queue.Full:
            try:
                old = self.inbox.get_nowait()
                self.dropped += 1
                if self.metrics:
                    self.metrics.count(self.name + ".dropped")
                self._skip(old[0], "queue-full")
            except queue.Empty:
                pass
            try:
                self.inbox.put_nowait(item)
            except queue.Full:
                self._skip(item[0], "queue-full")

    def _write(self, rec):
        with self._lock:
            self.out.write(rec)

    def _skip(self, record, reason):
        self._write({"t": record.get("t"), "frame": record.get("file"),
                     "mode": self.mode, "backend": self.backend.name,
                     "model": self.backend.model, "skipped": reason})

    def step(self, item):
        record, path = item
        now = time.monotonic()
        if (self.max_rate and self._last_start is not None
                and now - self._last_start < 1.0 / self.max_rate):
            self._skip(record, "max-rate")
            return
        self._last_start = now
        diff = self.mode == "diff" and self._prev is not None
        rec = {"t": record.get("t"), "frame": record.get("file"),
               "mode": "diff" if diff else "describe",
               "backend": self.backend.name, "model": self.backend.model}
        t0 = time.perf_counter()
        try:
            images = []
            if diff:
                rec["prev_frame"] = self._prev[0].get("file")
                images.append(encode_image(self._prev[1], self.max_side))
                prompt = DIFF_PROMPT.format(hint=_hint(record.get("bbox")))
            else:
                prompt = DESCRIBE_PROMPT
            images.append(encode_image(path, self.max_side))
            text = self.backend.run({"prompt": prompt, "images": images})["text"]
        except Exception as exc:
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            rec["text"] = ""
            rec["skipped"] = "error: " + repr(exc)[:200]
            self.error = repr(exc)
            self._write(rec)
            if self.metrics:
                self.metrics.count(self.name + ".errors")
            return
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        rec["text"] = text
        self._write(rec)
        self._prev = (record, path)
        if self.metrics:
            self.metrics.timing(self.name + ".latency", rec["latency_s"])

    def stop(self):
        super().stop()
        # log unprocessed frames as skipped so the record is complete
        while True:
            try:
                self._skip(self.inbox.get_nowait()[0], "shutdown")
            except queue.Empty:
                break
        with self._lock:
            self.out.close()
