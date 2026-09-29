"""Shared plumbing for live stages. Every workstream builds against this file;
change it only through the integrator. See docs/CONTRACTS.md."""
import queue
import threading
import time


class Stage(threading.Thread):
    """A live stage: consumes from `inbox` (may be None for sources), writes
    JSONL through the session, reports timings to `metrics` (may be None).

    Subclasses implement `step(item)`; sources override `run()` instead.
    `stop()` must return within ~2 s and must flush output.
    """

    name = "stage"

    def __init__(self, clock, session, metrics=None, inbox_size=8):
        super().__init__(daemon=True)
        self.clock = clock
        self.session = session
        self.metrics = metrics
        self.inbox = queue.Queue(maxsize=inbox_size)
        self.dropped = 0
        self.error = None
        self._stopping = threading.Event()

    def offer(self, item):
        """Non-blocking enqueue. Drops oldest when full, counts and reports."""
        try:
            self.inbox.put_nowait(item)
        except queue.Full:
            try:
                self.inbox.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            if self.metrics:
                self.metrics.count(self.name + ".dropped")
            self.inbox.put_nowait(item)

    def run(self):
        while not self._stopping.is_set():
            try:
                item = self.inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                self.step(item)
            except Exception as exc:  # a stage must never kill the recording
                self.error = repr(exc)
                if self.metrics:
                    self.metrics.count(self.name + ".errors")
            if self.metrics:
                self.metrics.timing(self.name + ".step", time.perf_counter() - t0)
                self.metrics.gauge(self.name + ".queue", self.inbox.qsize())

    def step(self, item):
        raise NotImplementedError

    def stop(self):
        self._stopping.set()


class NullMetrics:
    """Stand-in until metrics.py lands; same interface."""

    def timing(self, name, seconds): pass
    def gauge(self, name, value): pass
    def count(self, name, n=1): pass
