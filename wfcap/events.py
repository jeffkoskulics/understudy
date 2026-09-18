"""Input event capture.

Pixel diffing alone cannot tell you that a button was pressed when the button
did not visibly change, and it can never tell you *where* the user was aiming.
The click coordinate is the single most valuable number in the session: paired
with OCR bounding boxes it recovers the label of the control that was used.

Keystrokes are recorded as metadata by default -- that a field was typed into,
for how long, how many characters -- plus the structural keys (Tab, Enter,
Backspace, shortcuts) that carry workflow meaning without carrying content.
The typed values themselves come back from OCR of the resulting frame, so
`full` mode exists only for cases where that fails.
"""
import threading
import time

from pynput import mouse, keyboard

TYPING_IDLE_FLUSH = 1.2  # seconds of no keypress that ends a typing run
SCROLL_COALESCE = 0.4    # scroll ticks within this window become one record

# Named keys that describe workflow structure rather than content.
STRUCTURAL = {
    "enter", "tab", "backspace", "delete", "esc", "up", "down", "left",
    "right", "home", "end", "page_up", "page_down", "f1", "f2", "f3", "f4",
    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}


class EventRecorder:
    def __init__(self, clock, sink, frontmost, key_mode="metadata", on_activity=None):
        self.clock = clock
        self.sink = sink
        self.frontmost = frontmost
        self.key_mode = key_mode
        self.on_activity = on_activity or (lambda kind: None)
        self._lock = threading.Lock()
        self._run = None           # in-progress typing run
        self._scroll = None        # in-progress scroll coalesce
        self._mouse_pos = (0, 0)
        self._listeners = []
        self._timer = None

    # -- emit ------------------------------------------------------------
    def _emit(self, record):
        win = self.frontmost()
        record["app"] = win.get("app", "")
        record["window"] = win.get("title", "")
        record["window_id"] = win.get("window_id")
        self.sink.write(record)

    # -- typing runs -----------------------------------------------------
    def _flush_run(self, now=None):
        """Close the open typing run. Caller must hold the lock."""
        run = self._run
        if not run:
            return
        self._run = None
        rec = {
            "t": round(run["t0"], 3),
            "type": "typing",
            "duration": round(run["last"] - run["t0"], 3),
            "chars": run["chars"],
            "keys": run["keys"],
        }
        if self.key_mode == "full" and run["text"]:
            rec["text"] = "".join(run["text"])
        self._emit(rec)
        # Signal only now, not per keystroke: the frame worth keeping is the
        # one showing the completed value, not each character arriving.
        self.on_activity("typing-end")

    def _arm_flush_timer(self):
        """(Re)start the idle timer that closes a typing run mid-stream.

        Without this a run is only written when the next click arrives, so a
        session ending on a long typing burst would lose its final record.
        """
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(TYPING_IDLE_FLUSH, self._idle_flush)
        self._timer.daemon = True
        self._timer.start()

    def _idle_flush(self):
        with self._lock:
            self._flush_run()

    def _flush_scroll(self):
        """Close the open scroll coalesce. Caller must hold the lock."""
        s = self._scroll
        if not s:
            return
        self._scroll = None
        self._emit({
            "t": round(s["t0"], 3), "type": "scroll",
            "x": s["x"], "y": s["y"], "dx": s["dx"], "dy": s["dy"],
            "duration": round(s["last"] - s["t0"], 3),
        })

    # -- handlers --------------------------------------------------------
    def _on_move(self, x, y):
        # Not recorded: pointer paths are high volume and low information.
        # Kept only so a click knows where it happened on platforms that do
        # not hand coordinates to the click callback.
        self._mouse_pos = (int(x), int(y))

    def _on_click(self, x, y, button, pressed):
        if not pressed:
            return
        t = self.clock.t()
        with self._lock:
            self._flush_run()
            self._flush_scroll()
            self._emit({
                "t": round(t, 3), "type": "click",
                "button": str(button).rsplit(".", 1)[-1],
                "x": int(x), "y": int(y),
            })
        self.on_activity("click")

    def _on_scroll(self, x, y, dx, dy):
        t = self.clock.t()
        with self._lock:
            s = self._scroll
            if s and t - s["last"] < SCROLL_COALESCE:
                s["dx"] += dx
                s["dy"] += dy
                s["last"] = t
            else:
                self._flush_scroll()
                self._scroll = {"t0": t, "last": t, "x": int(x), "y": int(y),
                                "dx": dx, "dy": dy}
        self.on_activity("scroll")

    def _on_press(self, key):
        t = self.clock.t()
        name = getattr(key, "name", None)
        char = getattr(key, "char", None)
        with self._lock:
            self._flush_scroll()
            if name and name.lower() in STRUCTURAL:
                self._flush_run()
                self._emit({"t": round(t, 3), "type": "key", "key": name.lower()})
                self.on_activity("key")
                return
            if char is None:
                return  # bare modifier press; the next real key carries it
            run = self._run
            if run is None or t - run["last"] > TYPING_IDLE_FLUSH:
                self._flush_run()
                run = self._run = {"t0": t, "last": t, "chars": 0,
                                   "keys": 0, "text": []}
            run["last"] = t
            run["chars"] += 1
            run["keys"] += 1
            if self.key_mode == "full":
                run["text"].append(char)
            self._arm_flush_timer()

    # -- lifecycle -------------------------------------------------------
    def start(self):
        ml = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll,
                            on_move=self._on_move)
        kl = keyboard.Listener(on_press=self._on_press)
        for l in (ml, kl):
            l.daemon = True
            l.start()
            self._listeners.append(l)

    def stop(self):
        if self._timer:
            self._timer.cancel()
        for l in self._listeners:
            l.stop()
        with self._lock:
            self._flush_run()
            self._flush_scroll()
