"""Screen capture with live deduplication.

The loop samples at a fixed base rate (2 fps by default) but only *keeps* a
frame when something meaningful changed since the last kept frame -- comparing
against the last kept frame rather than the last sampled one, so a slow fade or
a progress bar creeping forward cannot drift past the threshold one
imperceptible step at a time.

Dropped frames are not lost information: the previous kept frame's `duration`
is extended, so a ten-minute pause reads as one frame held for ten minutes
rather than as a gap.

Three things can make a frame worth keeping, and they are recorded separately
because they mean different things downstream:

  window-switch  the frontmost window changed identity. In an application whose
                 features live in many windows, this is the strongest available
                 marker of one step ending and another beginning.
  click / input  the user did something. Kept even when the screen did not
                 visibly react, because the *attempt* is part of the workflow.
  diff           enough pixels changed on their own.

And one thing is deliberately NOT worth keeping: a window being dragged or
resized. That redraws most of the screen while conveying nothing, so a move is
suppressed until the bounds settle, then recorded as a single frame. Without
this, a workflow that shuffles windows would drown in near-duplicates.

Change detection runs on a 64x64 grid of cell means -- coarse on purpose, so
that caret blink and pointer movement register as nothing, while the set of
changed cells doubles as a bounding box of *where* the screen changed. That box
is what lets a later stage crop to the part that matters.
"""
import threading

import mss
import numpy as np
from PIL import Image

GRID = 64            # change-detection grid, GRID x GRID cells
CELL_DELTA = 10      # per-cell mean-luma change (0-255) counted as "changed"
POST_ACTION_DELAY = 0.4   # capture this long after a click, to catch the result
MOVE_SETTLE = 0.5    # bounds must hold still this long before a move is kept


class CaptureLoop(threading.Thread):
    def __init__(self, clock, session, monitor=1, fps=2.0, min_change=0.004,
                 heartbeat=120.0, jpeg_quality=92, frontmost=None):
        super().__init__(daemon=True)
        self.clock = clock
        self.session = session
        self.monitor = monitor
        self.interval = 1.0 / float(fps)
        self.min_change = min_change
        self.heartbeat = heartbeat
        self.jpeg_quality = jpeg_quality
        self.frontmost = frontmost or (lambda: {})

        self._stopping = threading.Event()
        self._wake = threading.Event()
        # reason -> due time. A dict rather than a queue so that repeated
        # requests of the same kind coalesce: holding a key down or spinning a
        # scroll wheel should schedule one frame, not one per event.
        self._forced = {}
        self._forced_lock = threading.Lock()

        self._prev_grid = None
        self._last_kept = None            # record dict of the last kept frame
        self._last_kept_t = -1e9
        self._win_id = None               # window identity as of the last sample
        self._win_bounds = None
        self._move_from = None            # bounds a move started from
        self._settle_at = None            # when the current move may be kept
        self.kept = 0
        self.sampled = 0
        self.moves_suppressed = 0
        self.geometry = None

    # -- external triggers ----------------------------------------------
    def force(self, reason, delay=0.0):
        """Request a keyframe, bypassing the dedup threshold.

        Called from the event thread. A click schedules two: one immediately
        (what the user was looking at when they aimed) and one after a short
        delay (what the click did).
        """
        due = self.clock.t() + delay
        with self._forced_lock:
            self._forced[reason] = min(self._forced.get(reason, due), due)
        self._wake.set()

    def on_activity(self, kind):
        if kind == "click":
            self.force("click")
            self.force("post-click", POST_ACTION_DELAY)
        elif kind in ("scroll", "key", "typing-end"):
            # Note there is no per-keystroke case. Typing a filename produces a
            # frame per character, each below the pixel threshold and identical
            # to the last; the informative moment is when the run *ends* and the
            # field is full.
            self.force("input", POST_ACTION_DELAY)

    def stop(self):
        self._stopping.set()
        self._wake.set()

    # -- analysis --------------------------------------------------------
    def _signature(self, img):
        """GRID x GRID array of cell mean luma."""
        small = img.convert("L").resize((GRID, GRID), Image.BOX)
        return np.asarray(small, dtype=np.int16)

    def _change(self, grid):
        """Return (changed_fraction, cell_bbox or None)."""
        if self._prev_grid is None:
            return 1.0, None
        diff = np.abs(grid - self._prev_grid) > CELL_DELTA
        frac = float(diff.mean())
        if not diff.any():
            return frac, None
        ys, xs = np.nonzero(diff)
        return frac, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    def _cells_to_pixels(self, bbox, w, h):
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        sx, sy = w / GRID, h / GRID
        return [int(x0 * sx), int(y0 * sy), int(round(x1 * sx)), int(round(y1 * sy))]

    def _window_reason(self, win, t):
        """Classify what the frontmost window did since the last sample.

        Returns a keep-reason, or None to keep sampling. Bounds changing on the
        *same* window is a drag or resize: suppressed until it settles, so one
        frame records where the window ended up instead of twenty recording its
        journey there.
        """
        wid, bounds = win.get("window_id"), win.get("bounds")
        prev_id, prev_bounds = self._win_id, self._win_bounds
        self._win_id, self._win_bounds = wid, bounds

        if prev_id is None:
            return None                       # first sample; nothing to compare
        if wid != prev_id:
            self._move_from = self._settle_at = None
            return "window-switch"

        if bounds != prev_bounds and bounds is not None:
            if self._move_from is None:
                self._move_from = prev_bounds
            self._settle_at = t + MOVE_SETTLE
            self.moves_suppressed += 1
            return "suppress"

        if self._move_from is not None and t >= (self._settle_at or 0):
            self._move_from = self._settle_at = None
            return "window-move"
        if self._move_from is not None:
            return "suppress"                 # still settling
        return None

    # -- main loop -------------------------------------------------------
    def _due_forced(self, now):
        """Pop every forced request that has come due; return its reason."""
        with self._forced_lock:
            due = [r for r, at in self._forced.items() if at <= now]
            for r in due:
                del self._forced[r]
        # A click and its follow-up can both come due on the same sample; the
        # click is the more specific description of why this frame exists.
        for preferred in ("click", "post-click", "input"):
            if preferred in due:
                return preferred
        return due[0] if due else None

    def _next_forced(self):
        with self._forced_lock:
            return min(self._forced.values()) if self._forced else None

    def run(self):
        with mss.mss() as sct:
            mon = sct.monitors[self.monitor]
            self.geometry = dict(mon)
            next_sample = self.clock.t()

            while not self._stopping.is_set():
                now = self.clock.t()
                nf = self._next_forced()
                wake_at = min(next_sample, nf) if nf is not None else next_sample
                delay = wake_at - now
                if delay > 0:
                    self._wake.wait(timeout=delay)
                    self._wake.clear()
                    continue  # re-evaluate: a click may have arrived meanwhile

                t = self.clock.t()
                forced = self._due_forced(t)
                if t >= next_sample:
                    # Re-anchor rather than accumulate, so a stall (a laptop
                    # sleeping, a slow grab) cannot leave the loop owing a
                    # burst of back-to-back frames it then fires at once.
                    next_sample = t + self.interval

                # Query the window *before* grabbing. Querying after leaves a
                # gap in which the user can switch windows, producing a frame
                # whose pixels show the old window but whose metadata names the
                # new one -- which makes a step cite text it does not contain.
                # Sampling first means a switch is noticed one frame later,
                # which is harmless, and the two always agree.
                win = self.frontmost()
                raw = sct.grab(mon)
                self.sampled += 1
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                wreason = self._window_reason(win, t)

                grid = self._signature(img)
                frac, cell_bbox = self._change(grid)
                idle = t - self._last_kept_t

                # A window being dragged suppresses pixel-diff and window
                # reasons, but never an explicit user action or the heartbeat.
                if wreason == "suppress":
                    if forced is None and idle < self.heartbeat:
                        continue
                    wreason = None

                reason = forced or wreason
                if reason is None:
                    if frac >= self.min_change:
                        reason = "diff"
                    elif idle >= self.heartbeat:
                        reason = "heartbeat"
                    else:
                        continue
                if self._last_kept is None:
                    reason = "first"

                self._prev_grid = grid
                self._write_frame(img, t, reason, frac, cell_bbox, win)

        self._close_last()

    def _write_frame(self, img, t, reason, frac, cell_bbox, win):
        idx = self.kept
        img.save(self.session.frame_path(idx), "JPEG",
                 quality=self.jpeg_quality, subsampling=0)
        rec = {
            "idx": idx, "t": round(t, 3), "file": self.session.frame_rel(idx),
            "reason": reason, "change": round(frac, 5),
            "bbox": self._cells_to_pixels(cell_bbox, img.width, img.height),
            "app": win.get("app", ""), "window": win.get("title", ""),
            "window_id": win.get("window_id"), "window_bounds": win.get("bounds"),
            "w": img.width, "h": img.height,
        }
        if reason == "window-move" and self._move_from:
            rec["moved_from"] = self._move_from
        self._close_last(until=t)
        self._last_kept = rec
        self._last_kept_t = t
        self.kept += 1

    def _close_last(self, until=None):
        """Write the previous kept frame once its duration is known."""
        if self._last_kept is None:
            return
        end = self.clock.t() if until is None else until
        self._last_kept["duration"] = round(end - self._last_kept["t"], 3)
        self.session.frames.write(self._last_kept)
        if until is None:
            self._last_kept = None
