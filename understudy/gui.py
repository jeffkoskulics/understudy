"""A small control panel: pick a monitor, pick a folder, record, hand off.

The command line is fine once you know the four verbs, but the person who owns
the knowledge we are trying to capture is usually not the person who is
comfortable in a terminal, and they are being asked to do something slightly
awkward already -- narrate their own work while a stranger watches. So the
window states the two decisions that actually matter (which screen, where the
files go), and after that offers one button per step in the order they happen.

The hand-off buttons put text on the clipboard rather than opening anything,
because the destination is a chat window in a browser: there is nothing to
open, only something to paste. The audio file is the one exception -- it has to
be dragged in -- so that button reveals it in the file manager instead.

Tkinter, deliberately: it ships with Python on all three platforms, so the GUI
adds nothing to requirements.txt and cannot become the reason an install fails
on the machine of the one expert whose time we got.
"""
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .record import DEFAULT_ROOT, Recorder

PREVIEW_W = 260          # monitor thumbnail, max width in px
PREVIEW_H = 165          # and max height -- a portrait screen must not stretch the window


# -- system queries ------------------------------------------------------
# Each of these is a question the panel asks the OS and must survive a "no":
# a machine with no microphone, or a screen-capture permission not yet
# granted, should still show a usable window that says what is wrong.

def list_monitors():
    """[(index, label, geometry)] -- mss index 0 is every screen combined."""
    try:
        import mss
    except Exception as exc:                        # pragma: no cover
        return [], "screen capture unavailable: %s" % exc
    try:
        with mss.mss() as sct:
            mons = list(sct.monitors)
    except Exception as exc:
        return [], "cannot enumerate monitors: %s" % exc

    out = []
    for i, m in enumerate(mons):
        size = "%d x %d" % (m["width"], m["height"])
        if i == 0:
            label = "All monitors combined - %s" % size
        else:
            label = "Monitor %d - %s at (%d, %d)" % (i, size, m["left"], m["top"])
            if len(mons) > 1 and i == 1:
                label += "  [primary]"
        out.append((i, label, m))
    return out, None


def list_audio_devices():
    """[(device index or None, label)], default first."""
    devices = [(None, "System default input")]
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append((i, "%d: %s" % (i, d["name"])))
    except Exception:
        # No portaudio, no devices, no permission -- recording without audio is
        # still worth having, so this is not worth an error dialog.
        pass
    return devices


def reveal(path):
    """Show a file in the platform file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        elif os.name == "nt":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except Exception as exc:
        messagebox.showerror("Understudy", "Could not open %s\n\n%s" % (path, exc))


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=12)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.monitors, self.monitor_error = list_monitors()
        self.recorder = None
        self.session_dir = tk.StringVar(value="")
        self.status = tk.StringVar(value="Ready.")
        self._preview_image = None      # kept alive; Tk does not own the ref
        self._working = False
        self._results = queue.Queue()   # background work -> main thread

        self._build_source()
        self._build_destination()
        self._build_controls()
        self._build_handoff()

        ttk.Label(self, textvariable=self.status, foreground="#444").grid(
            row=4, column=0, sticky="w", pady=(10, 0))
        self._refresh_preview()
        self._sync_buttons()
        self._poll_results()

    # -- source ----------------------------------------------------------
    def _build_source(self):
        box = ttk.LabelFrame(self, text="1. Source", padding=10)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Record").grid(row=0, column=0, sticky="w")
        self.monitor_box = ttk.Combobox(box, state="readonly", width=44,
                                        values=[m[1] for m in self.monitors])
        self.monitor_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        # Index 1 is the primary screen; index 0 (everything at once) is
        # available but a poor default -- OCR over two screens of text costs
        # twice as much and answers a question nobody asked.
        if len(self.monitors) > 1:
            self.monitor_box.current(1)
        elif self.monitors:
            self.monitor_box.current(0)
        self.monitor_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        self.preview = ttk.Label(box, relief="solid", borderwidth=1,
                                 anchor="center", text="no preview")
        self.preview.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Button(box, text="Refresh", width=9,
                   command=self._refresh_monitors).grid(row=1, column=0, sticky="nw",
                                                        pady=(8, 0))

        ttk.Label(box, text="Microphone").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.audio_devices = list_audio_devices()
        self.audio_box = ttk.Combobox(box, state="readonly", width=44,
                                      values=[d[1] for d in self.audio_devices] +
                                             ["No audio"])
        self.audio_box.current(0)
        self.audio_box.grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))

        opts = ttk.Frame(box)
        opts.grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(10, 0))
        ttk.Label(opts, text="Frames/sec").pack(side="left")
        self.fps = tk.StringVar(value="2")
        ttk.Spinbox(opts, from_=0.5, to=10, increment=0.5, width=5,
                    textvariable=self.fps).pack(side="left", padx=(6, 16))
        self.full_keys = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Record typed characters (sensitive)",
                        variable=self.full_keys).pack(side="left")

        if self.monitor_error:
            ttk.Label(box, text=self.monitor_error, foreground="#a00").grid(
                row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _refresh_monitors(self):
        self.monitors, self.monitor_error = list_monitors()
        self.monitor_box["values"] = [m[1] for m in self.monitors]
        if self.monitors:
            self.monitor_box.current(1 if len(self.monitors) > 1 else 0)
        self._refresh_preview()

    def _selected_monitor(self):
        i = self.monitor_box.current()
        return self.monitors[i][0] if 0 <= i < len(self.monitors) else 1

    def _refresh_preview(self):
        """Grab the chosen screen once, small, so the choice is unambiguous.

        Two monitors of the same size are indistinguishable from their
        geometry alone, and picking the wrong one is only discovered after the
        expert has finished narrating.
        """
        self.preview.configure(image="", text="no preview")
        self._preview_image = None
        if not self.monitors or self.recorder:
            return
        try:
            import mss
            from PIL import Image, ImageTk
            with mss.mss() as sct:
                mon = sct.monitors[self._selected_monitor()]
                raw = sct.grab(mon)
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            scale = min(PREVIEW_W / img.width, PREVIEW_H / img.height)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            self._preview_image = ImageTk.PhotoImage(img.resize(size))
            self.preview.configure(image=self._preview_image, text="")
        except Exception as exc:
            self.preview.configure(text="preview unavailable\n%s" % exc)

    # -- destination -----------------------------------------------------
    def _build_destination(self):
        box = ttk.LabelFrame(self, text="2. Destination", padding=10)
        box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Folder").grid(row=0, column=0, sticky="w")
        self.out_dir = tk.StringVar(value=os.path.expanduser(DEFAULT_ROOT))
        ttk.Entry(box, textvariable=self.out_dir).grid(row=0, column=1, sticky="ew",
                                                       padx=(8, 8))
        ttk.Button(box, text="Browse...", command=self._browse_out).grid(row=0, column=2)

        ttk.Label(box, text="Name").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.name = tk.StringVar(value="")
        ttk.Entry(box, textvariable=self.name).grid(row=1, column=1, sticky="ew",
                                                    padx=(8, 8), pady=(8, 0))
        ttk.Label(box, text="blank = timestamp", foreground="#666").grid(
            row=1, column=2, sticky="w", pady=(8, 0))

    def _browse_out(self):
        chosen = filedialog.askdirectory(initialdir=self.out_dir.get() or "~")
        if chosen:
            self.out_dir.set(chosen)

    # -- record ----------------------------------------------------------
    def _build_controls(self):
        box = ttk.Frame(self)
        box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.record_btn = ttk.Button(box, text="Start recording", command=self._toggle)
        self.record_btn.pack(side="left")
        self.counter = ttk.Label(box, text="", foreground="#444")
        self.counter.pack(side="left", padx=(12, 0))

    def _toggle(self):
        if self.recorder:
            self._stop()
        else:
            self._start()

    def _start(self):
        out = self.out_dir.get().strip() or DEFAULT_ROOT
        try:
            fps = float(self.fps.get())
        except ValueError:
            messagebox.showerror("Understudy", "Frames/sec must be a number.")
            return
        if not 0.5 <= fps <= 10:
            messagebox.showerror("Understudy", "Frames/sec must be between 0.5 and 10.")
            return

        want_audio = self.audio_box.current() < len(self.audio_devices)
        device = (self.audio_devices[self.audio_box.current()][0]
                  if want_audio else None)
        try:
            self.recorder = Recorder(
                out=out, name=self.name.get().strip() or None, fps=fps,
                monitor=self._selected_monitor(), audio=want_audio,
                audio_device=device,
                keys="full" if self.full_keys.get() else "metadata")
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            messagebox.showerror("Understudy", "Could not start recording:\n\n%s" % exc)
            return

        self.session_dir.set(self.recorder.dir)
        self.record_btn.configure(text="Stop recording")
        self.status.set("Recording. Narrate as you work; stop when the job is done.")
        self._sync_buttons()
        self._tick()

    def _tick(self):
        if not self.recorder:
            return
        t, kept, sampled = self.recorder.status()
        self.counter.configure(text="%d:%02d   %d frames kept / %d sampled"
                                    % (int(t) // 60, int(t) % 60, kept, sampled))
        self.after(250, self._tick)

    def _stop(self):
        rec, self.recorder = self.recorder, None
        path = rec.stop()
        _, kept, sampled = rec.status()
        saved = (1 - kept / sampled) * 100 if sampled else 0
        self.record_btn.configure(text="Start recording")
        self.status.set("Saved %d frames (%.0f%% deduplicated) to %s"
                        % (kept, saved, path))
        self._sync_buttons()
        self._refresh_preview()

    # -- hand-off --------------------------------------------------------
    def _build_handoff(self):
        box = ttk.LabelFrame(self, text="3. Hand off to a chat LLM", padding=10)
        box.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        box.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        row = ttk.Frame(box)
        row.grid(row=0, column=0, sticky="ew")
        row.columnconfigure(1, weight=1)
        ttk.Label(row, text="Session").grid(row=0, column=0, sticky="w")
        ttk.Entry(row, textvariable=self.session_dir).grid(row=0, column=1, sticky="ew",
                                                           padx=(8, 8))
        ttk.Button(row, text="Browse...", command=self._browse_session).grid(row=0, column=2)

        step1 = ttk.Frame(box)
        step1.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.prepare_btn = ttk.Button(step1, text="Prepare audio + prompt",
                                      command=self._prepare)
        self.prepare_btn.pack(side="left")
        self.copy_prompt_btn = ttk.Button(step1, text="Copy prompt",
                                          command=self._copy_prompt)
        self.copy_prompt_btn.pack(side="left", padx=(8, 0))
        self.reveal_audio_btn = ttk.Button(step1, text="Show audio file",
                                           command=self._reveal_audio)
        self.reveal_audio_btn.pack(side="left", padx=(8, 0))

        ttk.Label(box, text="Paste the reply from the chat window here:",
                  foreground="#444").grid(row=2, column=0, sticky="w", pady=(10, 2))
        self.reply = tk.Text(box, height=6, wrap="word")
        self.reply.grid(row=3, column=0, sticky="nsew")
        box.rowconfigure(3, weight=1)

        step2 = ttk.Frame(box)
        step2.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.merge_btn = ttk.Button(step2, text="Merge reply", command=self._merge)
        self.merge_btn.pack(side="left")
        self.pack_btn = ttk.Button(step2, text="Build workflow.md", command=self._pack)
        self.pack_btn.pack(side="left", padx=(8, 0))
        self.copy_workflow_btn = ttk.Button(step2, text="Copy workflow.md",
                                            command=self._copy_workflow)
        self.copy_workflow_btn.pack(side="left", padx=(8, 0))
        self.show_workflow_btn = ttk.Button(step2, text="Show workflow.md",
                                            command=self._reveal_workflow)
        self.show_workflow_btn.pack(side="left", padx=(8, 0))

    def _browse_session(self):
        chosen = filedialog.askdirectory(initialdir=self.out_dir.get() or "~",
                                         title="Choose a session folder")
        if chosen:
            self.session_dir.set(chosen)
            self._sync_buttons()

    def _session(self):
        path = self.session_dir.get().strip()
        if not path or not os.path.isdir(path):
            messagebox.showerror("Understudy", "Choose a session folder first.")
            return None
        return path

    def _copy(self, text, what):
        """Put text on the clipboard and keep it there.

        A Tk clipboard belongs to the process that owns it, so without the
        update() the text vanishes from the system clipboard the moment this
        window closes -- after the user has already switched to the browser to
        paste it.
        """
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self.status.set("Copied %s (%d characters). Paste it into the chat window."
                        % (what, len(text)))

    def _copy_file(self, name, what):
        session = self._session()
        if not session:
            return
        path = os.path.join(session, name)
        if not os.path.exists(path):
            messagebox.showerror("Understudy", "%s does not exist yet." % name)
            return
        with open(path, encoding="utf-8") as fh:
            self._copy(fh.read(), what)

    def _run(self, work, done):
        """Run slow work off the Tk thread, report back on it.

        Packing OCRs every cited frame at a few seconds each, which would
        otherwise freeze the window long enough to look like a crash.

        The worker hands its result back through a queue that the main thread
        polls, rather than calling `after` itself: Tk is only safe to touch
        from the thread that created it, and a worker reaching in directly
        fails outright on some builds and corrupts the interpreter state on
        the rest.
        """
        def worker():
            try:
                self._results.put((done, work(), None))
            except Exception as exc:
                self._results.put((done, None, exc))

        self._busy(True)
        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self):
        """Drain finished background work, on the main thread."""
        while True:
            try:
                done, result, exc = self._results.get_nowait()
            except queue.Empty:
                break
            if exc is not None:
                self._failed(exc)
            else:
                done(result)
        self.after(100, self._poll_results)

    def _busy(self, busy):
        self._working = busy
        self._sync_buttons()
        self.master.configure(cursor="watch" if busy else "")

    def _failed(self, exc):
        self._busy(False)
        self.status.set("Failed: %s" % exc)
        messagebox.showerror("Understudy", str(exc))

    def _prepare(self):
        session = self._session()
        if not session:
            return
        from .handoff import prepare
        self.status.set("Preparing audio and prompt...")

        def done(info):
            self._busy(False)
            self._prepared = info
            if not info["segments"]:
                self.status.set("No narration found in this session; "
                                "you can skip straight to Build workflow.md.")
                return
            self.status.set("%d speech segments (%.0fs). Drag %s into the chat "
                            "window, then paste the prompt."
                            % (info["segments"], info["speech_seconds"],
                               os.path.basename(info["audio"] or "audio.wav")))
        self._run(lambda: prepare(session), done)

    def _copy_prompt(self):
        self._copy_file("transcribe-prompt.txt", "the transcription prompt")

    def _reveal_audio(self):
        session = self._session()
        if not session:
            return
        for name in ("audio.m4a", "audio.wav"):
            path = os.path.join(session, name)
            if os.path.exists(path):
                reveal(path)
                self.status.set("Drag %s into the chat window." % name)
                return
        messagebox.showerror("Understudy", "No audio in this session. "
                                           "Run Prepare first, or skip to packing.")

    def _merge(self):
        session = self._session()
        if not session:
            return
        text = self.reply.get("1.0", "end").strip()
        if not text:
            messagebox.showerror("Understudy", "Paste the chat reply first.")
            return
        from .handoff import merge
        try:
            written, missing = merge(session, text)
        except Exception as exc:
            self._failed(exc)
            return
        if missing:
            # Stated plainly rather than papered over: a gap means narration is
            # absent for that stretch, not shifted onto a neighbouring step.
            self.status.set("Merged %d segments. Missing: %s"
                            % (written, ", ".join(map(str, missing))))
        else:
            self.status.set("Merged %d segments. Now build workflow.md." % written)
        self._sync_buttons()

    def _pack(self):
        session = self._session()
        if not session:
            return
        from .pack import pack
        self.status.set("Reading the frames... this takes a few seconds per step.")

        def done(result):
            self._busy(False)
            md, _js, n = result
            self.status.set("Wrote %s -- %d characters, about %d tokens."
                            % (md, n, n // 4))
            self._sync_buttons()
        self._run(lambda: pack(session), done)

    def _copy_workflow(self):
        self._copy_file("workflow.md", "workflow.md")

    def _reveal_workflow(self):
        session = self._session()
        if not session:
            return
        path = os.path.join(session, "workflow.md")
        if not os.path.exists(path):
            messagebox.showerror("Understudy", "Build workflow.md first.")
            return
        reveal(path)

    # -- enablement ------------------------------------------------------
    def _sync_buttons(self):
        """Grey out anything that cannot work yet, so the order is obvious."""
        busy = self._working
        session = self.session_dir.get().strip()
        have = bool(session) and os.path.isdir(session)
        recording = self.recorder is not None

        def state(widget, on):
            widget.configure(state="normal" if on and not busy else "disabled")

        ready = have and not recording
        state(self.prepare_btn, ready)
        state(self.copy_prompt_btn,
              ready and os.path.exists(os.path.join(session, "transcribe-prompt.txt")))
        state(self.reveal_audio_btn,
              ready and any(os.path.exists(os.path.join(session, n))
                            for n in ("audio.m4a", "audio.wav")))
        state(self.merge_btn, ready)
        state(self.pack_btn, ready)
        exists = ready and os.path.exists(os.path.join(session, "workflow.md"))
        state(self.copy_workflow_btn, exists)
        state(self.show_workflow_btn, exists)
        self.record_btn.configure(state="disabled" if busy else "normal")


def main(argv=None):
    root = tk.Tk()
    root.title("Understudy")
    root.minsize(560, 640)
    app = App(root)

    def on_close():
        if app.recorder and not messagebox.askokcancel(
                "Understudy", "A recording is running. Stop it and quit?"):
            return
        if app.recorder:
            app._stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
