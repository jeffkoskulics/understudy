"""Session directory layout and append-only stream writers."""
import json
import os
import threading
import datetime


class JsonlWriter:
    """Append-only JSONL sink, flushed per record.

    Flushing every line costs little at our rates (a few hundred records a
    minute) and means a session that dies with the app -- a crash, a forced
    quit, a lid close -- is still fully readable up to the last instant.
    """

    def __init__(self, path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        with self._lock:
            self._fh.close()


class Session:
    def __init__(self, root: str, name: str = None):
        name = name or "session-" + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.name = name
        self.dir = os.path.join(os.path.expanduser(root), name)
        self.frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.frames = JsonlWriter(os.path.join(self.dir, "frames.jsonl"))
        self.events = JsonlWriter(os.path.join(self.dir, "events.jsonl"))

    @property
    def audio_path(self):
        return os.path.join(self.dir, "audio.wav")

    def frame_path(self, idx: int) -> str:
        return os.path.join(self.frames_dir, "%06d.jpg" % idx)

    def frame_rel(self, idx: int) -> str:
        return "frames/%06d.jpg" % idx

    def write_manifest(self, data: dict):
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def close(self):
        self.frames.close()
        self.events.close()
