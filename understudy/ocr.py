"""Run the platform OCR helper across a session's frames.

The helper is a long-lived process reading paths on stdin rather than a
per-image invocation, because process startup dominates at these image sizes.
Several are run in parallel: a single helper measured 4.5 s/frame at 38% CPU
on the development machine, which would make OCR slower than the recording it
is transcribing.

Results are keyed by file and written in frame order regardless of which
worker finished first, so the output lines up with frames.jsonl.
"""
import json
import os
import subprocess
import sys
import threading

HELPERS = {"darwin": "ocr_mac", "win32": "ocr_win.py"}


def helper_path():
    name = HELPERS.get(sys.platform)
    if not name:
        raise RuntimeError("no OCR helper for platform %r" % sys.platform)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "helpers", name)
    if not os.path.exists(path):
        if sys.platform == "win32":
            raise RuntimeError("OCR helper missing: %s" % path)
        raise RuntimeError(
            "OCR helper not built: %s\n  build it with:\n"
            "    swiftc -O helpers/ocr_mac.swift -o helpers/ocr_mac" % path)
    return path


def _command(exe):
    # A .py helper runs under this same interpreter, so it sees the same venv
    # (and the winrt packages installed into it).
    return [sys.executable, exe] if exe.endswith(".py") else [exe]


def _worker(exe, paths, out):
    proc = subprocess.Popen(_command(exe), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(("\n".join(paths) + "\n").encode())
    for line in stdout.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        out[rec.get("file")] = rec


def ocr_frames(paths, workers=None):
    """OCR every path; return a list of records in the order given."""
    if not paths:
        return []
    exe = helper_path()
    workers = workers or min(len(paths), os.cpu_count() or 2)
    # Round-robin rather than contiguous blocks: frames vary a lot in how much
    # text they hold, and interleaving keeps the workers evenly loaded.
    chunks = [paths[i::workers] for i in range(workers)]
    results = {}
    threads = [threading.Thread(target=_worker, args=(exe, c, results))
               for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [results.get(p, {"file": p, "error": "no result"}) for p in paths]


def ocr_session(session_dir, workers=None):
    frames = [json.loads(l) for l in open(os.path.join(session_dir, "frames.jsonl"))]
    paths = [os.path.join(session_dir, f["file"]) for f in frames]
    records = ocr_frames(paths, workers)
    out = os.path.join(session_dir, "ocr.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for frame, rec in zip(frames, records):
            rec["idx"] = frame["idx"]
            fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    return out


if __name__ == "__main__":
    print(ocr_session(sys.argv[1]))
