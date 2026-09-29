"""Package a session (or a time slice of it) into an uploadable bundle."""
import argparse
import hashlib
import json
import os
import shutil
import wave
import zipfile

AUDIO_FILES = (("audio.wav", None), ("audio_system.wav", "audio_system_offset"))
PASSTHROUGH = ("manifest.json", "system.json", "participants.json")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _in_range(rec, t_start, t_end):
    t = rec.get("t", rec.get("t0"))
    if t is None:
        return True
    if t_start is not None and t < t_start:
        return False
    return t_end is None or t <= t_end


def _slice_label(t_start, t_end):
    if t_start is None and t_end is None:
        return "full"
    f = lambda v: "end" if v is None else "%d" % round(v)
    return "t%s-%s" % (f(t_start if t_start is not None else 0), f(t_end))


def _audio_offset(manifest, key):
    if key:
        return float(manifest.get(key) or 0.0)
    a = manifest.get("audio") or {}
    return float(a.get("start_offset") or 0.0)


def _clip_wav(src, dst, offset, t_start, t_end):
    with wave.open(src, "rb") as r:
        rate, n = r.getframerate(), r.getnframes()
        first = 0 if t_start is None else max(0, int(round((t_start - offset) * rate)))
        last = n if t_end is None else min(n, int(round((t_end - offset) * rate)))
        if last <= first:
            return False
        r.setpos(first)
        with wave.open(dst, "wb") as w:
            w.setparams(r.getparams())
            left = last - first
            while left > 0:
                data = r.readframes(min(left, 65536))
                if not data:
                    break
                w.writeframes(data)
                left -= 65536
    return True


def build(session_dir, t_start=None, t_end=None, no_audio=False, no_frames=False,
          max_frames=None, out=None):
    """Build a bundle. Returns the staging dir, or the zip path if `out` ends in .zip."""
    session_dir = os.path.abspath(os.path.expanduser(session_dir))
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(session_dir)
    label = _slice_label(t_start, t_end)
    as_zip = bool(out) and out.endswith(".zip")
    if as_zip:
        stage = out[:-4] + ".staging"
    elif out:
        stage = os.path.abspath(out)
    else:
        stage = os.path.join(session_dir, "bundle-" + label)
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    manifest = {}
    mp = os.path.join(session_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as fh:
            manifest = json.load(fh)

    redactions = []
    kept_frames = []

    for name in sorted(os.listdir(session_dir)):
        if not name.endswith(".jsonl"):
            continue
        n_in = n_out = 0
        with open(os.path.join(session_dir, name), encoding="utf-8") as src, \
                open(os.path.join(stage, name), "w", encoding="utf-8") as dst:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                n_in += 1
                if not _in_range(rec, t_start, t_end):
                    continue
                if name == "frames.jsonl" and rec.get("file"):
                    kept_frames.append(rec["file"])
                dst.write(line + "\n")
                n_out += 1
        if n_out != n_in:
            redactions.append({"file": name, "kind": "time-slice", "dropped_records": n_in - n_out})

    for name in PASSTHROUGH:
        p = os.path.join(session_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(stage, name))

    if no_frames:
        redactions.append({"file": "frames/", "kind": "excluded", "reason": "--no-frames"})
    else:
        if max_frames and len(kept_frames) > max_frames:
            step = len(kept_frames) / float(max_frames)
            picked = [kept_frames[int(i * step)] for i in range(max_frames)]
            redactions.append({"file": "frames/", "kind": "downsampled",
                               "kept": len(picked), "of": len(kept_frames)})
            kept_frames = picked
        for rel in kept_frames:
            src = os.path.join(session_dir, rel)
            if os.path.exists(src):
                dst = os.path.join(stage, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    if no_audio:
        redactions.append({"file": "audio*.wav", "kind": "excluded", "reason": "--no-audio"})
    else:
        for name, key in AUDIO_FILES:
            src = os.path.join(session_dir, name)
            if not os.path.exists(src):
                continue
            off = _audio_offset(manifest, key)
            if _clip_wav(src, os.path.join(stage, name), off, t_start, t_end):
                if t_start is not None or t_end is not None:
                    redactions.append({"file": name, "kind": "clipped", "offset": off})
            else:
                redactions.append({"file": name, "kind": "empty-after-clip"})

    files = []
    for root, _, names in os.walk(stage):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, stage).replace(os.sep, "/")
            files.append({"path": rel, "size": os.path.getsize(p), "sha256": _sha256(p)})
    files.sort(key=lambda f: f["path"])

    bm = {"session": os.path.basename(session_dir), "slice": label,
          "t_start": t_start, "t_end": t_end, "files": files,
          "total_bytes": sum(f["size"] for f in files), "redactions": redactions}
    with open(os.path.join(stage, "bundle-manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(bm, fh, indent=2)

    if as_zip:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, names in os.walk(stage):
                for n in names:
                    p = os.path.join(root, n)
                    z.write(p, os.path.relpath(p, stage))
        shutil.rmtree(stage)
        return out
    return stage


def _human(n):
    if n < 1024:
        return "%d B" % n
    for unit in ("KB", "MB", "GB"):
        n /= 1024.0
        if n < 1024 or unit == "GB":
            return "%.1f %s" % (n, unit)


def listing(bundle_dir):
    """Human-readable dry-run listing of a built bundle."""
    with open(os.path.join(bundle_dir, "bundle-manifest.json"), encoding="utf-8") as fh:
        bm = json.load(fh)
    lines = ["%10s  %s" % (_human(f["size"]), f["path"]) for f in bm["files"]]
    lines.append("%10s  total (%d files)" % (_human(bm["total_bytes"]), len(bm["files"])))
    return "\n".join(lines)


def add_slice_args(p):
    p.add_argument("session", help="session folder")
    p.add_argument("--from", dest="t_start", type=float, default=None, help="slice start (s)")
    p.add_argument("--to", dest="t_end", type=float, default=None, help="slice end (s)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-frames", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)


def main(argv=None):
    p = argparse.ArgumentParser(prog="understudy bundle")
    add_slice_args(p)
    p.add_argument("--out", default=None, help="output folder, or .zip file")
    p.add_argument("--dry-run", action="store_true", help="build in a temp folder, list, discard")
    a = p.parse_args(argv)
    out, tmp = a.out, None
    if a.dry_run:
        import tempfile
        tmp = tempfile.mkdtemp(prefix="understudy-bundle-")
        out = os.path.join(tmp, "b")
    path = build(a.session, a.t_start, a.t_end, a.no_audio, a.no_frames, a.max_frames, out)
    if os.path.isdir(path):
        print(listing(path))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("bundle: " + path)
    return 0
