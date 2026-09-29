"""participants.json: map speaker ids to display names, applied at read time.

Records in speakers.jsonl only ever carry ids, so renaming later is
retroactive. CLI: `understudy name <session> S2 "Maria"`, `... list`.
"""
import json
import os

FILE = "participants.json"
LOCAL = "local"


def _path(session_dir):
    return os.path.join(session_dir, FILE)


def load(session_dir):
    try:
        with open(_path(session_dir), encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(session_dir, mapping):
    tmp = _path(session_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _path(session_dir))


def name(session_dir, speaker_id, display_name):
    """Set (or, with an empty name, clear) a display name."""
    m = load(session_dir)
    if display_name:
        m[speaker_id] = display_name
    else:
        m.pop(speaker_id, None)
    save(session_dir, m)
    return m


def read_speakers(session_dir):
    """speakers.jsonl records with `name` filled in from participants.json."""
    names = load(session_dir)
    out = []
    try:
        with open(os.path.join(session_dir, "speakers.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                n = names.get(r.get("speaker_id"))
                if n:
                    r["name"] = n
                else:
                    r.pop("name", None)
                out.append(r)
    except OSError:
        pass
    return out


def summary(session_dir):
    """[{speaker_id, name, talk_s, sample_t}] sorted by id; sample_t is the
    start of the speaker's longest turn, a good place to listen."""
    names = load(session_dir)
    acc = {}
    for r in read_speakers(session_dir):
        sid = r.get("speaker_id")
        d = max(0.0, float(r.get("t1", 0)) - float(r.get("t0", 0)))
        a = acc.setdefault(sid, {"talk_s": 0.0, "best": -1.0, "sample_t": r.get("t0", 0.0)})
        a["talk_s"] += d
        if d > a["best"]:
            a["best"], a["sample_t"] = d, r.get("t0", 0.0)
    return [{"speaker_id": s, "name": names.get(s), "talk_s": round(a["talk_s"], 1),
             "sample_t": round(float(a["sample_t"]), 1)}
            for s, a in sorted(acc.items())]


def _resolve(session, root):
    if os.path.isdir(session):
        return session
    return os.path.join(os.path.expanduser(root), session)


def main(argv, root="~/Recordings"):
    import argparse
    p = argparse.ArgumentParser(prog="understudy name")
    p.add_argument("session", help="session directory or name")
    p.add_argument("args", nargs="*", help='<id> "<name>" | list')
    p.add_argument("--local-name", help="display name for the mic speaker")
    a = p.parse_args(argv)
    d = _resolve(a.session, root)
    if not os.path.isdir(d):
        print("no such session: %s" % d)
        return 1
    if a.local_name:
        name(d, LOCAL, a.local_name)
    if len(a.args) == 2:
        name(d, a.args[0], a.args[1])
        print("%s -> %s" % (a.args[0], a.args[1]))
        return 0
    if a.args in ([], ["list"]):
        rows = summary(d)
        if not rows:
            print("no speakers recorded")
        for r in rows:
            print("%-8s %-16s talk %6.1fs  sample at t=%.1fs"
                  % (r["speaker_id"], r["name"] or "-", r["talk_s"], r["sample_t"]))
        return 0
    p.error('usage: <session> <id> "<name>" | <session> list')
