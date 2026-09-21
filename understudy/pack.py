"""Turn a raw session into something that fits through a chat box.

A 30-minute recording holds roughly 200,000 tokens of OCR text. No one is
pasting that. Measured over real sessions, the reductions that work are, in
order of how much they buy:

  step segmentation   one entry per user action instead of one per frame (~7x)
  bbox cropping       only the text inside the region that changed (~1.3x)
  window-keyed delta  compare a window against the last time *that* window was
                      in front, not against whatever window preceded it (~1.2x)

Together those took a measured 203k tokens to about 26k.

Note the ordering: steps are chosen *before* OCR runs, so only the frames a
step actually cites get recognised. OCR costs ~3.3 s/frame and does not
parallelise well, so this is a bigger saving than throwing cores at it -- on
the reference session, 18 frames instead of 52.
"""
import json
import os
import re

from . import activity as activity_mod
from . import ocr as ocr_mod

CONF_MIN = 0.5        # Vision confidence below this is mostly noise
MAX_LABEL = 60        # a control label is short; a paragraph is not a control
LABEL_RIGHT = 200     # px to look right of a click, for checkbox-style labels
NEW_LINES_PER_STEP = 10
POST_CLICK = 0.4      # a click's consequence shows up about this much later


def _key(text):
    """Fold OCR jitter so the same line does not read as new every frame."""
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower().strip()))


def _load(session_dir):
    def rows(name):
        path = os.path.join(session_dir, name)
        if not os.path.exists(path):
            return []
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    manifest = {}
    mpath = os.path.join(session_dir, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
    return rows("frames.jsonl"), rows("events.jsonl"), manifest


def _frame_at(frames, t):
    """The first frame at or after t -- what the screen looked like by then."""
    for f in frames:
        if f["t"] >= t - 0.05:
            return f
    return frames[-1] if frames else None


def segment(frames, events):
    """Choose the steps. One per user action, plus one per window switch.

    A window switch with no click behind it still matters: in an application
    whose features live in separate windows, arriving somewhere new is itself
    a step, and it is the cheapest boundary available because the title bar
    names it without any OCR.
    """
    steps = []
    for e in events:
        delay = POST_CLICK if e["type"] == "click" else 0.0
        frame = _frame_at(frames, e["t"] + delay)
        if frame:
            steps.append({"t": e["t"], "event": e, "frame": frame})

    claimed = {s["frame"]["idx"] for s in steps}
    for f in frames:
        if f["reason"] == "window-switch" and f["idx"] not in claimed:
            steps.append({"t": f["t"], "event": None, "frame": f})

    steps.sort(key=lambda s: s["t"])
    for i, s in enumerate(steps, 1):
        s["id"] = "S%03d" % i
    return steps


def _lines(record, bbox=None):
    out = []
    for line in record.get("lines", []):
        if line["conf"] < CONF_MIN or not _key(line["text"]):
            continue
        if bbox:
            bx, by, bw, bh = line["box"]
            cx, cy = bx + bw / 2, by + bh / 2
            if not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
                continue
        out.append(line)
    return out


def label_at(record, x, y):
    """The text of the control at (x, y), or None.

    Prefers a line whose box contains the point, then one immediately to its
    right -- the usual geometry for a checkbox or radio button whose label sits
    beside it rather than on it. Anything long is rejected: matching a
    paragraph of body text would produce a confident, wrong answer, which is
    worse than admitting the click had no label.
    """
    hits = []
    for line in _lines(record):
        bx, by, bw, bh = line["box"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            hits.append((0, len(line["text"]), line["text"]))
        elif by <= y <= by + bh and 0 < bx - x < LABEL_RIGHT:
            hits.append((1, len(line["text"]), line["text"]))
    if not hits:
        return None
    hits.sort()                       # containing first, then shortest
    label = hits[0][2]
    return label if len(label) <= MAX_LABEL else None


def describe(step, record):
    e = step["event"]
    if e is None:
        return "switch to this window"
    if e["type"] == "click":
        label = label_at(record, e["x"], e["y"]) if record else None
        button = "" if e.get("button") == "left" else e.get("button", "") + "-"
        if label:
            return '%sclick "%s"' % (button, label)
        return "%sclick at (%d, %d)" % (button, e["x"], e["y"])
    if e["type"] == "typing":
        if "text" in e:
            return 'type "%s"' % e["text"]
        return "type %d characters over %.1fs" % (e["chars"], e.get("duration", 0))
    if e["type"] == "scroll":
        return "scroll %s" % ("down" if e.get("dy", 0) < 0 else "up")
    if e["type"] == "key":
        return "press %s" % e["key"]
    return e["type"]


def _narration(session_dir):
    path = os.path.join(session_dir, "transcript.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8"))


def _attach_narration(steps, narration):
    """Give each step the speech that overlaps it.

    A step owns the time from when it happened until the next step begins, so
    a sentence spoken while the user was still looking at a dialog is filed
    with that dialog rather than with whatever they did next.
    """
    if not narration:
        return
    bounds = [(s, steps[i + 1]["t"] if i + 1 < len(steps) else float("inf"))
              for i, s in enumerate(steps)]
    for step, end in bounds:
        step["said"] = [n["text"] for n in narration
                        if n["start"] < end and n["end"] > step["t"]]


def build(session_dir, workers=None):
    frames, events, manifest = _load(session_dir)
    if not frames:
        raise RuntimeError("no frames.jsonl in %s" % session_dir)
    steps = segment(frames, events)

    # OCR only the frames the steps cite, not every frame in the session.
    cited = []
    seen = set()
    for s in steps:
        idx = s["frame"]["idx"]
        if idx not in seen:
            seen.add(idx)
            cited.append(s["frame"])
    records = ocr_mod.ocr_frames(
        [os.path.join(session_dir, f["file"]) for f in cited], workers)
    by_idx = {f["idx"]: r for f, r in zip(cited, records)}

    per_window = {}
    for s in steps:
        frame = s["frame"]
        record = by_idx.get(frame["idx"], {})
        s["ocr"] = record
        visible = {_key(l["text"]): l["text"]
                   for l in _lines(record, frame.get("bbox"))}
        base = per_window.get(frame["window_id"], {})
        s["new"] = [v for k, v in visible.items() if k not in base]
        # Accumulate rather than replace: cropping means each frame only ever
        # sees part of its window, so replacing would make unchanged text
        # outside the crop look new every time it came back into view.
        per_window[frame["window_id"]] = {**base, **visible}
        s["action"] = describe(s, record)
    _attach_narration(steps, _narration(session_dir))

    # Activity labels are optional: `understudy activity` may not have been
    # run, and a step keeps its own description either way.
    segs = activity_mod.load(session_dir)
    for s in steps:
        seg = activity_mod.label_at(segs, s["t"]) if segs else None
        if seg:
            # The rule label, not the embedding name: the name groups a whole
            # stretch and is already on the timeline above, while what a
            # single step wants is the kind of input it was.
            s["activity"] = seg["label"]
            s["activity_name"] = seg.get("name")
    return steps, manifest, segs


def _activity_table(segs):
    """A timeline of what the user was doing, ahead of the step detail."""
    out = ["## Activity", "",
           "One line per stretch of work, before the step-by-step below."]
    for s in segs:
        where = ("%s - %s" % (s.get("app") or "", s.get("window") or "")).strip(" -")
        name = s.get("name")
        label = "%s (%s)" % (name, s["label"]) if name else s["label"]
        out.append("- [%s-%s] **%s** %s"
                   % (_mmss(s["start"]), _mmss(s["end"]), label, where))
    out.append("")
    return out


def to_markdown(steps, manifest, session_name="", segs=None):
    out = []
    dur = manifest.get("duration", 0)
    out.append("# Workflow recording%s" % (": " + session_name if session_name else ""))
    out.append("")
    out.append("%d steps over %d:%02d. Each step is one user action; `new:` lists "
               "text that appeared on screen as a result." % (len(steps), dur // 60, dur % 60))
    out.append("")
    if segs:
        out.extend(_activity_table(segs))
    last_window = None
    for s in steps:
        f = s["frame"]
        window = "%s - %s" % (f.get("app") or "?", f.get("window") or "")
        if window != last_window:
            out.append("")
            out.append("## %s" % window.rstrip(" -"))
            last_window = window
        out.append("")
        out.append("**%s** [%s] %s%s"
                   % (s["id"], _mmss(s["t"]), s["action"],
                      "  _(%s)_" % s["activity"] if s.get("activity") else ""))
        for said in s.get("said", []):
            out.append("  - said: %s" % said)
        if s["new"]:
            shown = s["new"][:NEW_LINES_PER_STEP]
            out.append("  - new: %s" % " | ".join(shown))
            if len(s["new"]) > len(shown):
                out.append("  - (%d more lines)" % (len(s["new"]) - len(shown)))
    return "\n".join(out) + "\n"


def _mmss(t):
    return "%d:%02d" % (int(t) // 60, int(t) % 60)


def pack(session_dir, workers=None):
    steps, manifest, segs = build(session_dir, workers)
    name = os.path.basename(os.path.normpath(session_dir))
    md = to_markdown(steps, manifest, name, segs)
    md_path = os.path.join(session_dir, "workflow.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)
    # A machine-readable sidecar with the same step ids, so an answer that
    # cites S007 can be merged back against the frame it came from.
    json_path = os.path.join(session_dir, "steps.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([{"id": s["id"], "t": s["t"], "action": s["action"],
                    "frame": s["frame"]["file"], "app": s["frame"].get("app"),
                    "window": s["frame"].get("window"), "new": s["new"],
                    "said": s.get("said", []), "activity": s.get("activity"),
                    "activity_name": s.get("activity_name")}
                   for s in steps], fh, indent=1, ensure_ascii=False)
    return md_path, json_path, len(md)


if __name__ == "__main__":
    import sys
    md, js, n = pack(sys.argv[1])
    print("%s  (%d chars, ~%d tokens)" % (md, n, n // 4))
    print(js)
