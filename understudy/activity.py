"""Classify what the user was doing, from the record alone -- no model.

Most of the answer is already in the session. `capture.py` writes the app, the
window title, why each frame was kept, how much of the screen changed and
where; `events.py` writes clicks, typing runs, scrolls and keys. Between them
that is enough to separate reading from typing from waiting, which is the
distinction the frames themselves are worst at: a screenshot of a spreadsheet
looks the same whether it is being read or filled in, and the event stream
says which.

So this stage runs first and for free, and the embedding stage in `embed.py`
refines it rather than replacing it.

The unit is a *segment*, not a frame. Hundreds of frames an hour is the wrong
granularity to label -- a user is not doing a different thing every 500 ms --
and the two boundaries that do mean something are already recorded: arriving
in a different window, and stopping for long enough that the previous activity
has clearly ended. Everything between two boundaries is aggregated and labelled
once.
"""
import json
import os

# Boundaries.
IDLE_GAP = 8.0        # no events for this long ends the current activity
MIN_SEGMENT = 0.75    # shorter than this is noise from adjacent boundaries
SLIVER = 3.0          # an event-free scrap this short belongs to what follows
KIND_RUN = 2          # events of a new kind before it counts as a new activity

# Event types fold to the kind of work they are: reading a page and reading a
# document are the same activity, and a scroll is not a click however the
# platform reports it.
KIND = {"click": "pointing", "key": "pointing", "typing": "typing",
        "scroll": "scrolling"}

# Rule thresholds. Deliberately blunt: these decide between labels that are
# already far apart, and a tuned threshold would only be tuned to one machine.
IDLE_MIN = 15.0       # doing nothing this long, with a still screen, is idle
STILL = 0.01          # mean change fraction below this is a static screen
BUSY = 0.05           # ...and above this the screen is doing something itself
TYPING_CHARS = 20     # a real typing run, not a keyboard shortcut


def _rows(session_dir, name):
    path = os.path.join(session_dir, name)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _session_end(frames, manifest):
    if manifest.get("duration"):
        return float(manifest["duration"])
    if frames:
        last = frames[-1]
        return last["t"] + last.get("duration", 0.0)
    return 0.0


def boundaries(frames, events, end):
    """Where one activity stops and the next begins. -> sorted list of times

    Two sources, and they answer different questions. A window switch says the
    user went somewhere else; an idle gap says they stopped. The gap
    contributes two boundaries, not one, so the pause becomes a segment of its
    own instead of being absorbed into whatever came before it -- a ten-minute
    break inside a "typing" segment would misreport both the activity and how
    long it took.
    """
    cuts = {0.0}
    for f in frames:
        if f.get("reason") == "window-switch":
            cuts.add(f["t"])

    # A third boundary the record gives for free: the kind of input changed.
    # Without it a stretch of reading that happens to follow a few clicks in
    # the same window is absorbed into them and reported as clicking, which
    # is the most common way this misreads a real session.
    #
    # Both sides have to be sustained. One stray click inside a long read is
    # not a boundary, and neither is the alternation of clicking and typing
    # that *is* form-filling -- splitting on every switch there would turn one
    # activity into a dozen one-event ones.
    runs = []
    for e in events:
        kind = KIND.get(e["type"])
        if kind is None:
            continue
        if runs and runs[-1][0] == kind:
            runs[-1][1].append(e)
        else:
            runs.append([kind, [e]])
    for before, after in zip(runs, runs[1:]):
        if len(before[1]) >= KIND_RUN and len(after[1]) >= KIND_RUN:
            cuts.add(after[1][0]["t"])

    # An idle gap ends whatever was happening IDLE_GAP after the last event,
    # and the next event starts something new. `prev` stays None until there
    # has been an event, because a recording that opens with a quiet stretch
    # has no earlier activity for that stretch to be the end of -- cutting it
    # anyway split every event-free session into a phantom prologue.
    prev = None
    for e in events:
        if prev is None:
            if e["t"] > IDLE_GAP:
                cuts.add(e["t"])
        elif e["t"] - prev > IDLE_GAP:
            cuts.add(prev + IDLE_GAP)
            cuts.add(e["t"])
        prev = max(prev or 0.0, e["t"] + e.get("duration", 0.0))
    if prev is not None and end - prev > IDLE_GAP:
        cuts.add(prev + IDLE_GAP)

    ordered = sorted(t for t in cuts if 0.0 <= t <= end)
    kept = []
    for t in ordered:
        if not kept or t - kept[-1] >= MIN_SEGMENT:
            kept.append(t)

    # A window switch lands a boundary a moment before the first click in the
    # new window, leaving a scrap of dead air between them. It is not an
    # activity of its own -- it is the start of the next one -- so the scrap
    # is given to the segment that follows rather than to the one before it,
    # which is a different window.
    out = []
    for i, t in enumerate(kept):
        stop = kept[i + 1] if i + 1 < len(kept) else end
        quiet = not any(t <= e["t"] < stop for e in events)
        if quiet and stop - t < SLIVER and i + 1 < len(kept):
            kept[i + 1] = t             # the next segment starts here instead
            continue
        out.append(t)
    return out


def features(frames, events, start, end):
    """Aggregate one segment's frames and events into numbers. -> dict"""
    fs = [f for f in frames if start <= f["t"] < end]
    es = [e for e in events if start <= e["t"] < end]
    duration = max(end - start, 0.0)

    changes = [f.get("change", 0.0) for f in fs]
    areas = []
    for f in fs:
        bbox, w, h = f.get("bbox"), f.get("w"), f.get("h")
        if bbox and w and h:
            areas.append(abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(w * h))

    typing = [e for e in es if e["type"] == "typing"]
    return {
        "duration": round(duration, 2),
        "frames": len(fs),
        "clicks": sum(1 for e in es if e["type"] == "click"),
        "typing_runs": len(typing),
        "chars": sum(e.get("chars", 0) for e in typing),
        "scrolls": sum(1 for e in es if e["type"] == "scroll"),
        "keys": sum(1 for e in es if e["type"] == "key"),
        "events": len(es),
        "events_per_min": round(len(es) / (duration / 60.0), 2) if duration else 0.0,
        "window_switches": sum(1 for f in fs if f.get("reason") == "window-switch"),
        "window_moves": sum(1 for f in fs if f.get("reason") == "window-move"),
        "mean_change": round(sum(changes) / len(changes), 5) if changes else 0.0,
        "max_change": round(max(changes), 5) if changes else 0.0,
        "mean_bbox_area": round(sum(areas) / len(areas), 4) if areas else 0.0,
    }


def classify(f):
    """Label a segment from its features. -> (label, why)

    Ordered most specific first, and every branch states which numbers put it
    there. When a label looks wrong on a real recording, the `why` says which
    rule to argue with -- which is the whole reason this stage is rules and
    not a model.
    """
    if not f["events"]:
        if f["mean_change"] >= BUSY and f["frames"] >= 3:
            return "watching", "screen changing on its own, no input"
        if f["duration"] >= IDLE_MIN and f["mean_change"] < STILL:
            return "idle", "no input for %.0fs, screen static" % f["duration"]
        return "waiting", "no input, screen mostly static"

    if f["window_moves"] and f["window_moves"] >= f["frames"] / 2.0:
        return "arranging-windows", "most frames are window moves"

    if f["chars"] >= TYPING_CHARS and f["clicks"] <= 1:
        return "typing", "%d characters, %d clicks" % (f["chars"], f["clicks"])

    if f["typing_runs"] >= 2 and f["clicks"] >= 2:
        return "form-filling", ("%d typing runs interleaved with %d clicks"
                                % (f["typing_runs"], f["clicks"]))

    if f["chars"]:
        return "editing", "%d characters alongside %d clicks" % (f["chars"], f["clicks"])

    if f["scrolls"] >= 2 and f["clicks"] <= 1:
        return "reading", "%d scrolls, no typing" % f["scrolls"]

    if f["window_switches"] >= 2 and f["clicks"] <= 2:
        return "app-switching", "%d window switches" % f["window_switches"]

    if f["clicks"]:
        return "navigating", "%d clicks, no typing" % f["clicks"]

    return "other", "input present but no rule matched"


def _where(frames, start, end):
    """The app and window a segment happened in, by how long each was held."""
    held = {}
    for f in frames:
        if not (start <= f["t"] < end):
            continue
        key = (f.get("app") or "", f.get("window") or "")
        held[key] = held.get(key, 0.0) + max(f.get("duration", 0.0), 0.01)
    if not held:
        return "", "", {}
    app, window = max(held.items(), key=lambda kv: kv[1])[0]
    apps = {}
    for (a, _), secs in held.items():
        apps[a] = round(apps.get(a, 0.0) + secs, 2)
    return app, window, apps


def segments(session_dir):
    """Split a session into labelled activity segments. -> list of dicts"""
    frames = _rows(session_dir, "frames.jsonl")
    events = _rows(session_dir, "events.jsonl")
    manifest = {}
    mpath = os.path.join(session_dir, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
    if not frames:
        raise RuntimeError("no frames.jsonl in %s -- is that a session "
                           "directory?" % session_dir)

    end = _session_end(frames, manifest)
    cuts = boundaries(frames, events, end)
    out = []
    for i, start in enumerate(cuts):
        stop = cuts[i + 1] if i + 1 < len(cuts) else end
        if stop - start < MIN_SEGMENT:
            continue
        f = features(frames, events, start, stop)
        label, why = classify(f)
        app, window, apps = _where(frames, start, stop)
        out.append({
            "id": "A%03d" % (len(out) + 1),
            "start": round(start, 2), "end": round(stop, 2),
            "duration": f["duration"],
            "app": app, "window": window, "apps": apps,
            "label": label, "why": why,
            "frames": [fr["idx"] for fr in frames if start <= fr["t"] < stop],
            "features": f,
        })
    return out


def label_at(segs, t):
    """The segment covering time `t`, or None. Used to tag steps in `pack`."""
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s
    return segs[-1] if segs and t >= segs[-1]["end"] else None


def load(session_dir):
    """Read activity.jsonl if it has been written, else []."""
    return _rows(session_dir, "activity.jsonl")


def write(session_dir, segs):
    """Write activity.jsonl. -> path

    Separate from `analyse` because `embed` adds names to the same records
    and rewrites them in place, rather than keeping a second file that can
    drift out of step with this one.
    """
    path = os.path.join(session_dir, "activity.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for s in segs:
            fh.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
    return path


def analyse(session_dir):
    """Classify and write activity.jsonl. -> (segments, path)"""
    segs = segments(session_dir)
    return segs, write(session_dir, segs)


def _mmss(t):
    return "%d:%02d" % (int(t) // 60, int(t) % 60)


def to_text(segs, width=22):
    lines = []
    for s in segs:
        where = ("%s - %s" % (s["app"], s["window"])).strip(" -")
        lines.append("%-5s %s-%s  %-*s %s"
                     % (s["id"], _mmss(s["start"]), _mmss(s["end"]),
                        width, s.get("name") or s["label"], where[:60]))
    return "\n".join(lines)


def summary(segs):
    """Seconds spent per label, longest first. -> list of (label, seconds)"""
    total = {}
    for s in segs:
        key = s.get("name") or s["label"]
        total[key] = total.get(key, 0.0) + s["duration"]
    return sorted(((k, round(v, 1)) for k, v in total.items()),
                  key=lambda kv: -kv[1])


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(
        prog="understudy activity",
        description="Classify user activity across a recorded session.")
    ap.add_argument("session", nargs="?")
    ap.add_argument("--embed", action="store_true",
                    help="also label segments with a local text embedding "
                         "model (see understudy activity --install)")
    ap.add_argument("--labels", default=None,
                    help="comma-separated labels to match against, with "
                         "--embed; omit to discover them by clustering")
    ap.add_argument("--clusters", type=int, default=0,
                    help="cluster into this many activities with --embed "
                         "(default: chosen from the session length)")
    ap.add_argument("--install", action="store_true",
                    help="install the embedding model support and exit")
    args = ap.parse_args(argv)

    if args.install:
        from . import embed
        try:
            if embed.is_installed():
                print("The embedding support is already installed.")
            else:
                embed.install(progress=lambda m: print(m))
                print("Installed. Re-run with --embed.")
        except embed.EmbedError as exc:
            print(str(exc))
            return 1
        if not args.session:
            return 0

    if not args.session:
        ap.error("a session directory is required")

    try:
        segs, path = analyse(args.session)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    if not segs:
        print("No activity found in this session.")
        return 0

    if args.embed:
        from . import embed
        try:
            labels = [l.strip() for l in args.labels.split(",")] if args.labels else None
            embed.label(args.session, segs, labels=labels,
                        clusters=args.clusters or None,
                        progress=lambda m: print(m))
        except embed.EmbedError as exc:
            print(str(exc))
            print("\nThe rule-based labels below were still written.")

    print(to_text(segs))
    print()
    for label, secs in summary(segs):
        print("%6.0fs  %s" % (secs, label))
    print("\n%s" % path)
    return 0
