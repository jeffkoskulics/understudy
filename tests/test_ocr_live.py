import json
import os
import sys
import textwrap

from understudy.live.ocr_live import LiveOCR

FAKE = textwrap.dedent('''
    import json, sys
    SCREENS = {"a": ["File", "Load Port 1", "x"], "b": ["File", "Alarm History", "Load  Port 1"]}
    for line in sys.stdin:
        p = line.strip()
        name = p.rsplit("/", 1)[-1].rsplit("\\\\", 1)[-1]
        if name == "bad":
            print(json.dumps({"file": p, "error": "unreadable"}), flush=True)
            continue
        print(json.dumps({"file": p, "lines": [{"text": t} for t in SCREENS[name]]}), flush=True)
''')


class Sess:
    def __init__(self, d):
        self.dir = str(d)
        self._w = {}

    def writer(self, name):
        from understudy.session import JsonlWriter
        return self._w.setdefault(name, JsonlWriter(os.path.join(self.dir, name)))


def test_sets_added_removed_and_union(tmp_path):
    helper = tmp_path / "fake_ocr.py"
    helper.write_text(FAKE)
    o = LiveOCR(None, Sess(tmp_path), command=[sys.executable, str(helper)])
    o.probe()
    o.step(({"t": 1.0, "file": "frames/a"}, "a"))
    o.step(({"t": 2.0, "file": "frames/b"}, "b"))
    o.step(({"t": 3.0, "file": "frames/bad"}, "bad"))
    o.finish()
    recs = [json.loads(l) for l in open(tmp_path / "live_ocr.jsonl")]
    assert recs[0]["text"] == ["File", "Load Port 1"]          # 'x' dropped, sorted
    assert recs[1]["added"] == ["Alarm History"] and recs[1]["removed"] == []
    assert recs[2]["error"] == "unreadable"
    union = json.load(open(tmp_path / "ocr_text.json"))
    assert union["count"] == 3
    assert union["text"]["File"] == {"first": 1.0, "last": 2.0, "frames": 2}
    assert union["text"]["Alarm History"]["first"] == 2.0
