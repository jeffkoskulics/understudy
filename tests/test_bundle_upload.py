import json
import os
import wave

import pytest

from understudy import bundle, upload


def make_session(tmp_path):
    d = tmp_path / "s1"
    (d / "frames").mkdir(parents=True)
    with open(d / "frames.jsonl", "w") as fh:
        for i in range(10):
            (d / "frames" / ("%06d.jpg" % i)).write_bytes(b"x" * 10)
            fh.write(json.dumps({"idx": i, "t": float(i), "file": "frames/%06d.jpg" % i}) + "\n")
    with open(d / "live_transcript.jsonl", "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"t0": float(i), "t1": i + 1.0, "text": "w%d" % i}) + "\n")
    with wave.open(str(d / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(100)
        w.writeframes(b"\x00\x00" * 1000)  # 10 s
    (d / "manifest.json").write_text(json.dumps({"audio": {"start_offset": 2.0}}))
    return str(d)


def test_slice_and_audio(tmp_path):
    s = make_session(tmp_path)
    out = bundle.build(s, 3, 6, out=str(tmp_path / "b"))
    recs = [json.loads(l) for l in open(os.path.join(out, "frames.jsonl"))]
    assert [r["idx"] for r in recs] == [3, 4, 5, 6]
    assert len(os.listdir(os.path.join(out, "frames"))) == 4
    with wave.open(os.path.join(out, "audio.wav")) as w:
        assert w.getnframes() == 300
    bm = json.load(open(os.path.join(out, "bundle-manifest.json")))
    assert bm["slice"] == "t3-6" and any(f["path"] == "audio.wav" for f in bm["files"])
    assert "audio.wav" in bundle.listing(out)


def test_flags_and_zip(tmp_path):
    s = make_session(tmp_path)
    out = bundle.build(s, no_audio=True, no_frames=True, out=str(tmp_path / "b.zip"))
    assert out.endswith(".zip") and os.path.exists(out)
    out = bundle.build(s, max_frames=3, out=str(tmp_path / "c"))
    assert len(os.listdir(os.path.join(out, "frames"))) == 3


class Stub:
    def __init__(self):
        self.objs = {}
        self.order = []

    def head_object(self, Bucket, Key):
        if Key not in self.objs:
            raise KeyError(Key)
        return {"ContentLength": self.objs[Key]}

    def upload_file(self, path, Bucket, Key, Config=None):
        self.objs[Key] = os.path.getsize(path)
        self.order.append(Key)

    def put_object(self, Bucket, Key, Body, **kw):
        self.objs[Key] = len(Body)
        self.order.append(Key)

    def delete_object(self, Bucket, Key):
        self.objs.pop(Key, None)


def test_upload_skip_and_index_last(tmp_path):
    out = bundle.build(make_session(tmp_path), out=str(tmp_path / "b"))
    c = Stub()
    cfg = {"bucket": "b"}
    r = upload.upload(out, client=c, cfg=cfg, hostname="h")
    assert r["prefix"] == "bundles/h/s1/full/"
    assert c.order[-1] == "bundles/h/s1/full/index.json"
    r2 = upload.upload(out, client=c, cfg=cfg, hostname="h")
    assert "frames.jsonl" in r2["skipped"] and "frames.jsonl" not in r2["uploaded"]
    r3 = upload.upload(out, dry_run=True, hostname="h")
    assert r3["uploaded"]


def test_config_roundtrip_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for ev in upload.ENV.values():
        monkeypatch.delenv(ev, raising=False)
    with pytest.raises(RuntimeError):
        upload.load_config()
    answers = iter(["acct", "bkt", "akid"])
    c = Stub()
    cfg = upload.configure(lambda _: next(answers), lambda _: 's"ecret', lambda cfg: c)
    if os.name != "nt":
        assert os.stat(upload.config_path()).st_mode & 0o777 == 0o600
    assert upload.load_config() == cfg
    monkeypatch.setenv("UNDERSTUDY_R2_BUCKET", "other")
    assert upload.load_config()["bucket"] == "other"
    assert not c.objs
