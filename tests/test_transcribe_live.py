import array
import json
import os

from understudy.live.transcribe_live import LiveTranscriber, RATE
from understudy.live.backends.audio_remote import RemoteBackend


class FakeBackend:
    name, model = "fake", "fake"

    def __init__(self):
        self.calls = []

    def run(self, payload):
        self.calls.append(payload["t0"])
        dur = len(payload["pcm"]) / 2 / RATE
        t0 = payload["t0"]
        words = []  # one word per absolute second, "w<second>"
        k = int(t0) + (0 if t0 == int(t0) else 1)
        while k + 1 <= t0 + dur + 1e-6:
            words.append({"w": "w%d" % k, "t0": k - t0 + 0.1,
                          "t1": k - t0 + 0.9})
            k += 1
        return {"words": words}


class M:
    def __init__(self):
        self.g, self.c = {}, {}

    def timing(self, n, s): pass
    def gauge(self, n, v): self.g[n] = v
    def count(self, n, k=1): self.c[n] = self.c.get(n, 0) + k


def tone(sec, amp=3000):
    return array.array("h", [amp if i % 2 else -amp
                             for i in range(int(sec * RATE))]).tobytes()


def feed(lt, source, seconds, amp=3000):
    for s in range(seconds):
        lt.on_chunk(source, float(s), tone(1, amp))


def drain(lt):
    while not lt.inbox.empty():
        lt._process(lt.inbox.get())


def recs(d):
    with open(os.path.join(d, "live_transcript.jsonl")) as fh:
        return [json.loads(x) for x in fh]


def test_no_duplicates_and_final(tmp_path):
    lt = LiveTranscriber(None, str(tmp_path), FakeBackend(), metrics=M(),
                         speaker=lambda a, b, s: "Ann" if s == "mic" else None)
    feed(lt, "mic", 12)
    drain(lt)
    lt.stop()
    r = recs(str(tmp_path))
    finals = [w["w"] for x in r if x["final"] for w in x["words"]]
    assert finals == ["w%d" % i for i in range(12)]
    assert any(not x["final"] for x in r)
    assert all(x["speaker"] == "Ann" for x in r)


def test_silence_skipped_and_sources_separate(tmp_path):
    be = FakeBackend()
    lt = LiveTranscriber(None, str(tmp_path), be, metrics=M())
    feed(lt, "system", 6, amp=0)
    feed(lt, "mic", 6)
    drain(lt)
    assert len(be.calls) == 1
    assert {x["source"] for x in recs(str(tmp_path))} == {"mic"}


def test_rtf_and_drop_never_block(tmp_path):
    m = M()
    lt = LiveTranscriber(None, str(tmp_path), FakeBackend(), metrics=m,
                         inbox_size=1)
    feed(lt, "mic", 30)  # nothing consumes: windows drop
    assert lt.dropped > 0 and m.c["transcribe.dropped"] == lt.dropped
    lt._process(lt.inbox.get())
    assert "transcribe.rtf" in m.g


def test_remote_stub():
    try:
        RemoteBackend().run({})
        assert False
    except NotImplementedError:
        pass
