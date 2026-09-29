import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from understudy.live.backends.vision_local import (
    BackendError, OllamaBackend, OpenAICompatBackend, RemoteBackend)
from understudy.live.vision import VisionAnalyzer, encode_image


class Sess:
    def __init__(self, d):
        self.dir = str(d)


@pytest.fixture
def server():
    reqs = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send({"models": [{"name": "qwen2.5vl:7b"}]})
            else:
                self._send({"data": [{"id": "m1"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reqs.append((self.path, body))
            if self.path == "/api/generate":
                self._send({"response": " hello "})
            else:
                self._send({"choices": [{"message": {"content": "hi"}}]})

    s = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}", reqs
    s.shutdown()


def img(tmp_path, name="a.jpg", size=(2000, 1000)):
    p = tmp_path / name
    Image.new("RGB", size, "white").save(p)
    return str(p)


def test_ollama(server):
    url, reqs = server
    b = OllamaBackend(host=url)
    assert b.probe()
    assert b.run({"prompt": "p", "images": ["x"]}) == {"text": "hello"}
    assert reqs[0][1]["images"] == ["x"] and reqs[0][1]["model"] == "qwen2.5vl:7b"
    with pytest.raises(BackendError, match="ollama pull"):
        OllamaBackend(model="nope", host=url).probe()


def test_openai(server):
    url, reqs = server
    b = OpenAICompatBackend(model="m1", host=url)
    assert b.probe()
    assert b.run({"prompt": "p", "images": ["x"]})["text"] == "hi"
    assert "base64,x" in json.dumps(reqs[0][1])
    with pytest.raises(BackendError):
        OpenAICompatBackend(model="zz", host=url).probe()


def test_unreachable_and_remote():
    with pytest.raises(BackendError, match="ollama serve"):
        OllamaBackend(host="http://127.0.0.1:1").probe()
    with pytest.raises(NotImplementedError):
        RemoteBackend().run({})


def test_downscale(tmp_path):
    im = Image.open(io.BytesIO(base64.b64decode(encode_image(img(tmp_path), 500))))
    assert max(im.size) == 500


class Fake:
    name, model = "fake", "fm"

    def __init__(self):
        self.calls = []

    def run(self, p):
        self.calls.append(p)
        return {"text": "t%d" % len(self.calls)}

    def probe(self):
        return True


def lines(tmp_path):
    return [json.loads(l) for l in open(tmp_path / "vision.jsonl")]


def test_describe_diff(tmp_path):
    be = Fake()
    v = VisionAnalyzer(None, Sess(tmp_path), backend=be, mode="diff", max_rate=0)
    a, b = img(tmp_path, "a.jpg"), img(tmp_path, "b.jpg")
    v.step(({"t": 1.0, "file": "a.jpg"}, a))
    v.step(({"t": 2.0, "file": "b.jpg", "bbox": [1, 2, 3, 4]}, b))
    assert len(be.calls[0]["images"]) == 1 and len(be.calls[1]["images"]) == 2
    assert "[1, 2, 3, 4]" in be.calls[1]["prompt"]
    v.stop()
    r = lines(tmp_path)
    assert r[0]["mode"] == "describe" and r[1]["mode"] == "diff"
    assert r[1]["prev_frame"] == "a.jpg" and "latency_s" in r[1]


def test_max_rate_and_queue_skips(tmp_path):
    be = Fake()
    v = VisionAnalyzer(None, Sess(tmp_path), backend=be, max_rate=0.1, inbox_size=1)
    a = img(tmp_path)
    v.step(({"t": 1, "file": "a"}, a))
    v.step(({"t": 2, "file": "b"}, a))
    v.offer(({"t": 3, "file": "c"}, a))
    v.offer(({"t": 4, "file": "d"}, a))
    v.stop()
    reasons = [r.get("skipped") for r in lines(tmp_path)]
    assert reasons == [None, "max-rate", "max-rate", "shutdown"]
    assert v.dropped == 0 and len(be.calls) == 1


def test_queue_full_only_when_backlogged(tmp_path):
    v = VisionAnalyzer(None, Sess(tmp_path), backend=Fake(), max_rate=0, inbox_size=1)
    a = img(tmp_path)
    v.offer(({"t": 1, "file": "a"}, a))
    v.offer(({"t": 2, "file": "b"}, a))
    v.stop()
    assert [r.get("skipped") for r in lines(tmp_path)] == ["queue-full", "shutdown"]


def test_on_change_skips_unchanged(tmp_path):
    be = Fake()
    v = VisionAnalyzer(None, Sess(tmp_path), backend=be, max_rate=0, on_change=True)
    a = img(tmp_path)
    v.offer(({"t": 1, "file": "a", "reasons": []}, a))
    v.offer(({"t": 2, "file": "b", "reasons": ["diff"]}, a))
    v.step(v.inbox.get_nowait())
    v.stop()
    r = lines(tmp_path)
    assert r[0]["skipped"] == "unchanged" and r[1]["frame"] == "b" and "skipped" not in r[1]


def test_error_recorded(tmp_path):
    class Bad(Fake):
        def run(self, p):
            raise BackendError("down")

    v = VisionAnalyzer(None, Sess(tmp_path), backend=Bad(), max_rate=0)
    v.step(({"t": 1, "file": "a"}, img(tmp_path)))
    v.stop()
    assert "down" in lines(tmp_path)[0]["skipped"]
