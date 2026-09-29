import base64
import io
import sys
import types

import pytest

pytest.importorskip("openvino")
from PIL import Image

from understudy.live.backends import vision_openvino


def jpeg_b64():
    buf = io.BytesIO()
    Image.new("RGB", (32, 16), "red").save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_run_decodes_images_and_passes_prompt(monkeypatch, tmp_path):
    calls = {}

    class Pipe:
        def __init__(self, path, device):
            calls["init"] = (path, device)

        def generate(self, prompt, generation_config=None, **kw):
            calls["prompt"], calls["kw"] = prompt, kw
            return types.SimpleNamespace(texts=["  a red screen  "])

    class Cfg:
        max_new_tokens = 0

    fake = types.SimpleNamespace(VLMPipeline=Pipe, GenerationConfig=Cfg)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake)
    be = vision_openvino.OpenVINOBackend(model=str(tmp_path), device="gpu")
    out = be.run({"prompt": "describe", "images": [jpeg_b64()]})
    assert out == {"text": "a red screen"}
    assert calls["init"] == (str(tmp_path), "GPU")
    assert tuple(calls["kw"]["image"].shape) == (1, 16, 32, 3)
    be.run({"prompt": "diff", "images": [jpeg_b64(), jpeg_b64()]})
    assert len(calls["kw"]["images"]) == 2
