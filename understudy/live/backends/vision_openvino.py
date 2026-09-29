"""In-process vision backend on OpenVINO GenAI, for Intel iGPUs and NPUs.

Ollama only uses NVIDIA/AMD/Apple GPUs, so on an Intel laptop without one it
runs on the CPU: 83 s per frame with gemma3:4b on a Core Ultra 7 255U. The same
laptop's iGPU described a frame in 9.4 s with InternVL2-1B through OpenVINO
(CPU 17.1 s; the NPU failed to compile the model), measured with
helpers/npu_probe.py. This backend runs that path inside understudy.

Same contract as the HTTP backends: `run({"prompt", "images": [base64 jpeg]})`
returns {"text"}. `model` is a Hugging Face id of a pre-converted OpenVINO model
(downloaded once, then cached) or a local folder.
"""
import base64
import io
import os

from .vision_local import BackendError

DEFAULT_MODEL = "OpenVINO/InternVL2-1B-int4-ov"


class OpenVINOBackend:
    name = "openvino"

    def __init__(self, model=DEFAULT_MODEL, device="GPU", max_new_tokens=200, timeout=None):
        self.model, self.device, self.max_new_tokens = model, device.upper(), max_new_tokens
        self._pipe = None

    def probe(self):
        try:
            import openvino as ov
            import openvino_genai  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                "OpenVINO is not installed. Run: .venv\\Scripts\\python "
                "helpers\\npu_probe.py --install") from exc
        devices = ov.Core().available_devices
        if self.device not in devices:
            raise BackendError("OpenVINO device %s not found (have: %s)"
                               % (self.device, ", ".join(devices)))

    def _path(self):
        if os.path.isdir(self.model):
            return self.model
        from huggingface_hub import snapshot_download
        return snapshot_download(self.model)

    def warm_up(self):
        """Download (first time) and compile the model before recording starts."""
        import openvino_genai as og
        try:
            self._pipe = og.VLMPipeline(self._path(), self.device)
        except Exception as exc:
            raise BackendError("could not load %s on %s: %s"
                               % (self.model, self.device, exc)) from exc

    def run(self, payload):
        import numpy as np
        import openvino as ov
        import openvino_genai as og
        from PIL import Image
        if self._pipe is None:
            self.warm_up()
        tensors = [ov.Tensor(np.asarray(Image.open(io.BytesIO(base64.b64decode(b)))
                                         .convert("RGB"), dtype=np.uint8)[None])
                   for b in payload["images"]]
        cfg = og.GenerationConfig()
        cfg.max_new_tokens = self.max_new_tokens
        kw = {"images": tensors} if len(tensors) > 1 else {"image": tensors[0]}
        res = self._pipe.generate(payload["prompt"], generation_config=cfg, **kw)
        text = res.texts[0] if hasattr(res, "texts") else str(res)
        return {"text": text.strip()}
