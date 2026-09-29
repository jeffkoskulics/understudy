"""Find the NPU and integrated GPU on this machine and measure what they can do.

The first real session ran on an Intel laptop with no discrete GPU, where Ollama
fell back to the CPU and took 83 s to describe one frame. Ollama cannot use an
Intel NPU or iGPU at all; OpenVINO can. This script answers, in order:

  1. Does Windows see an NPU ("Intel AI Boost") and which GPU is there?
  2. Does OpenVINO see them (CPU / GPU / NPU), and with which driver?
  3. How fast is each device on a small synthetic network? (proves the device
     actually runs work, and gives a rough CPU:GPU:NPU ratio)
  4. Optionally: how long does a real vision-language model take to describe a
     frame on each device? This is the number that decides whether live frame
     analysis can stay on this laptop.

Run from the understudy folder:

    .venv\\Scripts\\python helpers\\npu_probe.py --install          # one-time: OpenVINO
    .venv\\Scripts\\python helpers\\npu_probe.py                    # steps 1-3
    .venv\\Scripts\\python helpers\\npu_probe.py --vlm OpenVINO/<model-id> --image <frame.jpg>

Results are printed and written to npu_probe.json, small enough to paste into a
chat or copy into a session folder before `understudy upload`.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

RESULTS = {"platform": platform.platform(), "python": sys.version.split()[0]}

# Pre-converted OpenVINO VLMs to try, smallest first. The Hugging Face ids change
# over time; if all fail, browse https://huggingface.co/OpenVINO and pass --vlm.
VLM_CANDIDATES = [
    "OpenVINO/Qwen2-VL-2B-Instruct-int4-ov",
    "OpenVINO/InternVL2-1B-int4-ov",
    "OpenVINO/Phi-3.5-vision-instruct-int4-ov",
]
PROMPT = ("Describe the application on screen: its name, the main UI elements, "
          "the visible text that matters, and what the user appears to be doing.")


def section(title):
    print("\n== %s ==" % title)


def install():
    pkgs = ["openvino", "openvino-genai", "huggingface_hub", "pillow", "numpy"]
    print("Installing: " + " ".join(pkgs))
    return subprocess.call([sys.executable, "-m", "pip", "install", "-U"] + pkgs)


def windows_devices():
    """What Windows itself reports, independent of OpenVINO."""
    if os.name != "nt":
        return {"note": "not Windows; skipped"}
    ps = (
        "$n = Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match "
        "'AI Boost|\bNPU\b|Neural Processing' -or $_.Class -eq 'ComputeAccelerator' } | "
        "Select-Object FriendlyName,Status,Class,InstanceId;"
        "$g = Get-CimInstance Win32_VideoController | "
        "Select-Object Name,DriverVersion,AdapterRAM;"
        "$c = Get-CimInstance Win32_Processor | Select-Object Name;"
        "@{npu=@($n); gpu=@($g); cpu=@($c)} | ConvertTo-Json -Depth 4"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
        return json.loads(out or "{}")
    except Exception as exc:
        return {"error": repr(exc)}


def ov_core():
    try:
        import openvino as ov
    except ImportError:
        print("OpenVINO is not installed. Run this script with --install first.")
        sys.exit(2)
    return ov, ov.Core()


def ov_devices(ov, core):
    info = {"openvino": ov.__version__ if hasattr(ov, "__version__") else ov.get_version()}
    for dev in core.available_devices:
        d = {}
        for prop in ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "GPU_DEVICE_TOTAL_MEM_SIZE",
                     "OPTIMIZATION_CAPABILITIES"):
            try:
                v = core.get_property(dev, prop)
                d[prop] = v if isinstance(v, (int, float, str)) else str(v)
            except Exception:
                pass
        info[dev] = d
    return info


def synthetic_model(ov):
    """A small conv + matmul stack: enough work to separate the devices."""
    try:
        import openvino.opset13 as ops
    except ImportError:
        from openvino.runtime import opset13 as ops
    import numpy as np
    rng = np.random.default_rng(0)
    x = ops.parameter([1, 3, 224, 224], np.float32, name="x")
    h = x
    for cin, cout in ((3, 32), (32, 64), (64, 64)):
        w = ops.constant(rng.standard_normal((cout, cin, 3, 3)).astype(np.float32) * 0.05)
        h = ops.relu(ops.convolution(h, w, [2, 2], [1, 1], [1, 1], [1, 1]))
    h = ops.reshape(h, ops.constant(np.array([1, -1], dtype=np.int64)), False)
    w = ops.constant(rng.standard_normal((64 * 28 * 28, 1024)).astype(np.float32) * 0.01)
    h = ops.relu(ops.matmul(h, w, False, False))
    for _ in range(4):
        w = ops.constant(rng.standard_normal((1024, 1024)).astype(np.float32) * 0.03)
        h = ops.relu(ops.matmul(h, w, False, False))
    return ov.Model([h], [x], "npu_probe")


def bench_synthetic(ov, core, runs=50):
    import numpy as np
    model = synthetic_model(ov)
    inp = np.random.default_rng(1).standard_normal((1, 3, 224, 224)).astype(np.float32)
    out = {}
    for dev in core.available_devices:
        r = {}
        try:
            t = time.perf_counter()
            cm = core.compile_model(model, dev)
            r["compile_s"] = round(time.perf_counter() - t, 3)
            req = cm.create_infer_request()
            req.infer({0: inp})                     # warm-up
            t = time.perf_counter()
            for _ in range(runs):
                req.infer({0: inp})
            r["infer_ms"] = round((time.perf_counter() - t) / runs * 1000, 2)
        except Exception as exc:
            r["error"] = repr(exc)[:300]
        out[dev] = r
        print("  %-6s %s" % (dev, r))
    return out


def fetch_vlm(model_id):
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    print("  downloading %s (first time only; can be a few GB) ..." % model_id)
    return snapshot_download(model_id)


def load_image(ov, path):
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((1280, 1280))
    return ov.Tensor(np.asarray(im, dtype=np.uint8)[None]), im.size


def bench_vlm(ov, core, model_ids, image_path, devices, max_new_tokens):
    import openvino_genai as og
    tensor, size = load_image(ov, image_path)
    results = {"image": image_path, "image_size": size, "prompt": PROMPT}
    path = None
    for mid in model_ids:
        try:
            path = fetch_vlm(mid)
            results["model"] = mid
            break
        except Exception as exc:
            print("  %s: not available (%s)" % (mid, repr(exc)[:120]))
    if not path:
        results["error"] = "no VLM could be downloaded; pass --vlm with a current id"
        return results
    cfg = og.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    for dev in devices:
        if dev not in core.available_devices:
            results[dev] = {"error": "device not present"}
            continue
        r = {}
        try:
            t = time.perf_counter()
            pipe = og.VLMPipeline(path, dev)
            r["load_s"] = round(time.perf_counter() - t, 1)
            pipe.generate("Hi", image=tensor, generation_config=cfg)   # warm-up
            t = time.perf_counter()
            res = pipe.generate(PROMPT, image=tensor, generation_config=cfg)
            r["describe_s"] = round(time.perf_counter() - t, 1)
            text = res.texts[0] if hasattr(res, "texts") else str(res)
            r["text"] = text.strip()
        except Exception as exc:
            r["error"] = repr(exc)[:400]
        results[dev] = r
        print("  %-6s %s" % (dev, {k: v for k, v in r.items() if k != "text"}))
        if r.get("text"):
            print("         " + r["text"][:300].replace("\n", " "))
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--install", action="store_true", help="pip install OpenVINO and helpers, then exit")
    p.add_argument("--vlm", nargs="?", const="auto", default=None,
                   help="also time a vision-language model; 'auto' tries known OpenVINO ids")
    p.add_argument("--image", help="frame to describe, e.g. a session's frames\\000000.jpg")
    p.add_argument("--devices", default="CPU,GPU,NPU")
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--out", default="npu_probe.json")
    a = p.parse_args(argv)
    if a.install:
        return install()

    section("Windows device list")
    RESULTS["windows"] = windows_devices()
    print(json.dumps(RESULTS["windows"], indent=2)[:2000])

    ov, core = ov_core()
    section("OpenVINO devices")
    RESULTS["openvino"] = ov_devices(ov, core)
    print(json.dumps(RESULTS["openvino"], indent=2))
    if "NPU" not in core.available_devices:
        print("\nNo NPU visible to OpenVINO. If Windows lists 'Intel AI Boost' above,\n"
              "install the latest Intel NPU driver (search 'Intel NPU driver Windows')\n"
              "and re-run. If Windows lists nothing, this CPU has no NPU.")

    section("Synthetic benchmark (lower infer_ms is better)")
    RESULTS["synthetic"] = bench_synthetic(ov, core)

    if a.vlm:
        section("Vision-language model: time to describe one frame")
        if not a.image or not os.path.exists(a.image):
            print("  --image is required for --vlm (use a frame from a session)")
        else:
            ids = VLM_CANDIDATES if a.vlm == "auto" else [a.vlm]
            RESULTS["vlm"] = bench_vlm(ov, core, ids, a.image,
                                       [d.strip().upper() for d in a.devices.split(",")],
                                       a.max_new_tokens)
            print("  (Ollama on CPU took 83 s for the same job in the first session)")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2, default=str)
    print("\nWrote " + os.path.abspath(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
