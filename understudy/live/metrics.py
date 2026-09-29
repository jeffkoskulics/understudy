"""Instrumentation: 1 Hz sampler -> metrics.jsonl, machine profile -> system.json.

Implements the bus.py metrics interface (timing/gauge/count). See
docs/CONTRACTS.md for the record format.
"""
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:  # sampler degrades to counters/timings only
    psutil = None

PACKAGES = ["faster-whisper", "ctranslate2", "torch", "mss", "sounddevice",
            "numpy", "opencv-python", "Pillow", "psutil", "pynvml",
            "nvidia-ml-py", "ollama", "llama-cpp-python", "boto3",
            "onnxruntime", "sherpa-onnx", "resemblyzer", "soundcard"]


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


class _Timer:
    def __init__(self, metrics, name):
        self.metrics, self.name = metrics, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.metrics.timing(self.name, time.perf_counter() - self.t0)
        return False


class _Gpu:
    """Best-effort GPU sampling. sample() returns a dict or None."""

    def __init__(self):
        self.nvml = None
        self.smi = None
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            self.nvml = (pynvml, [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)])
            if n == 0:
                self.nvml = None
        except Exception:
            self.nvml = None
        if not self.nvml:
            self.smi = shutil.which("nvidia-smi")
        self.apple = sys.platform == "darwin" and platform.machine() == "arm64"

    def sample(self):
        try:
            if self.nvml:
                pynvml, handles = self.nvml
                out = []
                for h in handles:
                    u = pynvml.nvmlDeviceGetUtilizationRates(h)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    out.append({"util_pct": u.gpu, "mem_used_mb": m.used / 2**20,
                                "mem_total_mb": m.total / 2**20})
                return {"devices": out} if out else None
            if self.smi:
                r = subprocess.run(
                    [self.smi, "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                out = []
                for line in r.stdout.strip().splitlines():
                    a, b, c = [float(x) for x in line.split(",")]
                    out.append({"util_pct": a, "mem_used_mb": b, "mem_total_mb": c})
                return {"devices": out} if out else None
            if self.apple:
                # Best effort, no root: ioreg exposes the GPU's utilization.
                r = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                                   capture_output=True, text=True, timeout=2)
                import re
                m = re.search(r'"Device Utilization %"\s*=\s*(\d+)', r.stdout)
                if m:
                    return {"util_pct": float(m.group(1))}
        except Exception:
            pass
        return None


class Metrics:
    """Thread-safe metrics sink plus an optional 1 Hz sampler thread."""

    def __init__(self, session_dir, clock=None, interval=1.0):
        self.session_dir = str(session_dir)
        self.clock = clock
        self.interval = interval
        self._lock = threading.Lock()
        self._timings = {}
        self._gauges = {}
        self._counters = {}
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.monotonic()
        self._proc = psutil.Process() if psutil else None
        self._last_disk = None
        self._gpu = None
        self.path = os.path.join(self.session_dir, "metrics.jsonl")

    # --- bus.py interface -------------------------------------------------
    def timing(self, name, seconds):
        with self._lock:
            self._timings.setdefault(name, []).append(float(seconds))

    def gauge(self, name, value):
        with self._lock:
            self._gauges[name] = value

    def count(self, name, n=1):
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + n

    def timer(self, name):
        return _Timer(self, name)

    # --- sampler ----------------------------------------------------------
    def _now(self):
        return self.clock.t() if self.clock else time.monotonic() - self._t0

    def sample(self):
        """Take one sample and return the record (also resets timings)."""
        s0 = time.perf_counter()
        rec = {"t": round(self._now(), 3)}
        if psutil:
            try:
                rec["cpu_pct"] = self._proc.cpu_percent(None)
                rec["cpu_sys_pct"] = psutil.cpu_percent(None)
                rec["cpu_per_core"] = psutil.cpu_percent(None, percpu=True)
                rec["rss_mb"] = round(self._proc.memory_info().rss / 2**20, 1)
                rec["sys_mem_pct"] = psutil.virtual_memory().percent
                rec["disk_write_mb_s"] = self._disk_rate()
            except Exception:
                pass
        if self._gpu is None:
            self._gpu = _Gpu()
        g = self._gpu.sample()
        if g:
            rec["gpu"] = g
        with self._lock:
            rec["counters"] = dict(self._counters)
            rec["gauges"] = dict(self._gauges)
            timings, self._timings = self._timings, {}
        rec["timings"] = {}
        for k, v in timings.items():
            v.sort()
            rec["timings"][k] = {"n": len(v), "p50": _pct(v, .5),
                                 "p95": _pct(v, .95), "max": v[-1]}
        rec["sampler_overhead_s"] = round(time.perf_counter() - s0, 6)
        return rec

    def _disk_rate(self):
        try:
            b = psutil.disk_io_counters(perdisk=False).write_bytes
        except Exception:
            return 0.0
        now = time.monotonic()
        prev, self._last_disk = self._last_disk, (now, b)
        if not prev or now <= prev[0]:
            return 0.0
        return round(max(0, b - prev[1]) / 2**20 / (now - prev[0]), 3)

    def _write(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _run(self):
        if psutil:  # prime the cpu counters
            self._proc.cpu_percent(None)
            psutil.cpu_percent(None, percpu=True)
        while not self._stop.wait(self.interval):
            try:
                self._write(self.sample())
            except Exception:
                pass  # never raise into the recorder

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="metrics")
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
            try:
                self._write(self.sample())  # final flush
            except Exception:
                pass


# --- system.json ----------------------------------------------------------
def _cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _gpu_info():
    out = []
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            out.append({"name": name,
                        "vram_mb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20)})
        if out:
            return out
    except Exception:
        pass
    try:
        smi = shutil.which("nvidia-smi")
        if smi:
            r = subprocess.run([smi, "--query-gpu=name,memory.total",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            for line in r.stdout.strip().splitlines():
                n, m = line.rsplit(",", 1)
                out.append({"name": n.strip(), "vram_mb": int(float(m))})
        elif sys.platform == "darwin":
            r = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                               capture_output=True, text=True, timeout=5)
            for d in json.loads(r.stdout).get("SPDisplaysDataType", []):
                out.append({"name": d.get("sppci_model", "unknown"),
                            "vram_mb": None})
    except Exception:
        pass
    return out


def write_system_json(session_dir, extra=None):
    from importlib import metadata
    pkgs = {}
    for p in PACKAGES:
        try:
            pkgs[p] = metadata.version(p)
        except Exception:
            pass
    info = {
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cores_logical": os.cpu_count(),
        "cores_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_mb": round(psutil.virtual_memory().total / 2**20) if psutil else None,
        "gpus": _gpu_info(),
        "python": sys.version.split()[0],
        "packages": pkgs,
    }
    if extra:
        info.update(extra)
    path = os.path.join(str(session_dir), "system.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


# --- summary ----------------------------------------------------------------
def summarize(session_dir):
    """Return a dict summarising metrics.jsonl (peak/mean cpu, ram, latencies, drops)."""
    path = os.path.join(str(session_dir), "metrics.jsonl")
    recs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass

    def series(key):
        return [r[key] for r in recs if isinstance(r.get(key), (int, float))]

    def stat(vals):
        return {"mean": sum(vals) / len(vals), "peak": max(vals)} if vals else None

    stages = {}
    for r in recs:
        for k, v in r.get("timings", {}).items():
            s = stages.setdefault(k, {"n": 0, "p50": [], "p95": [], "max": 0.0})
            s["n"] += v["n"]
            s["p50"].append((v["p50"], v["n"]))
            s["p95"].append(v["p95"])
            s["max"] = max(s["max"], v["max"])
    latency = {}
    for k, s in stages.items():
        tot = sum(n for _, n in s["p50"]) or 1
        latency[k] = {"n": s["n"],
                      "p50": sum(p * n for p, n in s["p50"]) / tot,  # weighted approx
                      "p95": max(s["p95"]), "max": s["max"]}
    counters = recs[-1].get("counters", {}) if recs else {}
    gpu_util = []
    for r in recs:
        g = r.get("gpu") or {}
        devs = g.get("devices") or ([g] if "util_pct" in g else [])
        vals = [d["util_pct"] for d in devs if "util_pct" in d]
        if vals:
            gpu_util.append(max(vals))
    return {
        "samples": len(recs),
        "duration_s": recs[-1]["t"] - recs[0]["t"] if recs else 0,
        "cpu_pct": stat(series("cpu_pct")),
        "cpu_sys_pct": stat(series("cpu_sys_pct")),
        "rss_mb": stat(series("rss_mb")),
        "sys_mem_pct": stat(series("sys_mem_pct")),
        "disk_write_mb_s": stat(series("disk_write_mb_s")),
        "gpu_util_pct": stat(gpu_util),
        "latency": latency,
        "drops": {k: v for k, v in counters.items() if "dropped" in k},
        "errors": {k: v for k, v in counters.items() if "errors" in k},
    }


def _fmt(s):
    return "n/a" if not s else "mean %.1f  peak %.1f" % (s["mean"], s["peak"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: metrics <session_dir>")
        return 2
    s = summarize(argv[0])
    if not s["samples"]:
        print("no metrics.jsonl samples in", argv[0])
        return 1
    print("samples: %d over %.0f s" % (s["samples"], s["duration_s"]))
    print("process CPU %%:   %s" % _fmt(s["cpu_pct"]))
    print("system CPU %%:    %s" % _fmt(s["cpu_sys_pct"]))
    print("RSS MB:          %s" % _fmt(s["rss_mb"]))
    print("system mem %%:    %s" % _fmt(s["sys_mem_pct"]))
    print("disk write MB/s: %s" % _fmt(s["disk_write_mb_s"]))
    if s["gpu_util_pct"]:
        print("GPU util %%:      %s" % _fmt(s["gpu_util_pct"]))
    print("latency (s):")
    for k, v in sorted(s["latency"].items()):
        print("  %-28s n=%-6d p50=%.3f p95=%.3f max=%.3f"
              % (k, v["n"], v["p50"], v["p95"], v["max"]))
    print("drops:  ", s["drops"] or "none")
    print("errors: ", s["errors"] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
