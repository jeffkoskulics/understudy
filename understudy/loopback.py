"""System-audio (loopback) device discovery and capture backends.

Windows and Linux use the `soundcard` package (WASAPI loopback / PulseAudio or
PipeWire monitor source). macOS has no native loopback, so a virtual device
such as BlackHole is selected by name through sounddevice.

Everything here raises LoopbackUnavailable with a human-readable message when
capture isn't possible; audio.SystemAudioRecorder records that and carries on.
"""
import sys

VIRTUAL_NAMES = ("blackhole", "loopback audio", "soundflower", "vb-cable")

MAC_SETUP = (
    "No loopback device found. On macOS install BlackHole "
    "(brew install blackhole-2ch), open Audio MIDI Setup, create a "
    "Multi-Output Device containing your speakers plus BlackHole, set it as "
    "the system output, then restart Understudy.")


class LoopbackUnavailable(Exception):
    pass


def _platform():
    return sys.platform


def _is_virtual(name):
    n = (name or "").lower()
    return any(v in n for v in VIRTUAL_NAMES)


def _sc():
    try:
        import soundcard
        return soundcard
    except Exception as exc:
        raise LoopbackUnavailable(
            "soundcard package unavailable (pip install soundcard): %s" % exc)


def _sd():
    try:
        import sounddevice
        return sounddevice
    except Exception as exc:
        raise LoopbackUnavailable("sounddevice unavailable: %s" % exc)


def list_devices():
    """Input-capable devices, each marked with `loopback` (True if it can
    capture what the computer is playing). Never raises."""
    out = []
    if _platform() == "darwin":
        try:
            for i, d in enumerate(_sd().query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    out.append({"name": d["name"], "backend": "sounddevice",
                                "index": i, "loopback": _is_virtual(d["name"])})
        except Exception:
            pass
        return out
    try:
        for m in _sc().all_microphones(include_loopback=True):
            out.append({"name": m.name, "backend": "soundcard",
                        "index": None,
                        "loopback": bool(getattr(m, "isloopback", False))})
        return out
    except Exception:
        pass
    try:  # fall back to the plain input list
        for i, d in enumerate(_sd().query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append({"name": d["name"], "backend": "sounddevice",
                            "index": i,
                            "loopback": ".monitor" in d["name"].lower()})
    except Exception:
        pass
    return out


def open_loopback(device=None, samplerate=16000):
    """Return (reader, name). reader.read(frames) -> int16 mono numpy array
    (or None on timeout); reader.close() releases the device."""
    if _platform() == "darwin":
        return _open_mac(device, samplerate)
    return _open_soundcard(device, samplerate)


class _SCReader:
    def __init__(self, mic, samplerate):
        import numpy as np
        self.np = np
        self.ctx = mic.recorder(samplerate=samplerate, channels=1)
        self.rec = self.ctx.__enter__()

    def read(self, frames):
        data = self.rec.record(numframes=frames)
        return (self.np.clip(data.reshape(-1), -1, 1) * 32767).astype("int16")

    def close(self):
        self.ctx.__exit__(None, None, None)


def _open_soundcard(device, samplerate):
    sc = _sc()
    mics = sc.all_microphones(include_loopback=True)
    mic = None
    if device:
        mic = next((m for m in mics if device.lower() in m.name.lower()), None)
    else:
        try:
            spk = sc.default_speaker()
            mic = sc.get_microphone(spk.name, include_loopback=True)
        except Exception:
            mic = None
        if mic is None:
            mic = next((m for m in mics
                        if getattr(m, "isloopback", False)), None)
    if mic is None:
        raise LoopbackUnavailable(
            "No loopback/monitor source found (Linux: needs PulseAudio or "
            "PipeWire-pulse with a '.monitor' source; Windows: WASAPI).")
    return _SCReader(mic, samplerate), mic.name


class _SDReader:
    def __init__(self, sd, device, samplerate):
        self.stream = sd.InputStream(samplerate=samplerate, channels=1,
                                     dtype="int16", device=device)
        self.stream.start()

    def read(self, frames):
        data, _ = self.stream.read(frames)
        return data.reshape(-1)

    def close(self):
        self.stream.stop()
        self.stream.close()


def _open_mac(device, samplerate):
    sd = _sd()
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        if (device and device.lower() in d["name"].lower()) or \
                (not device and _is_virtual(d["name"])):
            return _SDReader(sd, i, samplerate), d["name"]
    raise LoopbackUnavailable(MAC_SETUP)
