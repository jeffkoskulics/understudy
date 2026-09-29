"""Continuous microphone capture.

One unbroken WAV for the whole session, 16 kHz mono -- the rate every speech
recogniser resamples to anyway, so nothing is gained by recording higher and a
lot of disk is spent. The point of a single continuous file is alignment: the
offset of the first sample against the session clock is recorded once, and from
then on any word in the transcript maps to a frame by simple arithmetic.

Audio failing (no device, permission refused) must never take the screen
recording down with it, so every failure here is recorded and swallowed.
"""
import queue
import threading
import wave

SAMPLE_RATE = 16000
CHANNELS = 1


def _tap(cb, source, t_end, data, frames=None):
    """Live tap. t0 is the session time of the chunk's first sample. Consumer
    errors are swallowed: they must never stop the recording."""
    if cb is None:
        return
    try:
        n = frames if frames is not None else len(data) // 2
        cb(source, max(0.0, t_end - n / SAMPLE_RATE), data)
    except Exception:
        pass


class SystemAudioRecorder(threading.Thread):
    """Records what the computer is playing (the other meeting participants)
    to audio_system.wav, 16 kHz mono. See loopback.py for the backends."""

    BLOCK = SAMPLE_RATE // 5   # 200 ms

    def __init__(self, clock, path, device=None, on_chunk=None):
        super().__init__(daemon=True)
        self.clock = clock
        self.path = path
        self.device = device
        self.on_chunk = on_chunk   # on_chunk("system", t0, pcm_int16_bytes)
        self.start_offset = None   # session time of the first sample
        self.frames_written = 0
        self.device_name = None
        self.error = None
        self._stopping = threading.Event()

    def run(self):
        try:
            from . import loopback
            reader, self.device_name = loopback.open_loopback(
                self.device, SAMPLE_RATE)
        except Exception as exc:
            self.error = str(exc)
            return
        try:
            wf = wave.open(self.path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            try:
                while not self._stopping.is_set():
                    arr = reader.read(self.BLOCK)
                    if arr is None or len(arr) == 0:
                        continue
                    now = self.clock.t()
                    data = arr.tobytes()
                    if self.start_offset is None:
                        self.start_offset = max(0.0, now - len(arr) / SAMPLE_RATE)
                    wf.writeframes(data)
                    self.frames_written += len(arr)
                    _tap(self.on_chunk, "system", now, data, len(arr))
            finally:
                wf.close()
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                reader.close()
            except Exception:
                pass

    def stop(self):
        self._stopping.set()

    def info(self):
        return {
            "path": "audio_system.wav",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "device": self.device_name,
            "start_offset": (round(self.start_offset, 3)
                             if self.start_offset is not None else None),
            "duration": round(self.frames_written / SAMPLE_RATE, 3),
            "error": self.error,
        }


def list_devices():
    """Input devices with a `loopback` flag; see loopback.list_devices."""
    from . import loopback
    return loopback.list_devices()


class AudioRecorder(threading.Thread):
    def __init__(self, clock, path, device=None, on_chunk=None):
        super().__init__(daemon=True)
        self.clock = clock
        self.path = path
        self.device = device
        self.on_chunk = on_chunk   # on_chunk("mic", t0, pcm_int16_bytes)
        self.start_offset = None   # session time of the first sample
        self.frames_written = 0
        self.error = None
        self._q = queue.Queue()
        self._stopping = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if self.start_offset is None:
            self.start_offset = self.clock.t()
        data = bytes(indata)
        self._q.put(data)
        _tap(self.on_chunk, "mic", self.clock.t(), data, frames)

    def run(self):
        try:
            import sounddevice as sd
        except Exception as exc:
            self.error = "sounddevice unavailable: %s" % exc
            return
        try:
            wf = wave.open(self.path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype="int16", device=self.device,
                                callback=self._callback):
                while not self._stopping.is_set() or not self._q.empty():
                    try:
                        chunk = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    wf.writeframes(chunk)
                    self.frames_written += len(chunk) // 2
            wf.close()
        except Exception as exc:
            self.error = str(exc)

    def stop(self):
        self._stopping.set()

    def info(self):
        return {
            "path": "audio.wav",
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "start_offset": (round(self.start_offset, 3)
                             if self.start_offset is not None else None),
            "duration": round(self.frames_written / SAMPLE_RATE, 3),
            "error": self.error,
        }
