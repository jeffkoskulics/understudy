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


class AudioRecorder(threading.Thread):
    def __init__(self, clock, path, device=None):
        super().__init__(daemon=True)
        self.clock = clock
        self.path = path
        self.device = device
        self.start_offset = None   # session time of the first sample
        self.frames_written = 0
        self.error = None
        self._q = queue.Queue()
        self._stopping = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if self.start_offset is None:
            self.start_offset = self.clock.t()
        self._q.put(bytes(indata))

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
