"""Local streaming backend: faster-whisper, same conventions as transcribe.py
(model name default, int8 on CPU). The model loads lazily on first run."""
import array

from ... import transcribe as batch


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, model=batch.DEFAULT_MODEL, device="auto",
                 compute_type="int8", language=None):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._whisper = None

    def run(self, payload):
        """payload: {"pcm": int16 bytes, "sample_rate": 16000, ...}
        -> {"text": str, "words": [{"w", "t0", "t1"}]}, times relative to the
        start of the window."""
        if self._whisper is None:
            self._whisper = batch._load_model(self.model, self.device,
                                              self.compute_type)
        pcm = array.array("h")
        pcm.frombytes(payload["pcm"][:len(payload["pcm"]) // 2 * 2])
        try:
            import numpy as np
            audio = np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
        except ImportError:  # faster-whisper needs numpy anyway
            audio = [s / 32768.0 for s in pcm]
        segments, _info = self._whisper.transcribe(
            audio, word_timestamps=True, vad_filter=False,
            language=self.language)
        words = []
        for seg in segments:
            for w in (seg.words or []):
                words.append({"w": w.word.strip(), "t0": w.start, "t1": w.end})
        return {"text": " ".join(w["w"] for w in words), "words": words}
