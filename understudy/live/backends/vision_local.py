"""Local vision backends over a localhost HTTP API (stdlib urllib only).

Backend contract: `name`, `model`, `run(payload) -> dict` with payload
{"prompt": str, "images": [base64 str, ...]} returning {"text": str}.
"""
import json
import urllib.error
import urllib.request


class BackendError(RuntimeError):
    pass


def _http(url, data=None, timeout=60.0, headers=None):
    body = None if data is None else json.dumps(data).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except (urllib.error.URLError, OSError) as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc


class OllamaBackend:
    name = "ollama"

    def __init__(self, model="qwen2.5vl:7b", host="http://localhost:11434",
                 timeout=120.0, num_predict=300):
        self.model, self.host = model, host.rstrip("/")
        self.timeout, self.num_predict = timeout, num_predict

    def run(self, payload):
        out = _http(self.host + "/api/generate", {
            "model": self.model, "prompt": payload["prompt"],
            "images": payload["images"], "stream": False,
            "options": {"temperature": 0.1, "num_predict": self.num_predict},
            "keep_alive": "30m",
        }, self.timeout)
        return {"text": (out.get("response") or "").strip()}

    def warm_up(self):
        """Load the model before recording, so the first frame doesn't pay for it."""
        _http(self.host + "/api/generate", {
            "model": self.model, "prompt": "ok", "stream": False,
            "options": {"num_predict": 1}, "keep_alive": "30m",
        }, max(self.timeout, 300.0))

    def probe(self):
        try:
            tags = _http(self.host + "/api/tags", timeout=5)
        except BackendError as exc:
            raise BackendError(
                f"Ollama is not reachable at {self.host}. Start it with "
                f"'ollama serve'. ({exc})") from exc
        names = [m.get("name", "") for m in tags.get("models", [])]
        want = self.model if ":" in self.model else self.model + ":latest"
        if want not in names:
            raise BackendError(
                f"Model '{self.model}' not found in Ollama. Run "
                f"'ollama pull {self.model}'. Installed: {', '.join(names) or 'none'}")
        return True


class OpenAICompatBackend:
    """llama.cpp server or LM Studio (/v1/chat/completions)."""
    name = "openai-compat"

    def __init__(self, model="qwen2.5-vl", host="http://localhost:8080",
                 timeout=120.0, max_tokens=300, api_key=None):
        self.model, self.host = model, host.rstrip("/")
        self.timeout, self.max_tokens, self.api_key = timeout, max_tokens, api_key

    def _headers(self):
        return {"Authorization": "Bearer " + self.api_key} if self.api_key else {}

    def run(self, payload):
        content = [{"type": "text", "text": payload["prompt"]}]
        for b in payload["images"]:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b}})
        out = _http(self.host + "/v1/chat/completions", {
            "model": self.model, "temperature": 0.1, "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }, self.timeout, self._headers())
        try:
            text = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected response: {str(out)[:200]}") from exc
        return {"text": (text or "").strip()}

    def probe(self):
        try:
            models = _http(self.host + "/v1/models", timeout=5,
                           headers=self._headers())
        except BackendError as exc:
            raise BackendError(
                f"No OpenAI-compatible server at {self.host}. Start llama.cpp "
                f"'llama-server -m <model> --mmproj <mmproj>' or LM Studio's "
                f"local server. ({exc})") from exc
        ids = [m.get("id", "") for m in models.get("data", [])]
        if ids and self.model not in ids:
            raise BackendError(
                f"Model '{self.model}' not served at {self.host}. Available: "
                f"{', '.join(ids)}")
        return True


class RemoteBackend:
    name = "remote"

    def __init__(self, model="", **kw):
        self.model = model

    def run(self, payload):
        raise NotImplementedError("remote vision backend is not implemented yet")

    def probe(self):
        raise NotImplementedError("remote vision backend is not implemented yet")
