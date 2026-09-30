"""OpenAI-compatible remote backend (delegates to an HTTP endpoint)."""
from .base import Backend
from . import register_backend


@register_backend("openai")
class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, cfg=None, base_url="http://127.0.0.1:8000", model="ascendforge", **kwargs):
        super().__init__(cfg, **kwargs)
        self.base_url = base_url
        self.model = model

    def generate(self, prompt, **kwargs):
        import json
        import urllib.request
        body = json.dumps({"model": self.model, "prompt": prompt}).encode()
        req = urllib.request.Request(f"{self.base_url}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def train_step(self, step):
        return 0.0

    def current_skill(self):
        return 1.0

    def evaluate(self):
        return {"backend": "openai"}
