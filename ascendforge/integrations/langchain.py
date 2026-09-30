"""LangChain LLM wrapper for ascendforge (optional dependency).

Use with langchain if installed; degrades gracefully otherwise.
"""
from ..backends import make_backend


class AscendForgeLLM:
    """A minimal LangChain-compatible LLM wrapper around the ascendforge backend."""

    def __init__(self, backend="mock", cfg=None, **kwargs):
        self.backend = make_backend(backend, cfg=cfg)

    def _call(self, prompt, **kwargs):
        out = self.backend.generate(prompt)
        return "ACCEPT" if out else "REJECT"

    def _identifying_params(self):
        return {"backend": self.backend.name}

    # LangChain BaseLLM duck-typing
    @property
    def _llm_type(self):
        return "ascendforge"

    def invoke(self, prompt, **kwargs):
        return self._call(prompt)

    def predict(self, prompt, **kwargs):
        return self._call(prompt)
