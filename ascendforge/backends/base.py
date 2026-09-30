"""Backend abstract interface."""


class Backend:
    name = "base"

    def __init__(self, cfg=None, **kwargs):
        self.cfg = cfg

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

    def train_step(self, step):
        raise NotImplementedError

    def current_skill(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
