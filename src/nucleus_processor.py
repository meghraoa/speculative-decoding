import torch
import torch.nn.functional as F

class NucleusProcessor:
    """Nucleus: Top-p sampling."""

    def __init__(self, temperature=0.6, top_p=0.9):
        self.temperature = temperature
        self.top_p = top_p

    def _process(self, logits):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        sorted_logits[sorted_indices_to_remove] = -1e4
        logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(-1))
        return logits

    def sample(self, probs):
        return torch.multinomial(probs, num_samples=1)

    def __call__(self, logits):
        proc = self._process(logits)
        return F.softmax(proc / self.temperature, dim=-1)
