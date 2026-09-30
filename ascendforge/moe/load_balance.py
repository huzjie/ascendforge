"""MoE load-balancing auxiliary losses."""


def aux_loss(load_counts, num_tokens):
    """Coefficient-of-variation style imbalance penalty."""
    if num_tokens == 0:
        return 0.0
    n = len(load_counts)
    mean = num_tokens / n
    var = sum((c - mean) ** 2 for c in load_counts) / n
    return var / (mean ** 2 + 1e-8)


def load_balance_loss(router_probs, expert_counts):
    """Combine routing probability and expert-count imbalance."""
    return aux_loss(expert_counts, sum(expert_counts))
