"""Fairness metrics for top-N recommendation (Ge et al. 2022).

STUB — implementation will land in the evaluation step.

Planned signatures:

    def kl_divergence(recommended_counts, target_distribution): ...
    def long_tail_rate(recommended_items, item_popularity_groups): ...
    def fairness_per_situation(per_user_metrics, situation_assignments): ...

References:
    Ge, Y., Liu, S., Gao, R., Xian, Y., Li, Y., Zhao, X., Pei, C., Sun, F.,
    Ge, J., Ou, W., & Zhang, Y. (2022). "Towards Long-term Fairness in
    Recommendation." WSDM 2022.
"""


def _stub(*_args, **_kwargs):
    raise NotImplementedError(
        "pipeline.step03_evaluation.fairness — not implemented yet")


kl_divergence = _stub
long_tail_rate = _stub
fairness_per_situation = _stub
