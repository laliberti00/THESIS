"""Situational archetype interpretability — post-training analysis.

STUB — implementation will land in the evaluation step.

Planned signatures:

    def characterise_situations(model, train_df, k_situations, top_n=10): ...
        '''For each of the K latent situations: feature activations,
        typical users, top-recommended items, candidate nominal label.'''

    def label_archetypes(characterisation, taxonomy): ...
        '''Map raw archetypes to readable labels (e.g. "Coffee shop,
        weekday morning") using the Foursquare category taxonomy.'''
"""


def _stub(*_args, **_kwargs):
    raise NotImplementedError(
        "pipeline.step03_evaluation.interpretability — not implemented yet")


characterise_situations = _stub
label_archetypes = _stub
