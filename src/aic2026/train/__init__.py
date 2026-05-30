# SPEC-0014: C1 DiacriticBERT training package.
"""Training code for the C1 diacritic-robust retrieval head.

Modules:
  - ``diacritic_noise``  : the controlled diacritic-noise function (pure, CPU).
  - ``diacritic_corpus`` : build the contrastive corpus from public Vietnamese
                           text (or any clean strings).
  - ``diacritic_bert``   : frozen-BGE-M3 + projection head + InfoNCE training
                           (lazy-imports torch; only needed on the GPU box).
"""
