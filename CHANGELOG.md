# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0 the API may change
between minor versions.

## [Unreleased]

## [0.2.0] - unreleased

First public release.

### Added

- `Capacity`: any monotone lower set function on 2^Θ, built from masses, probabilities,
  possibility distributions or credal sets, with Choquet expectations, credal vertices,
  pignistic and Shapley summaries, discounting and structure checks.
- Mass functions and fast O(n·2ⁿ) Möbius, zeta and commonality transforms, batched and
  differentiable.
- Combination rules: conjunctive, disjunctive, Dempster, Yager, Dubois–Prade, PCR5, PCR6,
  cautious, bold, averaging and Murphy, with Shafer and contextual discounting.
- `inconsistency`: pairwise disagreement, knowledge-base inconsistency measures, Shapley blame,
  blame-based discounting and conflict-dependent `resolve`.
- `belnap`: Belnap's four-valued view of pooled sources.
- `possibility`: t-norm, disjunctive, adaptive and Yager-style rules for possibility
  distributions.
- `RandomInterval`: random intervals on ℝ (p-boxes), with discretisation to finite frames.
- `SparseMass`: focal sets as int64 bitmasks for frames up to 62 elements, with Lowrance
  summarisation.
- Decision rules (`choice`, `decision`): Γ-maximin, Hurwicz, Walley maximality, interval
  dominance, E-admissibility and `decide_or_abstain`; uncertainty measures and the Jousselme
  distance.
- Neural layers and losses: `MassHead`, `EvidentialHead`, `PrototypeHead` (ENN) and belief,
  plausibility and Hurwicz likelihood losses.
- Examples: sensor fusion, evidential classifier, belief tracking on YOLO tracks, and
  `multi_agent.py`, where LLM agents (Claude, or a local model scored by token log-probabilities)
  diagnose an outage with Shapley blame, abstention and the cautious rule for agents that share
  a base model.
- A visual guide and figures under `docs/`.

[Unreleased]: https://github.com/sulcantonin/torchbelief/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sulcantonin/torchbelief/releases/tag/v0.2.0
