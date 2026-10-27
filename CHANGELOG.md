# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0 the API may change
between minor versions.

## [Unreleased]

## [0.3.0] - 2026-10-27

### Added

- Conversions: `implicability`, `mass_from_implicability`, `mass_from_plausibility`, and the
  disjunctive canonical decomposition `disjunctive_weights` / `from_disjunctive_weights`.
- `decombine_conjunctive` and `decombine_disjunctive`; Dempster `condition` and `decondition`.
- `corrections` module: `reinforce`, `negate`, and the contextual `contextual_reinforce`,
  `contextual_negate`, `contextual_dediscount` and `contextual_dereinforce` (Pichon et al. 2016),
  all differentiable in the correction degrees.
- `strong_dominance` and `weak_dominance`: set-valued decisions on a mass.
- `metrics` module: `discounted_accuracy`, `u65`, `u80`, `utility_score`,
  `error_rejection_curve` and `auerc`.
- `decision_scores`, `soft_decide`, `soft_dominance` and `max_mass_set`.
- Losses `pl_loss` (Denœux's E_pl, hard or soft labels) and `expected_utility_loss` (u65/u80).
- `EkNNHead`: evidential k-nearest neighbours with learned per-class γ.
- `learning` module: `ContextualCorrection` (learnable contextual discounting, reinforcement
  or negating, optionally per group), closed-form `fit_contextual`, and `hard_to_soft_labels`.
- Example `learned_correction.py`.
- `display` module: `to_string`, `to_html` and `to_latex` tables of a mass function.
- A `torch.compile(fullgraph=True)` test over heads, rules, decisions and losses.
- Visual guide sections on corrections and set-valued answers, with a new figure.
- `tests/test_evtools_crosscheck.py` checks these and the existing rules against
  [evtools](https://github.com/evtools-team/evtools) when `evtools-dst` is installed.

### Changed

- `contextual_discount` takes optional `contexts` (subset bitmasks); a partition of Θ gives
  Θ-contextual discounting. `discount` and `contextual_discount` moved to `corrections` and are
  still importable from `combine`.

### Fixed

- Dempster normalisation (`dempster`, `murphy`, normalised `cautious` and `condition`,
  `SparseMass.dempster`, `RandomInterval.conjunctive`) and `pignistic` divide by the sum of the
  non-empty masses instead of 1 − κ. In float32 at κ = 1 − 10⁻⁶ the old result was about 20 %
  off and its gradient about 40 %; it is now within 10⁻⁶ and 10⁻⁴.

## [0.2.0] - 2026-10-21

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

[Unreleased]: https://github.com/sulcantonin/torchbelief/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/sulcantonin/torchbelief/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/sulcantonin/torchbelief/releases/tag/v0.2.0
