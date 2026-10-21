<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/logo/wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/logo/wordmark-light.svg" alt="torchbelief" width="420">
  </picture>
</p>

<p align="center"><b>Differentiable belief functions and imprecise probabilities for PyTorch.</b><br>
Fuse sources that disagree, find out which one is wrong, and decide, or abstain, when the evidence doesn't single out an answer.</p>

<p align="center"><b>New to belief functions? Start with the <a href="https://github.com/sulcantonin/torchbelief/blob/main/docs/visual-guide.md">visual guide</a></b>: ten pictures, no theory required.</p>

---

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/pipeline-dark.svg">
  <img src="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/pipeline-light.svg" alt="Five steps: describe the evidence, convert it to a Capacity, check for disagreement, fuse the sources, decide or abstain.">
</picture>

A probability gives one number per event. `torchbelief` keeps **two**, a lower and an upper bound
`[Bel(A), Pl(A)]`, and carries them through fusion, conflict handling and decision making. Every
operation is a batched PyTorch op, so the whole pipeline is differentiable and can sit inside a network.

The logo shows the same idea: the solid inner flame is **belief**, the outlined outer flame is
**plausibility**, and the gap between them is the imprecision the library tracks.

## Install

```bash
pip install torchbelief     # needs Python ≥ 3.9 and torch ≥ 2.0
pip install -e ".[dev]"     # from a clone, + pytest
```

## Quickstart

Four sensors classify an object; one of them is broken.

```python
import torch, torchbelief as tb
from torchbelief import inconsistency as I, choice

F = tb.Frame(3, names=["car", "truck", "bike"])
lidar  = tb.Capacity.from_mass(tb.simple_support(F, F.subset(["car"]), 0.7), F)    # belief function
camera = tb.Capacity.from_probability(torch.tensor([0.75, 0.20, 0.05]), F)         # probability
radar  = tb.Capacity.from_possibility(torch.tensor([1.0, 0.6, 0.1]), F)            # possibility
broken = tb.Capacity.from_mass(tb.simple_support(F, F.subset(["bike"]), 0.9), F)   # faulty sensor
sources = [lidar, camera, radar, broken]

I.shapley_blame(sources)                       # tensor([0.050, 0.108, 0.083, 0.608])  ← the broken one
fixed, alpha = I.blame_discount(sources)       # discount each source by its blame
m, conflict = tb.combine([s.mass() for s in fixed], F, rule="pcr6")
fused = tb.Capacity.from_mass(m, F)

fused.pignistic()                              # tensor([0.790, 0.103, 0.107])
fused.expectation_bounds(torch.eye(3))         # lower [0.73, 0.04, 0.08], upper [0.88, 0.19, 0.17]
choice.decide_or_abstain(fused, torch.eye(3))  # tensor(0) → "car"   (−1 would mean abstain)
```

### On real video

<img src="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/tracking-demo.webp" alt="YOLO11n with ByteTrack on a rainy street. For each track a side panel shows YOLO's label this frame and torchbelief's label with its belief interval per class; where YOLO flips between car and truck, torchbelief shows car, or car or truck.">

YOLO with ByteTrack on real traffic. YOLO relabels every frame, so a hatchback flickers between
"car" and "truck". A belief function per track ([`examples/tracking.py`](https://github.com/sulcantonin/torchbelief/blob/main/examples/tracking.py),
two lines per frame) says "car or truck" while the evidence is split. Across the clip, YOLO
changes label 74 times and torchbelief 20 times. The [visual guide](https://github.com/sulcantonin/torchbelief/blob/main/docs/visual-guide.md#9-real-video-labels-that-dont-flicker)
explains how.

### With a team of LLM agents

Five agents look for the root cause of an outage, each with one slice of the evidence: logs,
metrics, the deploy diff, network flows, past incidents. [`examples/multi_agent.py`](https://github.com/sulcantonin/torchbelief/blob/main/examples/multi_agent.py)
turns each reply into a belief function over the causes `deploy, database, network, dns, cache`:

- **API models** (`--live`, Claude) name a *set* of plausible causes and a confidence. The set is a
  focal set, so "not the network" and "can't tell" need no special handling.
- **Local models** (`--local`, any Hugging Face causal LM) are not asked for a confidence. Each
  cause name is scored by its token log-probability after the prompt. The probability the model
  puts on *none* of the names is treated as ignorance, so an agent that would rather answer
  something else is discounted automatically.

With Qwen2.5-1.5B-Instruct as every agent (`--temperature 5`, replies saved in
[`docs/data`](https://github.com/sulcantonin/torchbelief/blob/main/docs/data/multi-agent-qwen2.5-1.5b.json)):

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/agents-dark.svg">
  <img src="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/agents-light.svg" alt="Three panels. Shapley blame: the history agent gets 0.53, the other four agents 0.08 or less. Team belief in deploy: Dempster's rule gives 0.99 to 1.00 and 1.00 after a debate round; the cautious rule gives 0.79 to 0.95 and 0.84 to 0.97 after the debate. With only the logs and history agents replied, deploy, network and dns are all still possible.">
</picture>

The agent that matched an old DNS incident takes most of the blame. With only two replies in,
no cause beats the others yet, so the system waits or escalates. A debate round adds no new
evidence; Dempster's rule still counts it as new and reaches certainty, while the cautious
rule does not.

Two settings matter for local models. Instruct models put nearly all probability on one token,
so `--temperature T` flattens the scores (log-probabilities divided by T) before they become
beliefs. And five prompts on one base model are not five independent witnesses: Dempster's rule
(`--rule dempster`, the default, right for different models or non-overlapping evidence) fuses
four agents at 0.5–0.9 into 0.99, while the cautious rule (`--rule cautious`) does not count
their shared knowledge twice.

## The core idea

Everything is a **Capacity**: a lower set function `v` on the power set 2^Θ, stored as a dense
tensor `(..., 2**n)` whose index is the subset bitmask (`0` = ∅, `2**n − 1` = Θ). Each formalism
is a constructor that feeds in, and each summary is a view that reads out:

```
           probability ─┐                         ┌─► pignistic / Shapley value
            belief fn  ─┤                         ├─► Choquet lower / upper expectation
           possibility ─┼─►  Capacity (v, v*) ────┼─► Möbius mass, commonality, contour
            credal set ─┤                         ├─► credal vertices (core)
  random interval (ℝ)  ─┘ (via discretize)        └─► Belnap (t, f) per event
```

Möbius, zeta and commonality transforms are the fast O(n·2ⁿ) ones, applied axis by axis on a
`(2,)*n` reshape. All functions broadcast over leading batch dimensions and are differentiable.

## Feature tour

### Representations

| Object | What it is | Highlights |
|---|---|---|
| `Frame(n, names)` | frame of discernment, n ≤ 16 | `subset([...])`, `elements(mask)`, membership/cardinality tables |
| mass tensors | `(..., 2**n)` Möbius masses | `vacuous`, `categorical`, `simple_support`, `from_probabilities`, `from_dirichlet`, `negation`, `random_mass`, `is_valid` |
| `Capacity` | any monotone lower set function | `from_mass / from_probability / from_possibility / from_credal`, `.upper`, `.dual()`, `.mass()`, `.commonality()`, `.contour()`, `.pignistic()`, `.credal_vertices()`, `.choquet(f)`, `.expectation_bounds(U)`, `.discount(α)` |
| structure checks | what kind of capacity is this? | `is_belief`, `is_probability`, `is_possibility`, `monotonicity_violation`, `convexity_violation` (2-monotonicity) |
| `possibility` | π distributions | t-norm conjunctive rules (min, product, Łukasiewicz), disjunctive, Dubois–Prade **adaptive** rule, Yager-style rule, discounting, consistency h |
| `RandomInterval` | random sets of intervals on ℝ (p-boxes) | `cdf_bounds` (the p-box), `prob_bounds([a,b])`, `mean_bounds`, `expectation_bounds(f)`, Dempster `conjunctive`, `mixture`, `from_pbox`, `discretize(edges)` → finite frame; optional sigmoid `temperature` for endpoint gradients |
| `SparseMass` | focal sets as int64 bitmasks, **n ≤ 62** | O(k₁k₂) `conjunctive / disjunctive / dempster / yager / dubois_prade`, `discount`, `belief / plausibility` of any event, `pignistic`, `expectation_bounds`, `summarize(k)` (outer approximation: Bel ↓, Pl ↑), `from_dense / to_dense` |

### Combination rules

All take masses `(..., 2**n)`; the two-source rules return `(mass, conflict)`. `tb.combine(list, F, rule)`
folds any of them over many sources. The rules differ mainly in what they do with the conflicting mass:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/rules-dark.svg">
  <img src="https://raw.githubusercontent.com/sulcantonin/torchbelief/main/docs/figures/rules-light.svg" alt="Stacked bars: two disagreeing sources combined by seven rules; each rule puts the conflicting mass somewhere different.">
</picture>

| Rule | Where the conflict goes | Assoc. | Idempotent | Use when |
|---|---|:-:|:-:|---|
| `conjunctive` | stays on ∅ (open world, TBM) | ✓ | | you want to *see* the conflict |
| `dempster` | renormalised away (gradient ∝ 1/(1−κ)) | ✓ | | independent, reliable, agreeing sources |
| `yager` | to Θ (ignorance); bounded gradients | | | conflict should mean "don't know" |
| `dubois_prade` | to A ∪ B of the conflicting pair | | | one of the two is probably right |
| `pcr5` / `pcr6` | back to A and B, proportionally (DSmT) | | | conflict should stay with whoever caused it |
| `cautious` | min of conjunctive weights (Denœux) | ✓ | ✓ | **non-distinct** sources (shared data/models) |
| `bold` | min of disjunctive weights | ✓ | ✓ | non-distinct sources, at least one reliable (disjunctive) |
| `disjunctive` | no conflict, A ∪ B | ✓ | | at least one source is right, unknown which |
| `average`, `murphy` | averaged, then Dempster (Murphy) | | | quick robust baseline |

Reliability: `discount(m, α)` (Shafer) and `contextual_discount(m, β)` (Mercier–Quost–Denœux:
per-class reliability, e.g. "this sensor is good at cars but bad at bikes"). The canonical
decomposition is exposed as `conjunctive_weights` / `from_conjunctive_weights`.

### Inconsistency: detect, measure, blame, resolve

| Function | What it does |
|---|---|
| `inconsistency.disagreement(a, b)` | sure-loss gap between two capacities. **Exact** for 2-monotone pairs (belief, possibility, probability): it is 0 iff the credal sets intersect (Frank's discrete sandwich theorem) |
| `disagreement_matrix(sources)` | all pairwise gaps `(..., S, S)` |
| `inconsistency_measures(sources)` | argumentation-style knowledge-base measures (Hunter–Konieczny): drastic, MI count, max, sum, *problematic* and *free* sources |
| `shapley_blame(sources, measure)` | Shapley inconsistency value per source; it sums to the total, and the outlier gets the blame |
| `blame_discount(sources)` | discount each source by its blame, then fuse with any rule |
| `resolve(a, b, tau, strategy, temperature)` | Dempster if the sources agree, otherwise a fallback (`yager`, credal `union`, `dubois_prade`, `pcr5`). `temperature` turns the hard switch into a differentiable blend |
| `credal_union`, `credal_intersection`, `specific_conflict`, `fuse_belief` | building blocks |

### Paraconsistent view: Belnap's four values

`torchbelief.belnap` reads each event as a bilattice value `(t, f) = (v(A), v(Ā))`. Pooling sources
with the knowledge join exposes contradiction instead of averaging it away.

```python
x = belnap.pool(sources)            # (n, 2) per singleton
belnap.label(x)                     # true / false / both / neither
belnap.degrees(x)                   # soft T, F, B, N (sums to 1)
belnap.contradiction(x), belnap.ignorance(x)
# bilattice algebra: negate, truth_and, truth_or, knowledge_join (⊕), knowledge_meet (⊗)
```

### Decisions

Utilities are a matrix `U (acts, n)`; `-1` means abstain.

| | |
|---|---|
| `choice` (any capacity) | `maximin` (Γ-maximin), `maximax`, `hurwicz(α)`, `pignistic_choice`, Walley `maximal_acts`, `interval_dominance_acts`, `e_admissible_acts`, **`decide_or_abstain(rule)`** |
| `decision` (masses) | `pignistic`, `plausibility_transform`, `interval` (Bel, Pl of singletons), `vacuity`, `expected_utility_bounds`, `decide(criterion)` |
| uncertainty measures | `nonspecificity`, `deng_entropy`, `pignistic_entropy`, `ambiguity_measure` (Jiroušek–Shenoy), `jousselme_distance`, `conflict_weight` |

### Neural layers and losses

| Layer | Output | Notes |
|---|---|---|
| `MassHead(d, F, focal)` | softmax over chosen focal sets | singletons + Θ, `"all"`, or an explicit list |
| `EvidentialHead(d, F)` | Dirichlet α → subjective-logic mass | evidential deep learning (Sensoy et al.) |
| `PrototypeHead(d, F, k)` | Denœux's **ENN**: distance-based evidence from k prototypes, fused by Dempster in closed form | m(Θ) → 1 away from the training data |

Losses: `pignistic_nll`, `belief_nll` (−log Bel{y}), `plausibility_nll` (−log Pl{y}),
`hurwicz_nll(λ)` interpolating the two, and `nonspecificity_penalty`.

From [`examples/evidential_classifier.py`](https://github.com/sulcantonin/torchbelief/blob/main/examples/evidential_classifier.py), probing far from the training blobs:

```
EvidentialHead + MLP          PrototypeHead (ENN)
[12, 12]   m(Θ)=0.004  → a    m(Θ)=1.000  → ABSTAIN
[-15, 5]   m(Θ)=0.004  → b    m(Θ)=0.999  → ABSTAIN
```

The Dirichlet head extrapolates evidence linearly, just as a softmax does. The prototype head knows it is out of its depth.

## Examples

```bash
python examples/sensor_fusion.py          # measures, blame, 5 rules × raw/discounted, Belnap table
python examples/evidential_classifier.py  # Dirichlet head vs ENN, in- and out-of-distribution
python examples/tracking.py --replay docs/data/osorno-yolo11n.json   # belief labels for saved YOLO tracks
python examples/tracking.py your.mp4      # live YOLO + ByteTrack (pip install ultralytics, AGPL-3.0)
python examples/multi_agent.py            # LLM agents triage an outage (scripted replies, no model)
python examples/multi_agent.py --local Qwen/Qwen2.5-1.5B-Instruct --temperature 5 --rule cautious
python examples/multi_agent.py --replay docs/data/multi-agent-qwen2.5-1.5b.json --temperature 5 --rule cautious
python examples/multi_agent.py --live     # Claude agents (pip install anthropic)
python docs/make_figures.py               # regenerate every figure in docs/figures (needs matplotlib)
```

The [visual guide](https://github.com/sulcantonin/torchbelief/blob/main/docs/visual-guide.md) walks through all of it with pictures and runnable snippets.

## Complexity and limits

| | cost | practical limit |
|---|---|---|
| dense transforms, conjunctive/disjunctive/Dempster/Yager | O(n·2ⁿ) | n ≤ 16 |
| Dubois–Prade, PCR5, `convexity_violation` | O(4ⁿ) memory | n ≲ 10 |
| PCR6 over S sources | O(2^{nS}) | small n·S |
| `credal_vertices`, `e_admissible_acts` | n! permutations | n ≲ 8 |
| `shapley_blame` | 2^S coalitions | S ≲ 12 sources |
| `SparseMass` | O(k₁k₂) per combination + `summarize(k)` | n ≤ 62 |

## Caveats

- `credal_vertices` (and so E-admissibility) is exact only for 2-monotone capacities.
- `credal_intersection` is a pointwise bound, not the full natural extension.
- Inconsistency *measures* are pairwise. Three sources can be jointly inconsistent while every pair is consistent.
- `cautious` needs non-dogmatic sources (m(Θ) > 0) and `bold` needs subnormal ones (m(∅) > 0). Discount first if necessary.
- `RandomInterval` queries are differentiable in the masses; for endpoint gradients pass a `temperature`.

## Roadmap

- learned compression of sparse combinations and box embeddings of focal sets (see [NOTES.md](https://github.com/sulcantonin/torchbelief/blob/main/NOTES.md))
- exact multi-source consistency and natural extension via a differentiable LP layer
- conditioning / Bayes-like updating of capacities (Dempster, geometric, generalized Bayes)
- continuous frames beyond intervals (random closed sets in ℝᵈ)

## Tests

```bash
pytest -q        # 42 tests: brute-force checks of every rule, roundtrips, gradients, layers
```

## References

Shafer 1976 · Smets & Kennes 1994 (TBM) · Yager 1987 · Dubois & Prade 1988 · Murphy 2000 ·
Smarandache & Dezert 2005 (PCR5) · Martin & Osswald 2006 (PCR6) · Denœux 2000 (ENN), 2008 (cautious/bold) ·
Mercier, Quost & Denœux 2008 (contextual discounting) · Walley 1991 · Troffaes 2007 (decision rules) ·
Frank 1982 (discrete sandwich theorem) · Hunter & Konieczny 2010 (Shapley inconsistency values) ·
Belnap 1977 · Ferson et al. 2003 (p-boxes) · Lowrance, Garvey & Strat 1986 (summarization) ·
Sensoy, Kaplan & Kandemir 2018 (evidential deep learning) · Jousselme et al. 2001.

## Citing

If you use torchbelief in research, please cite it. GitHub's "Cite this repository" button
gives APA and BibTeX from [`CITATION.cff`](https://github.com/sulcantonin/torchbelief/blob/main/CITATION.cff);
each release is archived on Zenodo with its own DOI.

## License

MIT © Antonin Sulc. Changes are listed in the [changelog](https://github.com/sulcantonin/torchbelief/blob/main/CHANGELOG.md).
