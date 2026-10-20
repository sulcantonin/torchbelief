# Notes

## Scaling beyond the dense 2^n representation (2026-10-19)

Dense masses cost 2^n numbers and Dubois–Prade costs O(4^n), so the package is limited to
n ≲ 16 (Dubois–Prade n ≲ 10). Question raised: can metric learning approximate the explosion?

Short answer: not in the worst case (a general mass function carries 2^n numbers), but it can
exploit structure; network-produced masses usually concentrate on a few focal sets.

Options, least to most speculative:

1. **Sparse focal sets with merging (do first).** *Done: `torchbelief.sparse.SparseMass`, with
   `summarize(k)` using Lowrance summarisation as the outer-approximation merge step.* Keep top-k focal sets as bitmasks; Dubois–Prade
   becomes O(k²). After each combination, merge the smallest focal sets into a superset. This is an
   outer approximation (Bel only drops, Pl only rises), so credal intervals stay valid but looser.
   No learning needed. Benchmark interval looseness against the exact dense version at n ≤ 12.
2. **Learned compression of the k² product back to k.** Train the merge step to minimise a distance
   between exact and compressed masses (Jousselme, or Bel/Pl discrepancy on sampled events so the
   2^n events are never enumerated). Approximate; needs empirical validation.
3. **Box embeddings of focal sets in R^d.** Intersection of boxes is exact; union is the bounding
   box (an outer approximation, consistent with Dubois–Prade); empty intersection = zero volume,
   a differentiable conflict signal. Would allow n well beyond 16. Research-grade: conflict
   detection is fragile because blurred box edges change whether two sets count as disjoint.

Only pursue 2 and 3 if option 1 is not enough, e.g. open-vocabulary label spaces.

Related roadmap items are in README.md. PCR5/6, cautious/bold rules, p-boxes (`RandomInterval`) and
the Belnap layer were implemented in 0.2.0.
