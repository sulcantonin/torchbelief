"""Four sensors classify an object as car / truck / bike; one sensor is broken.

Shows: heterogeneous sources (mass, probability, possibility) as Capacities,
inconsistency measures, Shapley blame, blame-based discounting, rule comparison,
the Belnap view, and decide-or-abstain.
"""

import torch

import torchbelief as tb
from torchbelief import belnap, choice, inconsistency as I

torch.set_printoptions(precision=3, sci_mode=False)
F = tb.Frame(3, names=["car", "truck", "bike"])
d = torch.float64

lidar = tb.Capacity.from_mass(tb.simple_support(F, F.subset(["car"]), 0.7, dtype=d), F)
camera = tb.Capacity.from_probability(torch.tensor([0.75, 0.2, 0.05], dtype=d), F)
radar = tb.Capacity.from_possibility(torch.tensor([1.0, 0.6, 0.1], dtype=d), F)   # fuzzy: car or truck
broken = tb.Capacity.from_mass(tb.simple_support(F, F.subset(["bike"]), 0.9, dtype=d), F)
sources = [lidar, camera, radar, broken]
names = ["lidar", "camera", "radar", "broken"]

print("pairwise disagreement\n", I.disagreement_matrix(sources))
meas = I.inconsistency_measures(sources)
print("drastic", int(meas["drastic"]), "| MI pairs", int(meas["mi"]), "| max", round(float(meas["max"]), 3))

phi = I.shapley_blame(sources)
print("Shapley blame", dict(zip(names, phi.tolist())))

discounted, alpha = I.blame_discount(sources, strength=1.5)
print("reliability α", dict(zip(names, alpha.tolist())))

util = torch.eye(3, dtype=d)          # 1 if we name the right class
masses = [s.mass() for s in sources]
dmasses = [s.mass() for s in discounted]
print("\nrule          sources     BetP(car,truck,bike)     decision")
for rule in ["dempster", "yager", "dubois_prade", "pcr6", "murphy"]:
    for tag, ms in (("raw", masses), ("blame-disc.", dmasses)):
        m, k = tb.combine(ms, F, rule)
        cap = tb.Capacity.from_mass(m, F)
        act = int(choice.decide_or_abstain(cap, util))
        print(f"{rule:13s}{tag:12s} {cap.pignistic().numpy().round(3)!s:22s}  {F.names[act] if act >= 0 else 'ABSTAIN'}")

print("\nBelnap view of the raw sources (knowledge join):")
x = belnap.pool(sources)
for name, lab, deg in zip(F.names, belnap.label(x), belnap.degrees(x)):
    print(f"  {name:6s} {belnap.LABELS[lab]:8s} T/F/B/N = {deg.numpy()}")
