"""torchbelief — differentiable belief functions and imprecise probabilities in PyTorch."""

from .frame import Frame
from .mass import (vacuous, categorical, simple_support, from_probabilities, from_dirichlet,
                   negation, random_mass, is_valid)
from .transforms import (belief, plausibility, commonality, subset_sum, superset_sum,
                         mass_from_belief, mass_from_commonality)
from .combine import (conjunctive, disjunctive, dempster, yager, dubois_prade, pcr5, pcr6,
                      cautious, bold, conjunctive_weights, from_conjunctive_weights, conflict,
                      average, murphy, combine, discount, contextual_discount)
from .decision import (pignistic, plausibility_transform, interval, vacuity,
                       expected_utility_bounds, decide, nonspecificity, deng_entropy,
                       pignistic_entropy, ambiguity_measure, jousselme_distance,
                       conflict_weight)
from .capacity import Capacity, choquet
from .continuous import RandomInterval
from .sparse import SparseMass
from . import possibility, inconsistency, choice, belnap
from .layers import (MassHead, EvidentialHead, PrototypeHead, pignistic_nll, belief_nll, plausibility_nll,
                     hurwicz_nll, nonspecificity_penalty)

__version__ = "0.2.0"
