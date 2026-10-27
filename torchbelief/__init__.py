"""torchbelief — differentiable belief functions and imprecise probabilities in PyTorch."""

from .frame import Frame
from .mass import (vacuous, categorical, simple_support, from_probabilities, from_dirichlet,
                   negation, random_mass, is_valid)
from .transforms import (belief, plausibility, commonality, implicability, subset_sum,
                         superset_sum, mass_from_belief, mass_from_commonality,
                         mass_from_implicability, mass_from_plausibility)
from .combine import (conjunctive, disjunctive, dempster, yager, dubois_prade, pcr5, pcr6,
                      cautious, bold, conjunctive_weights, from_conjunctive_weights,
                      disjunctive_weights, from_disjunctive_weights, conflict, average, murphy,
                      combine, decombine_conjunctive, decombine_disjunctive, condition,
                      decondition)
from .corrections import (discount, contextual_discount, contextual_dediscount, reinforce,
                          contextual_reinforce, contextual_dereinforce, negate, contextual_negate)
from .decision import (pignistic, plausibility_transform, interval, vacuity,
                       expected_utility_bounds, decision_scores, decide, soft_decide,
                       strong_dominance, weak_dominance, max_mass_set, soft_dominance, nonspecificity,
                       deng_entropy, pignistic_entropy, ambiguity_measure, jousselme_distance,
                       conflict_weight)
from .capacity import Capacity, choquet
from .continuous import RandomInterval
from .sparse import SparseMass
from . import possibility, inconsistency, choice, belnap, metrics
from .layers import (MassHead, EvidentialHead, PrototypeHead, EkNNHead, pignistic_nll, belief_nll,
                     plausibility_nll, hurwicz_nll, nonspecificity_penalty, pl_loss,
                     expected_utility_loss)
from .learning import ContextualCorrection, fit_contextual, hard_to_soft_labels

__version__ = "0.2.0"
