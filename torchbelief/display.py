"""Readable tables of a single mass function: plain text, HTML and LaTeX.

Each row is a focal set (or every subset with ``all_sets=True``) with its mass, belief,
plausibility and commonality. ``m`` is one mass (2**n,); index a batch first.
"""

from typing import List, Tuple

import torch

from .frame import Frame
from . import transforms as T

COLUMNS = ("m", "Bel", "Pl", "Q")


def _rows(m: torch.Tensor, frame: Frame, all_sets: bool, tol: float) -> List[Tuple[int, List[float]]]:
    if m.dim() != 1 or m.shape[0] != frame.size:
        raise ValueError(f"expected one mass of shape ({frame.size},); index a batch first")
    vals = torch.stack([m, T.belief(m, frame), T.plausibility(m, frame), T.commonality(m, frame)], -1)
    keep = range(frame.size) if all_sets else [A for A in range(frame.size) if abs(float(m[A])) > tol]
    order = sorted(keep, key=lambda A: (bin(A).count("1"), A))
    return [(A, vals[A].tolist()) for A in order]


def _label(A: int, frame: Frame, empty: str = "∅", theta: str = "Θ",
           fmt=lambda s: "{" + s + "}") -> str:
    if A == 0:
        return empty
    if A == frame.theta and frame.n > 1:
        return theta
    return fmt(", ".join(frame.elements(A)))


def to_string(m: torch.Tensor, frame: Frame, all_sets: bool = False, digits: int = 3,
              bar_width: int = 16, tol: float = 1e-9) -> str:
    """Aligned text table; ``bar_width > 0`` adds a bar proportional to m(A)."""
    rows = _rows(m, frame, all_sets, tol)
    labels = [_label(A, frame) for A, _ in rows]
    w = max([len(s) for s in labels] + [3])
    num = digits + 3
    lines = [f"{'set':<{w}}  " + "  ".join(f"{c:>{num}}" for c in COLUMNS)]
    top = max([abs(v[0]) for _, v in rows] + [1e-12])
    for label, (_, v) in zip(labels, rows):
        line = f"{label:<{w}}  " + "  ".join(f"{x:>{num}.{digits}f}" for x in v)
        if bar_width:
            line += "  " + "█" * round(bar_width * max(v[0], 0) / top)
        lines.append(line.rstrip())
    return "\n".join(lines)


def to_html(m: torch.Tensor, frame: Frame, all_sets: bool = False, digits: int = 3,
            tol: float = 1e-9) -> str:
    """An HTML <table>, e.g. for ``IPython.display.HTML``."""
    import html
    head = "".join(f"<th>{c}</th>" for c in ("set",) + COLUMNS)
    body = "".join(
        f"<tr><td>{html.escape(_label(A, frame))}</td>"
        + "".join(f'<td style="text-align:right">{x:.{digits}f}</td>' for x in v) + "</tr>"
        for A, v in _rows(m, frame, all_sets, tol))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def to_latex(m: torch.Tensor, frame: Frame, all_sets: bool = False, digits: int = 3,
             tol: float = 1e-9) -> str:
    """A ``tabular`` environment (needs no packages beyond plain LaTeX)."""
    esc = lambda s: s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")
    lines = [r"\begin{tabular}{l" + "r" * len(COLUMNS) + "}",
             "set & " + " & ".join(COLUMNS) + r" \\ \hline"]
    for A, v in _rows(m, frame, all_sets, tol):
        label = _label(A, frame, r"$\emptyset$", r"$\Theta$", lambda s: r"$\{$" + esc(s) + r"$\}$")
        lines.append(label + " & " + " & ".join(f"{x:.{digits}f}" for x in v) + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)
