"""Regenerate the figures in docs/figures (light + dark SVG of each).

    python docs/make_figures.py

Every number drawn comes from torchbelief itself, so the pictures stay honest
when the library changes.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import torchbelief as tb  # noqa: E402
import tracking  # noqa: E402  (examples/tracking.py: the belief layer for YOLO tracks)
import multi_agent  # noqa: E402  (examples/multi_agent.py: LLM agents as belief functions)
from torchbelief import belnap, choice, inconsistency as I  # noqa: E402
from torchbelief.continuous import RandomInterval  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")
D = torch.float64

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", border="#e1e0d9",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
                  neutral="#b4b2a9", band="#b7d3f6", box="#f3f2ee",
                  seq=["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", border="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"],
                 neutral="#6b6a64", band="#1c4a80", box="#242423",
                 seq=["#1f2633", "#14365e", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]),
}

F = tb.Frame(3, names=["car", "truck", "bike"])
CLASSES = F.names


def style(t):
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "svg.fonttype": "none",
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"], "ytick.labelcolor": t["ink2"],
        "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": t["ink"],
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": t["ink2"],
        "hatch.color": t["muted"], "hatch.linewidth": 0.8,
    })


def titled(ax, title, sub=None, t=None, size=None):
    """Bold title with an optional grey subtitle underneath, never overlapping."""
    lines = 0 if sub is None else sub.count("\n") + 1
    ax.set_title(title, pad=8 + 14 * lines, **({"fontsize": size} if size else {}))
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, va="bottom", fontsize=9.5, color=t["ink2"])


def save(fig, name, theme):
    fig.savefig(os.path.join(OUT, f"{name}-{theme}.svg"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def cap_mass(**focal):
    """Capacity from {"car": .6, "car+truck": .2, "all": .2} style keyword masses."""
    m = torch.zeros(F.size, dtype=D)
    for k, v in focal.items():
        idx = F.theta if k == "all" else F.subset(k.split("_"))
        m[idx] += v
    return m


# ---- 1. one number vs two -----------------------------------------------------------------

def fig_intervals(t):
    sources = [
        ("A probability", "the camera says 75 / 20 / 5 %",
         tb.Capacity.from_probability(torch.tensor([0.75, 0.2, 0.05], dtype=D), F)),
        ("A belief function", "the lidar is 70 % sure it is a car,\nthe other 30 % it can't tell",
         tb.Capacity.from_mass(cap_mass(car=0.7, all=0.3), F)),
        ("No information", "a sensor that saw nothing",
         tb.Capacity.vacuous(F, dtype=D)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), sharey=True)
    for ax, (title, sub, cap) in zip(axes, sources):
        bel, pl = cap.expectation_bounds(torch.eye(3, dtype=D))
        bet = cap.pignistic()
        for i in range(3):
            y = 2 - i
            ax.add_patch(Rectangle((0, y - 0.16), 1, 0.32, color=t["grid"], lw=0))
            ax.add_patch(Rectangle((0, y - 0.16), float(bel[i]), 0.32, color=t["series"][0], lw=0))
            ax.add_patch(Rectangle((float(bel[i]), y - 0.16), float(pl[i] - bel[i]), 0.32,
                                   color=t["band"], lw=0))
            ax.plot([float(bet[i])], [y], "o", ms=8, mfc=t["ink"], mec=t["surface"], mew=1.5)
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([2, 1, 0], CLASSES)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        titled(ax, title, sub, t)
    handles = [Rectangle((0, 0), 1, 1, color=t["series"][0]), Rectangle((0, 0), 1, 1, color=t["band"]),
               plt.Line2D([], [], marker="o", ls="", mfc=t["ink"], mec=t["surface"], ms=8)]
    fig.legend(handles, ["Bel: support you can count on", "Pl − Bel: could go either way",
                         "best single guess (pignistic)"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.12)
    return fig


# ---- 2. how subsets are stored ---------------------------------------------------------------

def fig_powerset(t):
    pos = {0: (0, 0), 1: (-1, 1), 2: (0, 1), 4: (1, 1), 3: (-1, 2), 5: (0, 2), 6: (1, 2), 7: (0, 3)}
    mass = {1: 0.5, 3: 0.3, 7: 0.2}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for a in pos:
        for b in pos:
            if a != b and (a & b) == a and bin(b).count("1") == bin(a).count("1") + 1:
                (x0, y0), (x1, y1) = pos[a], pos[b]
                ax.plot([x0, x1], [y0, y1], color=t["axis"], lw=1.5, zorder=1)
    for idx, (x, y) in pos.items():
        name = "∅" if idx == 0 else ("Θ = everything" if idx == 7 else
                                     "{" + ", ".join(F.elements(idx)) + "}")
        has = idx in mass
        ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.25), 0.84, 0.5, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=t["series"][0] if has else t["box"],
                                    ec=t["series"][0] if has else t["border"], lw=1.2, zorder=2))
        ink = "#ffffff" if has else t["ink"]
        ax.text(x, y + 0.07, name, ha="center", va="center", fontsize=10.5, color=ink,
                fontweight="bold" if has else "normal", zorder=3)
        ax.text(x, y - 0.12, f"m[{idx}] · 0b{idx:03b}" + (f" = {mass[idx]:.1f}" if has else ""),
                ha="center", va="center", fontsize=7.5, color="#ffffff" if has else t["muted"],
                family="monospace", zorder=3)
    ax.text(-1.75, 3.0, "bit 0 = car\nbit 1 = truck\nbit 2 = bike", fontsize=9.5, color=t["ink2"],
            family="monospace", va="center")
    ax.text(-1.8, 0.0, "the mass tensor m has 2³ = 8 slots,\none per subset; blue slots are\n"
            "non-zero in this example", fontsize=9.5, color=t["ink2"], ha="left", va="center")
    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-0.45, 3.4); ax.axis("off")
    ax.set_title("Every subset of {car, truck, bike} is a slot in the tensor")
    return fig


# ---- 3. where the conflict goes ---------------------------------------------------------------

GROUPS = [("{car}", [1]), ("{truck}", [2]), ("{bike}", [4]), ("two-class sets", [3, 5, 6]),
          ("Θ: don't know", [7]), ("∅: conflict", [0])]


def fig_rules(t):
    a = cap_mass(car=0.6, car_truck=0.2, all=0.2)
    b = cap_mass(bike=0.7, all=0.3)
    rows = [("source A", a, "mostly car"), ("source B", b, "mostly bike"), None]
    for name, fn, note in [
        ("conjunctive", lambda: tb.conjunctive(a, b, F), "keeps the conflict visible on ∅"),
        ("dempster", lambda: tb.dempster(a, b, F)[0], "renormalises it away: overconfident"),
        ("yager", lambda: tb.yager(a, b, F)[0], "turns it into “don't know”"),
        ("dubois_prade", lambda: tb.dubois_prade(a, b, F)[0], "“car or bike”: one of them is right"),
        ("pcr5", lambda: tb.pcr5(a, b, F)[0], "gives it back to whoever caused it"),
        ("cautious", lambda: tb.cautious(a, b, F)[0], "for sources that share data"),
        ("disjunctive", lambda: tb.disjunctive(a, b, F), "“at least one source is right”"),
    ]:
        rows.append((name, fn(), note))
    colors = t["series"] + [t["neutral"], t["surface"]]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = len(rows)
    ticks, labels = [], []
    for r in rows:
        y -= 1
        if r is None:
            ax.axhline(y, color=t["grid"], lw=1)
            ax.text(-0.01, y, "combined with ↓", ha="right", va="center", color=t["muted"], fontsize=9.5)
            continue
        name, m, note = r
        left = 0.0
        for (gname, idxs), c in zip(GROUPS, colors):
            w = float(m[idxs].sum())
            if w <= 1e-9:
                continue
            ax.barh(y, w, left=left, height=0.62, color=c, edgecolor=t["surface"], lw=2)
            if gname.startswith("∅"):
                ax.barh(y, w - 0.006, left=left + 0.003, height=0.56, fill=False, edgecolor=t["muted"],
                        lw=0.8, hatch="////")
            if w >= 0.075:
                light = c in (t["series"][0], t["series"][1], t["series"][2]) or (
                    c == t["neutral"] and t["surface"] != "#fcfcfb")
                ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if light else t["ink"],
                        bbox=dict(fc=t["surface"], ec="none", pad=1.5) if gname.startswith("∅") else None)
            left += w
        ax.text(1.02, y, note, va="center", fontsize=9.5, color=t["ink2"])
        ticks.append(y); labels.append(name)
    ax.set_yticks(ticks, labels, family="monospace")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1], ["0", "50 %", "100 %"])
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec=t["muted"] if g.startswith("∅") else c,
                         hatch="////" if g.startswith("∅") else None, lw=0.8)
               for (g, _), c in zip(GROUPS, colors)]
    ax.legend(handles, [g for g, _ in GROUPS], ncol=6, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.2, columnspacing=1.3, fontsize=9.5)
    ax.set_title("Two sources disagree: where does each rule put the conflicting mass?", pad=34)
    return fig


# ---- 4. who is to blame -------------------------------------------------------------------------

def sensors():
    lidar = tb.Capacity.from_mass(cap_mass(car=0.7, all=0.3), F)
    camera = tb.Capacity.from_probability(torch.tensor([0.75, 0.2, 0.05], dtype=D), F)
    radar = tb.Capacity.from_possibility(torch.tensor([1.0, 0.6, 0.1], dtype=D), F)
    broken = tb.Capacity.from_mass(cap_mass(bike=0.9, all=0.1), F)
    return ["lidar", "camera", "radar", "broken"], [lidar, camera, radar, broken]


def fig_blame(t):
    names, src = sensors()
    Dm = I.disagreement_matrix(src).numpy()
    phi = I.shapley_blame(src).numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    a1.imshow(Dm, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a1.text(j, i, f"{Dm[i, j]:.2f}", ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if Dm[i, j] > 0.45 else t["ink2"])
    a1.set_xticks(range(4), names); a1.set_yticks(range(4), names)
    a1.tick_params(length=0)
    for s in a1.spines.values():
        s.set_visible(False)
    titled(a1, "1 · Pairwise disagreement", "0 = compatible, 1 = flatly contradict", t)
    y = list(range(3, -1, -1))
    a2.barh(y, phi, height=0.55, color=t["series"][0])
    for yy, v in zip(y, phi):
        a2.text(v + 0.012, yy, f"{v:.0%}", va="center", fontsize=9.5, color=t["ink2"])
    a2.set_yticks(y, names); a2.tick_params(axis="y", length=0)
    a2.spines["left"].set_visible(False)
    a2.set_xlim(0, 0.75); a2.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"])
    a2.grid(axis="x", color=t["grid"], lw=0.8); a2.set_axisbelow(True)
    titled(a2, "2 · Shapley blame: who causes the inconsistency?",
           f"the shares add up to the total disagreement ({phi.sum():.2f})", t)
    fig.subplots_adjust(wspace=0.35)
    return fig


# ---- 5. decide or abstain ----------------------------------------------------------------------------

def fig_abstain(t):
    names, src = sensors()
    lidar, radar = src[0], src[2]
    sharp = tb.Capacity.from_mass(tb.dempster(lidar.mass(), radar.mass(), F)[0], F)
    vague = I.fuse_belief(src, "yager")[0]
    util = torch.eye(3, dtype=D)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    for ax, cap, title in [(axes[0], sharp, "lidar + radar agree"),
                           (axes[1], vague, "all four sensors, incl. the broken one (Yager)")]:
        lo, hi = (x.numpy() for x in cap.expectation_bounds(util))
        act = int(choice.decide_or_abstain(cap, util))                  # Walley maximality
        keep = choice.maximal_acts(cap, util).numpy()
        best_lo = lo.max()
        ax.axvline(best_lo, color=t["muted"], ls=(0, (3, 3)), lw=1)
        ax.text(best_lo, 2.55, " best guaranteed\n chance", fontsize=8.5, color=t["muted"], va="bottom")
        for i in range(3):
            y = 2 - i
            c = t["series"][0] if keep[i] else t["neutral"]
            ax.plot([lo[i], hi[i]], [y, y], color=c, lw=6, solid_capstyle="round")
            ax.plot([lo[i], hi[i]], [y, y], "|", color=c, ms=14, mew=2.5)
            ax.text(hi[i] + 0.02, y, f"[{lo[i]:.2f}, {hi[i]:.2f}]" + ("" if keep[i] else "  ruled out"),
                    va="center", fontsize=9, color=t["ink2"])
        ax.set_xlim(0, 1.25); ax.set_ylim(-0.6, 3.3)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_yticks([2, 1, 0], [f"say “{c}”" for c in CLASSES]); ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        verdict = f"→ decide: {CLASSES[act]}" if act >= 0 else "→ abstain (−1)"
        ax.set_title(f"{title}\n{verdict}", fontsize=11.5)
        ax.set_xlabel("chance of being right: [lower, upper]")
    fig.subplots_adjust(wspace=0.1)
    return fig


# ---- 6. knowing what you don't know --------------------------------------------------------------------

def fig_ood(t):
    torch.manual_seed(0)
    Fa = tb.Frame(3, names=["a", "b", "c"])
    centres = torch.tensor([[0., 3.], [-3., -2.], [3., -2.]])
    y = torch.randint(0, 3, (600,))
    x = centres[y] + torch.randn(600, 2)
    mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), tb.EvidentialHead(32, Fa))
    enn = tb.PrototypeHead(2, Fa, init=x[torch.randperm(600)[:12]])
    for model in (mlp, enn):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(300):
            m = model(x)
            loss = tb.belief_nll(m, y, Fa) + 0.05 * tb.nonspecificity_penalty(m, Fa)
            opt.zero_grad(); loss.backward(); opt.step()
    g = torch.linspace(-12, 12, 161)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.flatten(), gy.flatten()], -1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, model, title in [(axes[0], mlp, "Dirichlet head on an MLP"),
                             (axes[1], enn, "PrototypeHead (Denœux's ENN)")]:
        with torch.no_grad():
            theta = model(grid)[:, -1].reshape(161, 161).numpy()
        im = ax.imshow(theta, extent=(-12, 12, -12, 12), origin="lower", cmap=cmap, vmin=0, vmax=1)
        for k, mk in enumerate("os^"):
            sel = (y == k).numpy()
            ax.plot(x[sel, 0], x[sel, 1], mk, ms=3.2, mfc=t["surface"], mec=t["ink"], mew=0.6, alpha=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title)
    axes[0].text(-11.3, 10.4, "confident everywhere,\neven where it has never seen data",
                 fontsize=9.5, color=t["ink"], va="top",
                 bbox=dict(fc=t["surface"], ec="none", alpha=0.85, boxstyle="round,pad=0.3"))
    axes[1].text(-11.3, 10.4, "says “I don't know” (m(Θ) → 1)\naway from the training data",
                 fontsize=9.5, color="#ffffff" if t["surface"] == "#fcfcfb" else "#0b0b0b", va="top")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("m(Θ): how much the model says “I don't know”", color=t["ink2"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t["muted"], labelcolor=t["ink2"])
    return fig


# ---- 7. p-boxes --------------------------------------------------------------------------------------------

def fig_pbox(t):
    lo = torch.tensor([18., 20., 15.], dtype=D)
    hi = torch.tensor([22., 25., 30.], dtype=D)
    w = torch.tensor([0.4, 0.35, 0.25], dtype=D)
    ri = RandomInterval(lo, hi, w)
    xs = torch.linspace(12, 33, 600, dtype=D)
    Fl, Fu = ri.cdf_bounds(xs)
    mlo, mhi = ri.mean_bounds()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 2.2], hspace=0.35))
    labels = ["expert 1: 18–22 °C", "expert 2: 20–25 °C", "old sensor: 15–30 °C"]
    for i in range(3):
        a1.plot([lo[i], hi[i]], [2 - i, 2 - i], color=t["series"][0], lw=4 + 10 * float(w[i]),
                solid_capstyle="butt")
        a1.text(float(lo[i]), 2 - i + 0.3, f"{labels[i]}   (weight {w[i]:.2f})", va="bottom",
                fontsize=9, color=t["ink2"])
    a1.set_yticks([]); a1.spines["left"].set_visible(False); a1.set_ylim(-0.3, 3.0)
    a1.set_title("Evidence: three interval statements about tomorrow's temperature")
    a2.fill_between(xs, Fl, Fu, color=t["band"], lw=0, step=None)
    a2.plot(xs, Fu, color=t["series"][0], lw=2, label="upper CDF (plausibility)")
    a2.plot(xs, Fl, color=t["series"][0], lw=2, ls=(0, (4, 2)), label="lower CDF (belief)")
    q = 19.0
    bel, pl = ri.prob_bounds(-100., q)
    a2.annotate(f"P(T ≤ {q:.0f} °C) is somewhere\nin [{bel:.2f}, {pl:.2f}]",
                xy=(q, float((bel + pl) / 2)), xytext=(25.8, 0.38), fontsize=9.5, color=t["ink"],
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    a2.plot([q, q], [float(bel), float(pl)], color=t["ink"], lw=2.5)
    a2.set_ylim(-0.02, 1.05); a2.set_yticks([0, 0.5, 1])
    a2.grid(axis="y", color=t["grid"], lw=0.8); a2.set_axisbelow(True)
    a2.set_xlabel("temperature (°C)")
    a2.set_title(f"The resulting p-box: a band of possible CDFs (mean in [{mlo:.1f}, {mhi:.1f}] °C)", fontsize=11.5)
    a2.legend(loc="upper left")
    return fig


# ---- 8. Belnap ----------------------------------------------------------------------------------------------

def fig_belnap(t):
    names, src = sensors()
    pooled = belnap.pool(src)
    lidar = belnap.singletons(src[0])
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    quad = [((0.5, 0), "TRUE", "supported, not refuted"), ((0, 0.5), "FALSE", "refuted, not supported"),
            ((0.5, 0.5), "BOTH", "contradiction"), ((0, 0), "NEITHER", "no information")]
    for (x0, y0), lab, sub in quad:
        ax.add_patch(Rectangle((x0, y0), 0.5, 0.5, fc=t["box"] if lab in ("TRUE", "FALSE") else t["surface"],
                               ec=t["border"], lw=1))
        ax.text(x0 + 0.25, y0 + 0.1, lab, ha="center", va="bottom", fontsize=11, fontweight="bold",
                color=t["ink2"])
        ax.text(x0 + 0.25, y0 + 0.04, sub, ha="center", va="bottom", fontsize=8.5, color=t["muted"])
    ax.plot([0, 1], [1, 0], color=t["muted"], lw=1, ls=(0, (3, 3)),
            label="t + f = 1: one coherent source stays below")
    for i, c in enumerate(CLASSES):
        lx, ly = lidar[i].tolist(); px, py = pooled[i].tolist()
        ax.annotate("", xy=(px, py), xytext=(lx, ly),
                    arrowprops=dict(arrowstyle="->", color=t["series"][i], lw=1.4, shrinkA=6, shrinkB=6))
        ax.plot(lx, ly, "o", ms=9, mfc=t["surface"], mec=t["series"][i], mew=2)
        ax.plot(px, py, "o", ms=10, mfc=t["series"][i], mec=t["surface"], mew=1.5)
        ax.text(px - 0.03, py - 0.035, f"“{c}”", fontsize=10, color=t["ink"], fontweight="bold",
                ha="right", va="top")
    ax.plot([], [], "o", ms=9, mfc=t["surface"], mec=t["ink2"], mew=2, label="lidar alone")
    ax.plot([], [], "o", ms=10, mfc=t["ink2"], mec=t["surface"], label="all four sensors pooled")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=9.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("t: support for “it is X”"); ax.set_ylabel("f: support for “it is not X”")
    ax.set_title("Belnap view: contradictions are shown, not averaged away")
    return fig


# ---- 9. the pipeline ------------------------------------------------------------------------------------------

def fig_pipeline(t):
    fig, ax = plt.subplots(figsize=(12, 3.1))
    steps = [
        ("1 · Describe", "probability\nbelief function\npossibility\ninterval / p-box\nneural network"),
        ("2 · Convert", "Capacity\none [lower, upper]\nper event"),
        ("3 · Check", "disagreement\ninconsistency measures\nShapley blame"),
        ("4 · Fuse", "Dempster · Yager\nDubois–Prade · PCR6\ncautious · discount"),
        ("5 · Decide", "best act\nor abstain (−1)\nwhen unsure"),
    ]
    W, gap = 1.9, 0.45
    for i, (head, body) in enumerate(steps):
        x = i * (W + gap)
        hl = i == 1
        ax.add_patch(FancyBboxPatch((x, 0), W, 2.2, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=t["series"][0] if hl else t["box"],
                                    ec=t["series"][0] if hl else t["border"], lw=1.2))
        ink = "#ffffff" if hl else t["ink"]
        ax.text(x + 0.15, 1.95, head, fontsize=11.5, fontweight="bold", color=ink, va="top")
        ax.text(x + 0.15, 1.45, body, fontsize=9.5, color="#ffffff" if hl else t["ink2"], va="top",
                linespacing=1.45)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + W + gap - 0.04, 1.1), xytext=(x + W + 0.04, 1.1),
                        arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.5))
    ax.text(0, -0.3, "every step is a batched, differentiable PyTorch op: the whole chain can sit inside "
            "a network and be trained end to end", fontsize=9.5, color=t["muted"], va="top")
    ax.set_xlim(-0.05, 5 * W + 4 * gap + 0.05); ax.set_ylim(-0.6, 2.3); ax.axis("off")
    return fig


# ---- 10. tracking ambiguous objects in real video ------------------------------------------------------------
#
# Replays the YOLO detection log saved by docs/make_tracking_demo.py; no detector needed here.

TRACKS = [(26, "Suzuki SX4 hatchback", "a car"),
          (21, "Citroën Berlingo van", "honestly either"),
          (3, "Maxus pickup", "a truck, by COCO's convention")]


def _strip(ax, y, labels, t, h=0.38):
    """One row of coloured runs; a two-class set is drawn as two thin stacked bands."""
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    for a, b, lab in runs:
        cols = [t["neutral"]] if lab == "?" else [t["series"][tracking.CLASSES.index(c)] for c in lab.split(" | ")]
        k = len(cols)
        for j, c in enumerate(cols):
            ax.add_patch(Rectangle((a, y - h / 2 + j * h / k), b - a, h / k - (0.03 if k > 1 else 0),
                                   color=c, lw=0))
        if b - a >= 14:
            txt = "not sure yet" if lab == "?" else lab.replace(" | ", " or ")
            ax.text((a + b) / 2, y, txt, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color=t["ink"] if lab == "?" else "#ffffff")


def fig_tracking(t):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "data", "osorno-yolo11n.json")) as fh:
        tracks = tracking.replay(json.load(fh))
    fig = plt.figure(figsize=(11.5, 8.4))
    outer = fig.add_gridspec(3, 2, width_ratios=[1, 5.2], hspace=0.55, wspace=0.36, top=0.94, bottom=0.12)
    flips = lambda xs: sum(a != b for a, b in zip(xs, xs[1:]))
    for r, (tid, name, truth) in enumerate(TRACKS):
        rows = tracks[tid]
        n = len(rows)
        yolo, ours = [x["yolo"] for x in rows], [x["label"] for x in rows]
        thumb = fig.add_subplot(outer[r, 0])
        thumb.imshow(plt.imread(os.path.join(here, "figures", "tracking", f"{tid}.jpg")))
        thumb.axis("off")
        gs = outer[r, 1].subgridspec(3, 1, height_ratios=[1.25, 1, 1], hspace=0.14)
        top = fig.add_subplot(gs[0])
        _strip(top, 1, yolo, t)
        _strip(top, 0, ours, t)
        top.set_yticks([1, 0], ["YOLO, per frame", "torchbelief"])
        top.tick_params(axis="y", length=0); top.tick_params(axis="x", length=0, labelbottom=False)
        for sp in top.spines.values():
            sp.set_visible(False)
        top.set_ylim(-0.45, 1.45); top.set_xlim(0, n)
        titled(top, f"{name} (track {tid}): {truth}",
               f"YOLO switches label {flips(yolo)} times; torchbelief {flips(ours)} times", t, size=11.5)
        x = torch.arange(n).numpy() + 0.5
        for c in range(2):
            ax = fig.add_subplot(gs[c + 1], sharex=top)
            col = t["series"][c]
            bel = torch.tensor([v["bel"][c] for v in rows]).numpy()
            pl = torch.tensor([v["pl"][c] for v in rows]).numpy()
            ax.fill_between(x, bel, pl, color=col, alpha=0.25, lw=0, step="mid")
            ax.step(x, bel, where="mid", color=col, lw=1.8)
            ax.set_ylim(0, 1.05); ax.set_yticks([0, 1], ["", "1"])
            ax.set_ylabel(tracking.CLASSES[c], rotation=0, ha="right", va="center", labelpad=8, fontsize=10)
            ax.grid(axis="y", color=t["grid"], lw=0.8); ax.set_axisbelow(True)
            if c == 1:
                ticks = list(range(0, n, 30))
                ax.set_xticks(ticks, [f"{k // 30}" for k in ticks])
                if r == 2:
                    ax.set_xlabel("seconds since the track started (30 frames per second)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
    handles = [Rectangle((0, 0), 1, 1, color=t["series"][0]), Rectangle((0, 0), 1, 1, color=t["series"][1]),
               Rectangle((0, 0), 1, 1, color=t["neutral"]),
               plt.Line2D([], [], color=t["muted"], lw=2), Rectangle((0, 0), 1, 1, color=t["muted"], alpha=0.3)]
    fig.legend(handles, ["car", "truck", "not sure yet", "Bel: guaranteed", "up to Pl: still possible"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=9.5, handlelength=1.6)
    return fig


# ---- 11. a team of LLM agents ----------------------------------------------------------------------

def _intervals(ax, rows, t, hi=None):
    """Bel / Pl bars, one per (label, bel, pl) row, top to bottom."""
    n = len(rows)
    for k, (lab, bel, pl) in enumerate(rows):
        y = n - 1 - k
        ax.add_patch(Rectangle((0, y - 0.18), 1, 0.36, color=t["grid"], lw=0))
        ax.add_patch(Rectangle((0, y - 0.18), bel, 0.36, color=t["series"][0], lw=0))
        ax.add_patch(Rectangle((bel, y - 0.18), pl - bel, 0.36, color=t["band"], lw=0))
        ax.text(1.03, y, f"[{bel:.2f}, {pl:.2f}]", va="center", fontsize=9, color=t["ink2"])
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n - 1, -1, -1), [r[0] for r in rows])
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def fig_agents(t):
    """Real replies of Qwen2.5-1.5B-Instruct as five agents (examples/multi_agent.py --local),
    scored at temperature 5."""
    with open(os.path.join(os.path.dirname(__file__), "data", "multi-agent-qwen2.5-1.5b.json")) as fh:
        replies = json.load(fh)
    T = 5.0
    fr = tb.Frame(len(multi_agent.CAUSES), names=multi_agent.CAUSES)
    names = list(replies["round1"])
    m1 = [multi_agent.to_mass(replies["round1"][a], fr, T) for a in names]
    phi = I.shapley_blame([tb.Capacity.from_mass(m, fr) for m in m1]).numpy()

    def deploy(m):
        bel, pl = tb.interval(m, fr)
        return float(bel[0]), float(pl[0])

    team_d, _ = multi_agent.fuse(m1, fr, "dempster")
    team_c, alpha = multi_agent.fuse(m1, fr, "cautious")
    m2 = [tb.discount(multi_agent.to_mass(replies["round2"][a], fr, T), alpha[i], fr) for i, a in enumerate(names)]
    debate = {r: tb.combine([team_c] + m2, fr, r)[0] for r in ("dempster", "cautious")}
    early, _ = multi_agent.fuse([m1[names.index("logs")], m1[names.index("history")]], fr, "cautious")
    eb, ep = tb.interval(early, fr)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    y = list(range(len(names) - 1, -1, -1))
    worst = int(phi.argmax())
    a1.barh(y, phi, height=0.55, color=[t["series"][1] if i == worst else t["series"][0] for i in range(len(names))])
    for yy, v in zip(y, phi):
        a1.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=9.5, color=t["ink2"])
    a1.set_yticks(y, names); a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    a1.set_xlim(0, 0.75); a1.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"])
    a1.grid(axis="x", color=t["grid"], lw=0.8); a1.set_axisbelow(True)
    titled(a1, "1 · Who contradicts the team?", "Shapley blame; history matched an old\nDNS incident", t)

    _intervals(a2, [("round 1, Dempster", *deploy(team_d)), ("round 1, cautious", *deploy(team_c)),
                    ("+ debate, Dempster", *deploy(debate["dempster"])),
                    ("+ debate, cautious", *deploy(debate["cautious"]))], t)
    titled(a2, "2 · How sure is the team of “deploy”?",
           "five prompts on one model are not independent;\nDempster multiplies them up to certainty", t)

    _intervals(a3, [(c, float(eb[i]), float(ep[i])) for i, c in enumerate(fr.names)], t)
    titled(a3, "3 · Early: only logs and history replied",
           f"no cause beats the others: {multi_agent.label(early, fr).replace(' | ', ', ')}\n→ wait or escalate", t)

    handles = [Rectangle((0, 0), 1, 1, color=t["series"][0]), Rectangle((0, 0), 1, 1, color=t["band"])]
    fig.legend(handles, ["Bel: guaranteed", "up to Pl: still possible"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.62, -0.08), fontsize=9.5)
    fig.subplots_adjust(wspace=0.75)
    return fig


# ---- 12. corrections ------------------------------------------------------------------------------

def fig_corrections(t):
    m = cap_mass(car=0.6, car_truck=0.2, all=0.2)
    weak = torch.tensor([1.0, 0.3, 1.0], dtype=D)
    panels = [
        ("As reported", "car 60 %, car or truck 20 %,\ndon't know 20 %", m),
        ("Discounted", "unreliable when the truth is truck:\ntruck can't be ruled out", tb.contextual_discount(m, weak, F)),
        ("Reinforced", "too ready to call things trucks:\ntruck is less plausible", tb.contextual_reinforce(m, weak, F)),
        ("Negated", "says the opposite about bikes\n70 % of the time",
         tb.contextual_negate(m, torch.tensor([1.0, 1.0, 0.3], dtype=D), F)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), sharey=True)
    for ax, (title, sub, mm) in zip(axes, panels):
        bel, pl = tb.interval(mm, F)
        for i in range(3):
            y = 2 - i
            ax.add_patch(Rectangle((0, y - 0.16), 1, 0.32, color=t["grid"], lw=0))
            ax.add_patch(Rectangle((0, y - 0.16), float(bel[i]), 0.32, color=t["series"][0], lw=0))
            ax.add_patch(Rectangle((float(bel[i]), y - 0.16), float(pl[i] - bel[i]), 0.32,
                                   color=t["band"], lw=0))
            if mm is not m:                                   # the uncorrected Pl, for comparison
                ax.plot([float(tb.interval(m, F)[1][i])] * 2, [y - 0.24, y + 0.24], color=t["ink"], lw=1.2)
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.6)
        ax.set_yticks([2, 1, 0], CLASSES)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        titled(ax, title, sub, t)
    handles = [Rectangle((0, 0), 1, 1, color=t["series"][0]), Rectangle((0, 0), 1, 1, color=t["band"]),
               plt.Line2D([], [], color=t["ink"], lw=1.2)]
    fig.legend(handles, ["Bel", "Pl − Bel", "Pl as reported"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.12)
    return fig


FIGURES = {"pipeline": fig_pipeline, "intervals": fig_intervals, "powerset": fig_powerset,
           "rules": fig_rules, "blame": fig_blame, "abstain": fig_abstain, "ood": fig_ood,
           "pbox": fig_pbox, "belnap": fig_belnap, "tracking": fig_tracking, "agents": fig_agents,
           "corrections": fig_corrections}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:] or list(FIGURES)
    for theme, t in THEMES.items():
        style(t)
        for name in only:
            save(FIGURES[name](t), name, theme)
            print("wrote", f"{name}-{theme}.svg")
