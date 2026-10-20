"""Belief tracking: honest class labels for YOLO tracks.

On real video a detector flickers. A pickup is "car" for half a second, then "truck", then
"car" again. A hatchback is called "truck" for twenty frames in a row. The tracker keeps the
box, but the label shown is whatever the latest frame said, with a confident-looking score.

This example puts a belief function over the classes on every track. Each frame:

    m ← discount(m, 0.99)                                 forget a little: stale evidence fades
    m ← dubois_prade(m, simple_support(cls, 0.1 · conf))  add the detection; conflict goes to "A or B"

The 0.1 says consecutive frames of one network are far from independent: ten agreeing frames
are worth about one independent look. Dubois–Prade sends the mass that two frames dispute to
the union of what they said. When the detector alternates, the track ends up believing "car
or truck" and stays there until one side keeps winning.

The label shown is the set of classes no other class beats (Walley maximality): "truck",
"car | truck", or "?" when nothing can be ruled out yet.

    pip install ultralytics                               # optional; torchbelief does not need it
    python examples/tracking.py clip.mp4 --out tracked.mp4 --log detections.json
    python examples/tracking.py --replay detections.json  # belief layer only, no YOLO needed

ultralytics is AGPL-3.0. It is used here only as an optional detector; any tracker that gives
(track id, class, confidence) per frame works the same way.
"""

import argparse
import json
from collections import defaultdict

import torch

import torchbelief as tb
from torchbelief import choice

D = torch.float64
CLASSES = ["car", "truck", "bus"]


class BeliefTrack:
    """Class belief of one tracked object, updated from per-frame detections."""

    def __init__(self, frame, forget=0.99, weight=0.1):
        self.frame, self.forget, self.weight = frame, forget, weight
        self.m = tb.vacuous(frame, dtype=D)
        self.util = torch.eye(frame.n, dtype=D)

    def update(self, cls=None, conf=0.0):
        """One frame: a detection (class index, confidence), or ``None`` if the track was not
        matched. Returns the conflict between the detection and what the track believed."""
        self.m = tb.discount(self.m, self.forget, self.frame)
        if cls is None:
            return 0.0
        det = tb.simple_support(self.frame, 1 << cls, self.weight * conf, dtype=D)
        k = tb.conflict(self.m, det, self.frame)
        self.m, _ = tb.dubois_prade(self.m, det, self.frame)
        return float(k)

    def bounds(self):
        """(Bel, Pl) per class: what the evidence guarantees, and what it still allows."""
        return tb.interval(self.m, self.frame)

    def keep(self):
        """Bool per class: not beaten by any other class (Walley maximality)."""
        return choice.maximal_acts(tb.Capacity.from_mass(self.m, self.frame), self.util)

    def label(self):
        keep = self.keep()
        if bool(keep.all()):
            return "?"
        return " | ".join(n for n, k in zip(self.frame.names, keep) if k)


def replay(log, classes=CLASSES, **kw):
    """Run the belief layer over a detection log (a list of {f, id, cls, conf, box}).

    Returns {track id: per-frame dicts with the detection and the track's state}."""
    frame = tb.Frame(len(classes), names=classes)
    by_frame = defaultdict(dict)
    for r in log:
        if r["id"] is not None and r["cls"] in classes:
            by_frame[r["f"]][r["id"]] = r
    tracks, out = {}, defaultdict(list)
    for f in range(max(by_frame) + 1):
        seen = by_frame[f]
        for tid in sorted(set(tracks) | set(seen)):
            bt = tracks.setdefault(tid, BeliefTrack(frame, **kw))
            r = seen.get(tid)
            bt.update(classes.index(r["cls"]), r["conf"]) if r else bt.update()
            if r:
                bel, pl = bt.bounds()
                out[tid].append(dict(f=f, id=tid, yolo=r["cls"], conf=r["conf"], box=r["box"], label=bt.label(),
                                     bel=bel.tolist(), pl=pl.tolist()))
    return dict(out)


# ---- drawing ----------------------------------------------------------------------------------

COLOURS = {"car": (57, 135, 229), "truck": (235, 104, 52), "bus": (27, 175, 122)}   # RGB, as in the docs


def _font(size, bold=False):
    from PIL import ImageFont
    names = ["Helvetica.ttc", "Arial Bold.ttf" if bold else "Arial.ttf",
             "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        for path in ("/System/Library/Fonts/" + name, "/Library/Fonts/" + name,
                     "/usr/share/fonts/truetype/dejavu/" + name, name):
            try:
                return ImageFont.truetype(path, size, index=1 if bold and name.endswith(".ttc") else 0)
            except OSError:
                continue
    return ImageFont.load_default(size)


def draw(img, items, classes=CLASSES, cards=5):
    """Annotate one frame. Every track gets a box and an id tag on the video; the ``cards``
    largest also get a card in a side panel: YOLO's label this frame, torchbelief's label, and
    the [Bel, Pl] bar of each class (solid = Bel, light = up to Pl).

    ``img`` is a PIL RGB image; ``items`` are per-track dicts as returned by :func:`replay`.
    Returns a new, wider image."""
    from PIL import Image, ImageDraw
    s = img.height / 450
    panel = int(250 * s)
    out = Image.new("RGB", (img.width + panel, img.height), (24, 24, 23))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out, "RGBA")
    f_id, f_small, f_big, f_tiny = _font(int(12 * s), True), _font(int(12 * s)), _font(int(16 * s), True), _font(int(10 * s))
    grey, ink = (165, 164, 158), (240, 239, 236)

    def colour(lab):
        return COLOURS[lab] if lab in classes else ink

    for it in items:
        x1, y1, x2, y2 = it["box"]
        d.rounded_rectangle((x1, y1, x2, y2), radius=int(5 * s), outline=colour(it["label"]) + (255,),
                            width=max(2, int(2.5 * s)))
        tag = f"#{it['id']}"
        tw = d.textlength(tag, font=f_id)
        d.rounded_rectangle((x1, y1, x1 + tw + 10 * s, y1 + 18 * s), radius=int(4 * s), fill=(17, 17, 17, 220))
        d.text((x1 + 5 * s, y1 + 2 * s), tag, font=f_id, fill=colour(it["label"]))

    def area(it):
        x1, y1, x2, y2 = it["box"]
        return (x2 - x1) * (y2 - y1)
    shown = sorted(sorted(items, key=area, reverse=True)[:cards], key=lambda it: it["box"][0] + it["box"][2])
    ch = (img.height - 8 * s) / cards
    for i, it in enumerate(shown):
        x, y = img.width + 10 * s, 6 * s + i * ch
        d.rounded_rectangle((x, y, out.width - 8 * s, y + ch - 6 * s), radius=int(7 * s), fill=(36, 36, 34))
        lab = it["label"]
        d.text((x + 9 * s, y + 7 * s), f"#{it['id']}", font=f_id, fill=colour(lab))
        d.text((x + 46 * s, y + 7 * s), f"YOLO: {it['yolo']} {it['conf']:.2f}", font=f_small, fill=grey)
        tx, ty = x + 9 * s, y + 24 * s
        if lab == "?":
            d.text((tx, ty), "? not sure yet", font=f_big, fill=ink)
        else:
            for j, name in enumerate(lab.split(" | ")):
                if j:
                    d.text((tx, ty + 3 * s), "  or  ", font=f_small, fill=grey)
                    tx += d.textlength("  or  ", font=f_small)
                d.text((tx, ty), name, font=f_big, fill=COLOURS[name])
                tx += d.textlength(name, font=f_big)
        for c, name in enumerate(classes):
            by = y + 47 * s + c * 11 * s
            bx, bw, bh = x + 46 * s, out.width - x - 64 * s, 7 * s
            d.text((x + 9 * s, by - 3 * s), name, font=f_tiny, fill=grey)
            d.rectangle((bx, by, bx + bw, by + bh), fill=(58, 58, 55))
            d.rectangle((bx, by, bx + bw * it["pl"][c], by + bh), fill=COLOURS[name] + (95,))
            d.rectangle((bx, by, bx + bw * it["bel"][c], by + bh), fill=COLOURS[name] + (255,))
    return out


# ---- running YOLO ----------------------------------------------------------------------------

def detect(video, weights="yolo11n.pt", conf=0.1):
    """YOLO + ByteTrack via ultralytics. Returns the detection log and the frames (BGR)."""
    from ultralytics import YOLO
    log, frames = [], []
    for f, r in enumerate(YOLO(weights).track(video, stream=True, persist=True, verbose=False, conf=conf)):
        frames.append(r.orig_img)
        b = r.boxes
        ids = b.id.int().tolist() if b.id is not None else [None] * len(b)
        for box, tid, c, p in zip(b.xyxy.tolist(), ids, b.cls.int().tolist(), b.conf.tolist()):
            log.append(dict(f=f, id=tid, cls=r.names[c], conf=round(p, 3), box=[round(v, 1) for v in box]))
    return log, frames


def summary(log):
    flips = lambda xs: sum(x != y for x, y in zip(xs, xs[1:]))
    for tid, rows in sorted(replay(log).items(), key=lambda kv: -len(kv[1])):
        if len(rows) < 30:
            continue
        yolo, ours = [r["yolo"] for r in rows], [r["label"] for r in rows]
        count = lambda xs: ", ".join(f"{c} {xs.count(c)}" for c in sorted(set(xs), key=xs.count, reverse=True))
        print(f"track {tid:3d}  {len(rows):3d} frames  YOLO {flips(yolo):2d} switches ({count(yolo)})"
              f"   torchbelief {flips(ours):2d} switches ({count(ours)})")


def main():
    ap = argparse.ArgumentParser(description="torchbelief labels on YOLO tracks")
    ap.add_argument("video", nargs="?", help="video file or stream URL")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--out", default="tracked.mp4", help="annotated video to write")
    ap.add_argument("--log", help="also save the raw detection log (JSON) here")
    ap.add_argument("--replay", help="skip YOLO: run the belief layer over a saved log")
    a = ap.parse_args()
    if a.replay:
        return summary(json.load(open(a.replay)))
    if not a.video:
        ap.error("give a video, or --replay a saved log")

    import cv2
    import numpy as np
    from PIL import Image
    log, frames = detect(a.video, a.weights)
    if a.log:
        json.dump(log, open(a.log, "w"))
    per_frame = defaultdict(list)
    for rows in replay(log).values():
        for r in rows:
            per_frame[r["f"]].append(r)
    out = None
    for f, bgr in enumerate(frames):
        img = draw(Image.fromarray(bgr[..., ::-1].copy()), per_frame[f])
        if out is None:
            out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, img.size)
        out.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    out.release()
    summary(log)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
