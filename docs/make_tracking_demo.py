"""Regenerate the tracking demo from the source clip (needs ultralytics and ffmpeg).

    pip install ultralytics
    python docs/make_tracking_demo.py osorno.mp4

The clip is "Rainy City Traffic at Osorno Intersection" by Videas Cl on Pexels
(https://www.pexels.com/video/32069050/, Pexels license), downloaded at 2560×1440. This script
scales it to 1280×720, runs YOLO11n + ByteTrack, and writes

    docs/data/osorno-yolo11n.json         the raw detection log (what docs/make_figures.py replays)
    docs/figures/tracking-demo.webp       the annotated animation in the README and the visual guide
    docs/figures/tracking/<id>.jpg        one thumbnail per track shown in the timeline figure

The belief layer itself is examples/tracking.py; nothing here is specific to this clip
except the segment and the tracks picked for the figure.
"""

import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples")]
import tracking  # noqa: E402

LOG = os.path.join(HERE, "data", "osorno-yolo11n.json")
ANIM = os.path.join(HERE, "figures", "tracking-demo.webp")
THUMBS = os.path.join(HERE, "figures", "tracking")
SEGMENT = (375, 645, 2)     # first frame, last frame, step: the van, the hatchback and the pickups
WIDTH = 800
QUALITY = 60
SHOWN = {26: 563, 21: 437, 3: 253}       # tracks in the timeline figure → frame for the thumbnail


def main(clip):
    tmp = tempfile.mkdtemp()
    small = os.path.join(tmp, "clip720.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", "scale=1280:720", "-an",
                    "-c:v", "libx264", "-crf", "20", small], check=True)
    log, frames = tracking.detect(small, "yolo11n.pt")
    log = [r for r in log if r["id"] is not None and r["cls"] in tracking.CLASSES]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as fh:
        json.dump(log, fh, separators=(",", ":"))
    print("wrote", LOG, len(log), "detections")

    tracks = tracking.replay(log)
    per_frame = {}
    for rows in tracks.values():
        for r in rows:
            per_frame.setdefault(r["f"], []).append(r)

    os.makedirs(THUMBS, exist_ok=True)
    for tid, f in SHOWN.items():
        r = min(tracks[tid], key=lambda r: abs(r["f"] - f))
        f = r["f"]
        x1, y1, x2, y2 = map(int, r["box"])
        crop = frames[f][max(0, y1):y2, max(0, x1):x2]
        h = 160
        cv2.imwrite(os.path.join(THUMBS, f"{tid}.jpg"),
                    cv2.resize(crop, (int(crop.shape[1] * h / crop.shape[0]), h)), [cv2.IMWRITE_JPEG_QUALITY, 85])

    k = WIDTH / 1280
    head = 30
    a, b, step = SEGMENT
    out = []
    for f in range(a, b + 1, step):
        img = Image.fromarray(cv2.resize(frames[f], (WIDTH, int(720 * k)))[..., ::-1].copy())
        items = [dict(r, box=[v * k for v in r["box"]]) for r in per_frame.get(f, [])]
        img = tracking.draw(img, items)
        canvas = Image.new("RGB", (img.width, img.height + head), (17, 17, 17))
        canvas.paste(img, (0, head))
        d = ImageDraw.Draw(canvas)
        d.text((12, 8), "YOLO11n + ByteTrack on real traffic. Right: what YOLO says this frame, what "
               "torchbelief says, and its [Bel, Pl] per class.", font=tracking._font(13), fill=(225, 224, 220))
        out.append(canvas)
    # Animated WebP: a GIF of real footage is ~25 MB, this is a few MB and GitHub renders it.
    out[0].save(ANIM, save_all=True, append_images=out[1:], duration=1000 * step // 30, loop=0,
                quality=QUALITY, method=6)
    print("wrote", ANIM, f"{os.path.getsize(ANIM) / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
