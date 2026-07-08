# ============================================================================
# NOTE: This is the YOLO (neural network) DETECTOR variant — the original
# approach for the "eye". The make/miss logic (the "brain") is identical to
# the production system; only the ball detection differs. The final app uses
# the deterministic blob detector instead (see blob_detect.py and the README
# design decisions: YOLO lost the ball near the rim, was slower, and was not
# reproducible across hardware). Kept for reference and comparison.
#
# Running this file additionally requires: ultralytics (pip install
# ultralytics) and the BallDetector/detect_rim from the original project's
# score.py, which is not included in this repository.
# ============================================================================
"""
score_simple.py — Alternate Make/Miss Logic
===============================================================
Designed for ONE consistent camera angle: behind the shooter, facing the
basket square-on, like a free-throw shooter's view (a bit further back).

The logic, exactly as specified:
  1. Draw a box around the basket. Mark the y of the rim's TOP.
  2. Whenever the ball's y goes ABOVE the rim top => a shot attempt.
  3. The NEXT time the ball appears BELOW the rim's BOTTOM:
       - x inside the box's x-range  => MAKE
       - x outside                   => MISS

This works because from straight-on the rim's x-range IS the target, and every real
shot must rise above the rim line. The trade-off: it assumes this angle —
from the side or at depth this logic breaks

Usage:
    python score_simple.py clips/mine/lucas_clip2.mp4
        [--out simple_scored.mp4] [--json simple_shots.json]
        [--rim x1,y1,x2,y2]      # the box around the basket (recommended:
                                 #   in the app this is the user's one-time
                                 #   "drag a box around the rim" calibration)
        [--debug-rim rim.jpg]    # check what auto-detection picked
        [--roi-only]             # only run YOLO near the basket (faster)

Reuses the hybrid ball detector and rim auto-detection from score.py.
"""

import os
import sys
import json
import argparse
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import BallDetector, detect_rim   # noqa: E402

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
X_MARGIN = 0.05       # make-window shrink: ball center must be inside the
                      # box x-range minus 5% per side (validated: real makes
                      # return at 664-701, real misses at 559-870 — the
                      # margin only needs to absorb calibration error)
NEAR_X = 5.0          # ball must be within N*box_width of box center (in x)
                      # for an above-rim event to arm — filters passes lofted
                      # elsewhere in the gym
NET_LENGTH = 1.4      # net hangs ~1.4 rim-widths below the rim ring; the
                      # box's bottom edge (the verdict line) sits there, so
                      # the make/miss check happens after the ball clears
                      # the net, not the instant it dips past the iron
ARM_SLACK = 0.6       # arm even if the ball's center is up to 0.6 ball-
                      # diameters BELOW the rim top (deep shots arrive almost
                      # flat in image space; center may never clear the line)
DEPTH_MISS = 1.45     # verdict-crossing ball wider than 1.45x expected ball
                      # size is closer to the camera than the hoop => a
                      # rebound flying toward the camera, not a make
MIN_FLIGHT = 5        # verdict can't fire sooner than 5 frames after arming
COOLDOWN = 30         # frames after a verdict before re-arming
BELOW_TIMEOUT = 90    # if the ball never reappears below the net within 3s,
                      # log the attempt as UNRESOLVED (usually stuck ball or
                      # detection loss — better honest than guessed)

ORANGE = (0, 130, 255)
CYAN = (255, 220, 0)
GREEN = (0, 220, 80)
RED = (0, 60, 220)
WHITE = (255, 255, 255)


class SimpleShotCounter:
    """Two states. IDLE: watch for the ball above the rim top.
    AIRBORNE: watch for the ball below the rim bottom, judge by x."""

    def __init__(self, rim_box):
        self.x1, self.y_top, self.x2, self.y_bot = rim_box
        self.box_w = self.x2 - self.x1
        self.ball_d = self.box_w * 0.55
        self.cx = (self.x1 + self.x2) / 2
        self.state = "IDLE"
        self.armed_frame = 0
        self.cooldown_until = -1
        self.net_hits = 0        # detections inside the net zone this attempt
        self.prev = None         # (frame, x, y) of last detection, any state
        self.shots = []

    # (de)serialization for chunked runs
    def to_dict(self):
        return {"state": self.state, "armed_frame": self.armed_frame,
                "cooldown_until": self.cooldown_until,
                "net_hits": self.net_hits, "prev": self.prev,
                "shots": self.shots}

    def from_dict(self, d):
        self.state = d["state"]
        self.armed_frame = d["armed_frame"]
        self.cooldown_until = d["cooldown_until"]
        self.net_hits = d.get("net_hits", 0)
        self.prev = tuple(d["prev"]) if d.get("prev") else None
        self.shots = d["shots"]

    def update(self, frame_num, pos, ball_w=None):
        if self.state == "AIRBORNE" and \
                frame_num - self.armed_frame > BELOW_TIMEOUT:
            self.state = "IDLE"
            self.cooldown_until = frame_num + COOLDOWN
            shot = {"frame": frame_num, "verdict": "UNRESOLVED",
                    "x_at_return": None}
            self.shots.append(shot)
            return shot

        if pos is None:
            return None
        x, y = pos

        if self.state == "IDLE":
            # Rule 1 (loosened): ball near/above the rim top — deep shots
            # arrive nearly flat in image space, so allow the center to be
            # slightly below the line. Also arm if the ball is detected
            # inside the box itself while moving DOWN (arrived at rim level
            # without ever crossing the line in view).
            near = abs(x - self.cx) < NEAR_X * self.box_w
            moving_down = (self.prev is None or
                           (frame_num - self.prev[0] <= 4
                            and y >= self.prev[2]))
            # The two LOOSENED arming paths (slack band, box entry) require
            # the ball to be plausibly AT the rim's depth (right apparent
            # size) — a dribbled ball far away passes through these image
            # regions in 2D but reads much smaller. The original strict
            # line (validated on 100+ shots) stays unconditional.
            size_ok = (ball_w is not None
                       and 0.85 * self.ball_d <= ball_w <= 1.5 * self.ball_d)
            strict = y < self.y_top
            slack_band = y < self.y_top + ARM_SLACK * self.ball_d
            in_box = (self.x1 <= x <= self.x2
                      and self.y_top <= y <= self.y_bot)
            if frame_num > self.cooldown_until and near and \
                    (strict or (slack_band and size_ok)
                     or (in_box and moving_down and size_ok)):
                self.state = "AIRBORNE"
                self.armed_frame = frame_num
                self.net_hits = 0
        elif self.state == "AIRBORNE":
            # count detections inside the NET ZONE (in the box, below the
            # ring area). A ball tracked in there was in the net — direct
            # make evidence that survives the net kicking the exit sideways.
            net_top = self.y_top + 0.25 * (self.y_bot - self.y_top)
            if self.x1 <= x <= self.x2 and net_top <= y <= self.y_bot:
                self.net_hits += 1

            # Rule 2: first reappearance below the box bottom (= net bottom).
            # Guards: minimum flight time; return point near the basket; and
            # the triggering detection must actually be FALLING (a previous
            # detection a few frames back, visibly higher, x-consistent) —
            # a static false blob (e.g. a hand) can't resolve a shot.
            # the verdict detection must be an observed CROSSING of the
            # line: previous point AT/ABOVE the verdict line, current below.
            # A dribble or held ball (both points fully below the line) can
            # never resolve a shot, even if it moves downward.
            gap = frame_num - self.prev[0] if self.prev else 99
            falling = (self.prev is not None
                       and gap <= 4
                       and y - self.prev[2] >= 4
                       # physically plausible fall speed (no teleports)
                       and y - self.prev[2] <= 1.5 * self.box_w * gap
                       and self.prev[2] <= self.y_bot
                       and abs(x - self.prev[1]) <= 0.75 * self.box_w)
            # crossings only count in the net-exit band just below the line
            if self.y_bot < y <= self.y_bot + 2.0 * self.box_w \
                    and falling and \
                    frame_num - self.armed_frame >= MIN_FLIGHT and \
                    abs(x - self.cx) < NEAR_X * self.box_w:
                margin = self.box_w * X_MARGIN
                make = (self.x1 + margin) < x < (self.x2 - margin)
                # net-zone evidence overrides a slightly-kicked exit point
                if not make and self.net_hits >= 2 and \
                        abs(x - self.cx) <= self.box_w:
                    make = True
                # depth gate: a crossing ball much larger than rim-depth
                # size is a rebound flying TOWARD the camera (2D overlap
                # with the box, but feet in front of the hoop) => MISS
                depth_flag = bool(ball_w and ball_w > self.ball_d * DEPTH_MISS)
                if make and depth_flag:
                    make = False
                shot = {"frame": frame_num,
                        "verdict": "MAKE" if make else "MISS",
                        "x_at_return": round(x, 1),
                        "net_hits": self.net_hits,
                        "ball_w": ball_w,
                        "depth_flag": depth_flag}
                self.shots.append(shot)
                self.state = "IDLE"
                self.cooldown_until = frame_num + COOLDOWN
                self.prev = (frame_num, x, y)
                return shot
        self.prev = (frame_num, x, y)
        return None


def _pick_gated(cands, detector, frame_num):
    """Candidate selection with the same continuity gating as
    BallDetector.detect(): confidence, penalized by distance from the
    predicted ball position. Prevents verdict-triggering jumps to false
    blobs. Updates the detector's last/vel state."""
    if not cands:
        return None
    pred = None
    age = frame_num - detector.last_frame
    if detector.last is not None and age <= 12:
        pred = (detector.last[0] + detector.vel[0] * age,
                detector.last[1] + detector.vel[1] * age)

    def score(item):
        c, _ = item
        s = c[2]
        if pred is not None:
            d = np.hypot(c[0] - pred[0], c[1] - pred[1])
            s -= d / 400.0
            if d > detector.rim_w * 6:
                s -= 1.0
        return s

    best, source = max(cands, key=score)
    if source == "blob":
        if pred is None or np.hypot(best[0] - pred[0],
                                    best[1] - pred[1]) > detector.rim_w * 3:
            return None
    if detector.last is not None and 0 < age <= 12:
        detector.vel = ((best[0] - detector.last[0]) / age,
                        (best[1] - detector.last[1]) / age)
    detector.last = (best[0], best[1])
    detector.last_frame = frame_num
    return (*best, source)


def process(args):
    # ── rim box ──
    state = None
    if args.state and args.start > 0:
        try:
            with open(args.state) as f:
                state = json.load(f)
        except FileNotFoundError:
            pass

    if state:
        rim = tuple(state["rim"])
        box = tuple(state["box"])
    else:
        if args.box:
            # user-drawn box around the ENTIRE basket (rim top -> net bottom)
            box = tuple(int(v) for v in args.box.split(","))
            rim = (box[0], box[1], box[2],
                   box[1] + int((box[2] - box[0]) * 0.3))  # ring ~ top 30%
            print(f"Using manual basket box: {box}")
        else:
            if args.rim:
                rim = tuple(int(v) for v in args.rim.split(","))
                print(f"Using manual rim ring: {rim}")
            else:
                print("Auto-detecting rim (pass --box to override)...")
                rim = detect_rim(args.input, args.debug_rim)
                if rim is None:
                    print("ERROR: rim not found. Pass --box x1,y1,x2,y2")
                    sys.exit(1)
                print(f"Auto-detected rim ring: {rim}")
            # extend the ring down to the net bottom => the full basket box
            rim_w = rim[2] - rim[0]
            box = (rim[0], rim[1], rim[2],
                   rim[3] + int(rim_w * NET_LENGTH))
        print(f"Basket box (rim top -> net bottom): {box}")

    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = BallDetector(args.model, rim, (w, h), args.conf, args.imgsz)
    # widen the zoom/tracking region well beyond the default: from this
    # camera distance the rim is small, and we want the ball tracked through
    # most of its flight, not just at the hoop
    bcx = (box[0] + box[2]) // 2
    bcy = (box[1] + box[3]) // 2
    half = max(int((box[2] - box[0]) * 4), 300)
    detector.roi = (max(0, bcx - half), max(0, bcy - half),
                    min(w, bcx + half), min(h, bcy + half))
    counter = SimpleShotCounter(box)
    if state:
        counter.from_dict(state["counter"])
        if state.get("last_pos"):
            detector.last = tuple(state["last_pos"])
        detector.last_frame = state.get("det_last_frame", args.start)
        detector.vel = tuple(state.get("det_vel", (0.0, 0.0)))

    end = args.end if args.end > 0 else total
    out_path = args.out
    if args.start > 0 or end < total:
        out_path = f"{args.out}.part{args.start:06d}.mp4"
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    if args.start > 0:
        warm_from = max(0, args.start - 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_from)
        for _ in range(args.start - warm_from):
            ret, wf = cap.read()
            if not ret:
                break
            detector.bg.apply(wf[detector.roi[1]:detector.roi[3],
                                 detector.roi[0]:detector.roi[2]])

    trail = deque(maxlen=25)
    last_verdict, last_verdict_frame = None, -999
    frame_num = args.start

    print(f"Processing frames {args.start}..{end} of {total}")
    while frame_num < end:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if args.roi_only:
            # only look in the (enlarged) region around the basket — faster,
            # and with proper continuity gating (prefer the candidate nearest
            # the ball's predicted position, like the full detector does)
            rx1, ry1, rx2, ry2 = detector.roi
            crop = frame[ry1:ry2, rx1:rx2]
            fg = detector.bg.apply(crop)
            cands = [(c, "roi") for c in detector._yolo(crop, 512, rx1, ry1)]
            if not cands:
                cands = [(c, "blob") for c in
                         detector._blob_fallback(crop, fg, rx1, ry1)]
            det = _pick_gated(cands, detector, frame_num)
        else:
            det = detector.detect(frame, frame_num)

        pos = None
        if det:
            cx, cy, conf, bx1, by1, bx2, by2, source = det
            pos = (cx, cy)
            trail.append(pos)
            color = CYAN if source == "blob" else ORANGE
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        else:
            trail.append(None)

        verdict = counter.update(frame_num, pos,
                                 (bx2 - bx1) if det else None)
        if verdict:
            last_verdict, last_verdict_frame = verdict, frame_num
            print(f"  frame {frame_num}: {verdict['verdict']}"
                  f" (x_return={verdict['x_at_return']})")

        # ── draw ──
        pts = [p for p in trail if p is not None]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], ORANGE,
                     max(1, int(6 * i / len(pts))))
        # basket box + the two decision lines
        cv2.rectangle(frame, (counter.x1, counter.y_top),
                      (counter.x2, counter.y_bot), GREEN, 2)
        cv2.line(frame, (0, counter.y_top), (w, counter.y_top),
                 (200, 200, 0), 1)            # attempt line (rim top)
        cv2.line(frame, (0, counter.y_bot), (w, counter.y_bot),
                 (200, 200, 0), 1)            # verdict line (rim bottom)

        makes = sum(1 for s in counter.shots if s["verdict"] == "MAKE")
        n = sum(1 for s in counter.shots if s["verdict"] != "UNRESOLVED")
        hud = f"{makes}/{n} makes"
        if counter.state == "AIRBORNE":
            hud += "  [ball up]"
        cv2.putText(frame, hud, (9, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4)
        cv2.putText(frame, hud, (9, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    WHITE, 2)
        if last_verdict and frame_num - last_verdict_frame < fps * 1.5:
            v = last_verdict["verdict"]
            color = GREEN if v == "MAKE" else RED
            cv2.putText(frame, v, (w // 2 - 90, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8)
            cv2.putText(frame, v, (w // 2 - 90, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

        writer.write(frame)
        if frame_num % 100 == 0:
            print(f"  {frame_num}/{total}")

    cap.release()
    writer.release()

    finished = frame_num >= total or args.end <= 0 or args.end >= total
    if args.state:
        with open(args.state, "w") as f:
            json.dump({"rim": list(rim), "box": list(box),
                       "counter": counter.to_dict(),
                       "last_pos": detector.last,
                       "det_last_frame": detector.last_frame,
                       "det_vel": list(detector.vel),
                       "frame_num": frame_num}, f)
    if not finished:
        print(f"CHUNK DONE at frame {frame_num} (of {total})")
        return

    makes = sum(1 for s in counter.shots if s["verdict"] == "MAKE")
    n = sum(1 for s in counter.shots if s["verdict"] != "UNRESOLVED")
    with open(args.json, "w") as f:
        json.dump({"video": args.input, "rim": list(rim),
                   "box": list(box),
                   "logic": "simple_above_below",
                   "shots": counter.shots,
                   "makes": makes, "attempts": n}, f, indent=2)
    print(f"\nDone. {makes}/{n} makes "
          f"({len(counter.shots) - n} unresolved)")
    print(f"  Annotated video: {args.out}")
    print(f"  Shot data:       {args.json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out", default="simple_scored.mp4")
    p.add_argument("--json", default="simple_shots.json")
    p.add_argument("--box", default=None,
                   help="basket box x1,y1,x2,y2: rim top to NET BOTTOM "
                        "(the app's user-drawn calibration)")
    p.add_argument("--rim", default=None,
                   help="rim ring only; net bottom gets auto-extended")
    p.add_argument("--debug-rim", default=None)
    p.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "yolov8s.pt"))
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--roi-only", action="store_true")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--state", default=None)
    args = p.parse_args()
    process(args)
