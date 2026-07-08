"""
blob_detect.py — Lightweight ball detector (no neural net)
===========================================================
This is the reference implementation for the in-browser (cv.html) detector,
so it deliberately uses ONLY operations that port 1:1 to JavaScript canvas
pixel math:

  1. Running-average background of the region around the basket
     (static tripod => background is stable; the ball is what changes)
  2. Motion mask:   |frame - background| > threshold
  3. Color mask:    orange-ish pixels (wide HSV range)
  4. Ball = motion AND orange, cleaned up, blob of ball-like size
  5. Continuity gating: prefer the blob nearest the predicted position

Run as a script to validate against a clip using the same SimpleShotCounter
logic as score_simple.py:

    python blob_detect.py clips/mine/lucas_clip2.mp4 --box 661,285,701,352
        [--start N --end N --state s.json]   # chunked, like the others
"""

import sys
import os
import json
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from counter import SimpleShotCounter   # noqa: E402

MOTION_THRESHOLD = 25     # gray-level difference vs background
BG_ALPHA = 0.05           # background adapts slowly (ball never sticks)


class BlobDetector:
    def __init__(self, box, frame_size):
        x1, y1, x2, y2 = box
        self.box_w = x2 - x1
        self.ball_d = self.box_w * 0.55          # expected ball diameter
        w, h = frame_size
        cx = (x1 + x2) // 2
        # Watch ONLY the basket's vertical corridor — the shooter stays out
        # of view, so the only moving orange thing here is the ball arriving,
        # passing through, or bouncing off. (Matches the logic's philosophy:
        # everything is decided at the basket.)
        self.roi = (max(0, int(cx - self.box_w * 5.0)),
                    max(0, int(y1 - self.box_w * 3.0)),
                    min(w, int(cx + self.box_w * 5.0)),
                    min(h, int(y2 + self.box_w * 2.2)))
        self.center = (cx, (y1 + y2) / 2)
        self.bg = None                            # running-average gray bg
        self.warmup = 150                          # frames before trusting
                                                  # detections (bg converging)
        self.last = None
        self.last_frame = -999
        self.vel = (0.0, 0.0)

    def detect(self, frame, frame_num):
        """Returns (cx, cy, w) or None."""
        rx1, ry1, rx2, ry2 = self.roi
        crop = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.bg is None:
            self.bg = gray.copy()
            return None
        motion = (np.abs(gray - self.bg) > MOTION_THRESHOLD).astype(
            np.uint8) * 255
        self.bg = (1 - BG_ALPHA) * self.bg + BG_ALPHA * gray
        if self.warmup > 0:
            self.warmup -= 1
            return None

        # ball color: red-orange, BOTH sides of the HSV hue wrap (leather
        # balls often read as dark red, hue ~170+, value as low as ~50)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 90, 35]),
                         np.array([18, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 90, 35]),
                         np.array([179, 255, 255]))
        color = cv2.bitwise_or(m1, m2)

        mask = cv2.bitwise_and(motion, color)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if not (self.ball_d * 0.35 < cw < self.ball_d * 1.9):
                continue
            if not (0.45 < cw / max(ch, 1) < 2.2):
                continue
            cands.append((x + cw // 2 + rx1, y + ch // 2 + ry1, cw))
        if not cands:
            return None


        # gives approximation via local linearlization on where the ball should be
        pred = None
        age = frame_num - self.last_frame
        if self.last is not None and age <= 12:
            pred = (self.last[0] + self.vel[0] * age,
                    self.last[1] + self.vel[1] * age)


        # if there is a prediction, take the blob closest to it — but a
        # ball cannot teleport: if even the closest blob contradicts the
        # predicted motion by > 3 box-widths, it's a false blob (shooter's
        # body, reflection). Report NOTHING rather than tracking garbage.
        if pred is not None:
            best = min(cands, key=lambda c:
                       np.hypot(c[0] - pred[0], c[1] - pred[1]))
            if np.hypot(best[0] - pred[0],
                        best[1] - pred[1]) > self.box_w * 3:
                return None
        else:
            # no live track: re-acquire with the blob nearest the basket
            best = min(cands, key=lambda c:
                       np.hypot(c[0] - self.center[0],
                                c[1] - self.center[1]))

        if self.last is not None and 0 < age <= 12:
            self.vel = ((best[0] - self.last[0]) / age,
                        (best[1] - self.last[1]) / age)
        self.last = (best[0], best[1])
        self.last_frame = frame_num
        return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--box", required=True,
                   help="basket box x1,y1,x2,y2 (rim top to net bottom)")
    p.add_argument("--json", default="blob_shots.json")
    p.add_argument("--out", default=None,
                   help="write an annotated video (trail, box, lines, "
                        "verdicts, tally) — same overlay style as the app")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--state", default=None)
    args = p.parse_args()

    box = tuple(int(v) for v in args.box.split(","))
    cap = cv2.VideoCapture(args.input)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = args.end if args.end > 0 else total

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    writer = None
    out_path = args.out
    if args.out:
        if args.start > 0 or (args.end > 0 and args.end < total):
            out_path = f"{args.out}.part{args.start:06d}.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    det = BlobDetector(box, (w, h))
    counter = SimpleShotCounter(box)

    state = None
    if args.state and args.start > 0:
        try:
            with open(args.state) as f:
                state = json.load(f)
        except FileNotFoundError:
            pass
    if state:
        counter.from_dict(state["counter"])
        if state.get("last"):
            det.last = tuple(state["last"])
        det.last_frame = state.get("last_frame", args.start)
        det.vel = tuple(state.get("vel", (0.0, 0.0)))

    if args.start > 0:
        warm = max(0, args.start - 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm)
        for fn in range(warm, args.start):
            ret, f = cap.read()
            if not ret:
                break
            det.detect(f, fn)     # warm the background model

    from collections import deque
    trail = deque(maxlen=25)
    last_v, last_v_frame = None, -999

    frame_num = args.start
    n_det = 0
    while frame_num < end:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        d = det.detect(frame, frame_num)
        pos = (d[0], d[1]) if d else None
        if pos:
            n_det += 1
        v = counter.update(frame_num, pos, d[2] if d else None)
        if v:
            print(f"  frame {frame_num}: {v['verdict']} "
                  f"(x_return={v['x_at_return']})")
            if v['verdict'] != 'UNRESOLVED':
                last_v, last_v_frame = v['verdict'], frame_num

        if writer is not None:
            trail.append(pos)
            pts = [p for p in trail if p is not None]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0, 130, 255),
                         max(1, int(6 * i / len(pts))))
            if pos:
                cv2.circle(frame, pos, 5, (0, 130, 255), -1)
            # basket box + decision lines
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          (0, 220, 80), 2)
            cv2.line(frame, (0, box[1]), (w, box[1]), (200, 200, 0), 1)
            cv2.line(frame, (0, box[3]), (w, box[3]), (200, 200, 0), 1)
            # tally
            makes = sum(1 for s in counter.shots if s['verdict'] == 'MAKE')
            n = sum(1 for s in counter.shots if s['verdict'] != 'UNRESOLVED')
            hud = f"{makes}/{n} makes"
            if counter.state == 'AIRBORNE':
                hud += '  [ball up]'
            cv2.putText(frame, hud, (9, 31), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 4)
            cv2.putText(frame, hud, (9, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            # verdict flash
            if last_v and frame_num - last_v_frame < fps * 1.5 \
                    and counter.state != 'AIRBORNE':
                color = (0, 220, 80) if last_v == 'MAKE' else (0, 60, 220)
                cv2.putText(frame, last_v, (w//2 - 90, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8)
                cv2.putText(frame, last_v, (w//2 - 90, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
            writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
        print(f"  Annotated video: {out_path}")

    print(f"detections: {n_det}/{frame_num - args.start} frames")
    if args.state:
        with open(args.state, "w") as f:
            json.dump({"counter": counter.to_dict(), "last": det.last,
                       "last_frame": det.last_frame,
                       "vel": list(det.vel), "frame_num": frame_num}, f)
    if frame_num >= total or end >= total:
        makes = sum(1 for s in counter.shots if s["verdict"] == "MAKE")
        n = sum(1 for s in counter.shots if s["verdict"] != "UNRESOLVED")
        with open(args.json, "w") as f:
            json.dump({"shots": counter.shots, "makes": makes,
                       "attempts": n}, f, indent=2)
        print(f"Done. {makes}/{n} makes "
              f"({len(counter.shots) - n} unresolved)")
    else:
        print(f"CHUNK DONE at {frame_num}")


if __name__ == "__main__":
    main()
