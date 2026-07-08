"""
viz_pipeline.py — Visualize the blob detector's "eye".
=======================================================
2x3 panel of the detection pipeline (same math as blob_detect.py / cv.html):

  [1] current frame        [2] background memory     [3] motion mask
      (ROI around basket)      (learned idle scene       |current-background|
                                — the ball averages       > 25
                                out of existence)
  [4] color mask           [5] motion AND color      [6] result
      red/orange HSV                                     ball circled

Still:  python viz_pipeline.py clips/mine/lucas_clip2.mp4 \
            --box 661,285,701,352 --frame 205 --out pipeline.png
Video:  python viz_pipeline.py clips/mine/lucas_clip2.mp4 \
            --box 661,285,701,352 --start 190 --end 260 --out pipeline.mp4
"""

import argparse
import cv2
import numpy as np

MOTION_THRESHOLD = 25
BG_ALPHA = 0.05


def caption(tile, text):
    bar = np.full((34, tile.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([tile, bar])


def render_panel(crop, bg, box_roi, box_w):
    """All six tiles for one frame. bg is the float32 background model."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    motion = (np.abs(gray - bg) > MOTION_THRESHOLD).astype(np.uint8) * 255

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 90, 35]), np.array([18, 255, 255])),
        cv2.inRange(hsv, np.array([160, 90, 35]), np.array([179, 255, 255])))

    both = cv2.bitwise_and(motion, color)
    both = cv2.morphologyEx(both, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    both = cv2.dilate(both, np.ones((5, 5), np.uint8))

    t1 = crop.copy()
    cv2.rectangle(t1, box_roi[:2], box_roi[2:], (0, 220, 80), 2)
    t2 = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    t3 = cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR)
    t4 = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
    t5 = cv2.cvtColor(both, cv2.COLOR_GRAY2BGR)

    # circle only the SINGLE best candidate (nearest the basket), like the
    # real detector's selection — not every plausible blob
    t6 = crop.copy()
    ball_d = box_w * 0.55
    bcx = (box_roi[0] + box_roi[2]) / 2
    bcy = (box_roi[1] + box_roi[3]) / 2
    contours, _ = cv2.findContours(both, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best, best_d = None, 1e9
    for c in contours:
        bx, by, bw_, bh = cv2.boundingRect(c)
        if ball_d * 0.35 < bw_ < ball_d * 1.9 and 0.45 < bw_/max(bh, 1) < 2.2:
            d = ((bx + bw_/2 - bcx)**2 + (by + bh/2 - bcy)**2) ** 0.5
            if d < best_d:
                best_d, best = d, (bx + bw_//2, by + bh//2, max(bw_, bh))
    if best:
        cv2.circle(t6, best[:2], int(best[2]), (0, 130, 255), 3)

    return np.vstack([
        np.hstack([caption(t1, "1. current frame"),
                   caption(t2, "2. background memory (idle scene - no ball!)"),
                   caption(t3, "3. motion = |current - background| > 25")]),
        np.hstack([caption(t4, "4. color: red/orange pixels (HSV)"),
                   caption(t5, "5. motion AND color"),
                   caption(t6, "6. result: the ball")]),
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--box", required=True)
    p.add_argument("--frame", type=int, default=None, help="still mode")
    p.add_argument("--start", type=int, default=None, help="video mode")
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--out", default="pipeline.png")
    args = p.parse_args()

    x1, y1, x2, y2 = (int(v) for v in args.box.split(","))
    box_w = x2 - x1
    cx = (x1 + x2) // 2
    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    rx1 = max(0, int(cx - box_w * 5.0)); ry1 = max(0, int(y1 - box_w * 3.0))
    rx2 = min(W, int(cx + box_w * 5.0)); ry2 = min(H, int(y2 + box_w * 2.2))
    box_roi = (x1-rx1, y1-ry1, x2-rx1, y2-ry1)

    last = args.frame if args.frame is not None else args.end
    first = args.frame if args.frame is not None else args.start
    writer = None

    bg = None
    for fn in range(last + 1):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if bg is None:
            bg = gray.copy()
            continue
        if fn >= first:
            panel = render_panel(crop, bg, box_roi, box_w)
            if args.frame is not None:                 # still mode
                cv2.imwrite(args.out, panel)
                print(f"saved {args.out} ({panel.shape[1]}x{panel.shape[0]})")
                return
            if writer is None:                          # video mode
                writer = cv2.VideoWriter(
                    args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (panel.shape[1], panel.shape[0]))
            writer.write(panel)
        bg = (1 - BG_ALPHA) * bg + BG_ALPHA * gray
    if writer is not None:
        writer.release()
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
