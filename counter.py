"""
counter.py — The make/miss logic (the "brain").
================================================
Detector-agnostic: feed it one ball position per frame from ANY detector
(blob detector, YOLO, anything) and it counts attempts, makes, and misses.

Two states. IDLE: watch for the ball near/above the rim top (arms an
attempt). AIRBORNE: watch for the ball to reappear FALLING below the box
bottom — inside the box's x-range = MAKE, outside = MISS. Guards ensure a
verdict can only come from a physically plausible, observed crossing.
"""

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


