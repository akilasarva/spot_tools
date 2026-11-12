"""Headless sim of the tablet-lease-handback bug (no ROS).

Story being modelled: autonomy is following an omniplanner plan, the robot
gets stuck before waypoint k, an operator takes the lease on the tablet and
pushes the robot past waypoint k, then hands the lease back. Desired: the
robot continues forward along the route. Bug: it walks back to waypoint k.

Uses the real SpotExecutor, LeaseManager, follow_trajectory_continuous,
IdentityPlanner / MidLevelPlanner and FakeSpot. Only the lease client, the
arm stow and the feedback sink are stubbed.

    PYTHONPATH=spot_tools/src:robot_executor_interface/robot_executor_interface/src \
        python spot_tools/examples/test_lease_handback.py --plot-dir /tmp/out

Each scenario prints PASS/FAIL. PASS means: after handback the robot never
moves backwards along the route by more than BACKTRACK_TOL metres, and the
sequence reaches the final waypoint.
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field

import types

import numpy as np

# The YOLO detector is never used in these scenarios; avoid the heavy
# ultralytics dependency by stubbing the module before spot_skills imports it.
if "ultralytics" not in sys.modules:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        sys.modules["ultralytics"] = types.SimpleNamespace(YOLOE=None, YOLO=None)
import shapely
import shapely.ops
from bosdyn.client.exceptions import LeaseUseError
from robot_executor_interface.action_descriptions import ActionSequence, Follow, Gaze
from robot_executor_interface.mid_level_planner import (
    IdentityPlanner,
    MidLevelPlanner,
    OccupancyMap,
)

import spot_executor.spot_executor as se_mod
import spot_executor.fake_spot as fake_spot_mod
from spot_executor.fake_spot import FakeSpot, Lease, LeaseOwner
from spot_executor.spot_executor import SpotExecutor

BACKTRACK_TOL = 0.5  # metres
AUTONOMY = "understanding-fake-spot"

# The real stow_arm drives the arm through the command client; irrelevant here.
se_mod.stow_arm = lambda spot: None


# --------------------------------------------------------------------------
# plan construction, mirroring omniplanner
# --------------------------------------------------------------------------
def compile_points_plan(waypoints, frame="map"):
    """Copy of omniplanner_ros.goto_points_ros.compile_points_plan:
    one Follow per start->goal leg, 10 interpolated points each."""
    actions = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        xs = np.interp(np.linspace(0, 1, 10), [0, 1], [a[0], b[0]])
        ys = np.interp(np.linspace(0, 1, 10), [0, 1], [a[1], b[1]])
        # path2d arrives Nx3 (x, y, yaw) after the ROS round trip
        # (spot_tools_ros.utils.path_to_waypoints)
        path = np.column_stack([xs, ys, np.zeros_like(xs)])
        actions.append(Follow(frame=frame, path2d=path))
    return actions


def compile_single_follow(waypoints, frame="map"):
    legs = [f.path2d for f in compile_points_plan(waypoints, frame)]
    path = np.vstack([legs[0]] + [leg[1:] for leg in legs[1:]])
    return [Follow(frame=frame, path2d=path)]


ROUTES = {
    "straight": np.array([[0, 0], [10, 0], [20, 0], [30, 0]], float),
    "L": np.array([[0, 0], [10, 0], [10, 10], [10, 20]], float),
    # two parallel legs 4 m apart: tests nearest-point snapping (H2)
    "U": np.array([[0, 0], [12, 0], [12, 4], [0, 4]], float),
    # out-and-back along the same corridor (TSP / goto-points can do this):
    # where the robot ends up is ambiguous between the two legs
    "out_back": np.array([[0, 0], [10, 0], [0, 0.5]], float),
    # paper clip: 16 m out, a rounded U-turn (radius 4), 16 m back, 8 m apart
    "clip": np.array([[0, 0], [16, 0], [18.83, 1.17], [20, 4], [18.83, 6.83],
                      [16, 8], [0, 8]], float),
    # straight with a waypoint every 3 m
    "many": np.array([[3.0 * i, 0] for i in range(11)], float),
}


def route_points(sc):
    if isinstance(sc.route, str):
        return ROUTES[sc.route]
    return np.asarray(sc.route, float)


def track_progress(route, pts, s0=0.0, back=1.0, ahead=2.0):
    """Arc length of each point, following the robot along the route: each
    point is projected only onto [s - back, s + ahead] around the previous
    one, so the metric cannot jump to a parallel part of the route."""
    out, s = [], s0
    for x, y in pts:
        lo, hi = max(0.0, s - back), min(route.length, s + ahead)
        if hi - lo < 1e-6:
            out.append(s)
            continue
        sub = shapely.ops.substring(route, lo, hi)
        s = lo + sub.line_locate_point(shapely.Point(x, y))
        out.append(s)
    return np.array(out)


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------
class ScriptedLeaseClient:
    """`owner` is the true lease holder. `poll_lag` delays what list_leases()
    (i.e. the LeaseManager poll) sees, so that robot_command can hit the
    lease loss first, as it can on the real robot."""

    def __init__(self, poll_lag=0.0):
        self._owner = AUTONOMY
        self._prev = AUTONOMY
        self._t_change = 0.0
        self.poll_lag = poll_lag

    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, value):
        self._prev, self._owner, self._t_change = self._owner, value, time.time()

    def list_leases(self):
        seen = self._prev if time.time() - self._t_change < self.poll_lag else self._owner
        return [Lease(LeaseOwner(seen))]


_orig_robot_command = fake_spot_mod.FakeCommandClient.robot_command


def _robot_command_checking_lease(self, command, *args, **kwargs):
    """Real Spot rejects commands from a client that no longer holds the
    lease with LeaseUseError. FakeSpot never does; emulate it when enabled."""
    spot = self.fake_spot
    if getattr(spot, "raise_on_lease_loss", False) and not spot.lease_client.owner.startswith("understanding"):
        time.sleep(0.05)
        raise LeaseUseError(None, None)
    return _orig_robot_command(self, command, *args, **kwargs)


fake_spot_mod.FakeCommandClient.robot_command = _robot_command_checking_lease


class Recorder:
    """Feedback sink with the attributes SpotExecutor/follower touch."""

    def __init__(self):
        self.plan_valid = True
        self.break_out_of_waiting_loop = False
        self.targets = []  # (t, x, y)
        self.done = []  # (t, action_index_guess, success)
        self.messages = []

    def print(self, level, *args):
        msg = " ".join(str(a) for a in args)
        if level != "DEBUG":
            self.messages.append((time.time(), level, msg))

    def path_following_progress_feedback(self, progress_point, target_point):
        self.targets.append((time.time(), target_point.x, target_point.y))

    def path_follow_MLP_feedback(self, *args):
        pass

    def follow_path_feedback(self, *args):
        pass

    def log_lease_takeover(self, s):
        self.messages.append((time.time(), "LEASE", s))

    def report_action_done(self, sequence, command, success):
        self.done.append((time.time(), next(i for i, a in enumerate(sequence.actions) if a is command), success))


class GridRefresher:
    """Stands in for OccupancyGridROSUpdater: an empty robot-centred grid that
    is re-published around the FakeSpot pose. `frozen` stops updates (H3)."""

    def __init__(self, spot, omap, size=160, res=0.12, period=0.2, obstacles=()):
        self.spot, self.omap = spot, omap
        self.obstacles = obstacles
        self.size, self.res, self.period = size, res, period
        self.frozen = False
        self.stop = False
        self.push()
        threading.Thread(target=self._run, daemon=True).start()

    def push(self):
        x, y = self.spot.get_pose()[:2]
        origin = np.eye(4)
        origin[:2, 3] = [x - self.size * self.res / 2, y - self.size * self.res / 2]
        robot = np.eye(4)
        robot[:2, 3] = [x, y]
        grid = np.zeros((self.size, self.size), dtype=np.int8)
        if self.obstacles:
            jj, ii = np.meshgrid(np.arange(self.size), np.arange(self.size))
            gx, gy = origin[0, 3] + jj * self.res, origin[1, 3] + ii * self.res
            for ox, oy, r in self.obstacles:
                grid[(gx - ox) ** 2 + (gy - oy) ** 2 <= r * r] = 100
        self.omap.set_grid(grid, self.res, origin, robot, time.time())

    def _run(self):
        while not self.stop:
            if not self.frozen:
                self.push()
            time.sleep(self.period)


# --------------------------------------------------------------------------
# scenario
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    route: str
    plan: str  # "per_leg" (omniplanner) or "single"
    planner: str  # "identity" or "astar"
    overshoot: float  # metres past waypoint k along the route
    lateral: float = 0.0  # operator leaves robot this far left of the route
    k: int = 1  # waypoint the robot is stuck before
    stuck_before: float = 4.0  # takeover happens this far before waypoint k
    grid_freeze_after_handback: float = 0.0  # H3: seconds the MLP pose stays stale
    goal_tolerance: float = 2.8
    lookahead: float = 2.0
    # real-robot lease loss: robot_command raises LeaseUseError and the
    # LeaseManager poll only notices 0.5 s later
    lease_error: bool = False
    # append a Gaze at the final waypoint (PDDL goto-poi -> inspect)
    gaze_after: bool = False
    # "forward": must not walk back after handback (pass-through waypoint)
    # "return":  k is the last waypoint (goal / POI); the robot SHOULD go back
    #            and finish (and gaze) within goal tolerance of it
    expect: str = "forward"
    desc: str = ""  # plot title override for the operator action
    # operator drives this polyline (absolute x, y) instead of along the route
    drive: tuple = ()
    obstacles: tuple = ()  # (x, y, r) circles in the A* grid
    poll_lag: float = None  # LeaseManager sees lease changes this late (s)
    pose_fail_p: float = 0.0  # takeover pose reads fail with this probability
    tf_fail: bool = False  # transform lookup fails in the passed-waypoint check
    retake: tuple = ()  # (gap_s, extra_m): tablet takes the lease again
    s_handback: float = None  # route progress at handback, if the drive knows it


@dataclass
class Result:
    sc: Scenario
    trace: list = field(default_factory=list)  # (t, x, y, phase)
    targets: list = field(default_factory=list)
    s_handback: float = 0.0
    backtrack: float = 0.0
    finished: bool = False
    final_dist: float = 0.0
    gaze_dist: float = None  # robot distance to the POI when Gaze ran
    missed: list = field(default_factory=list)  # waypoints ahead never reached
    clearance: float = None  # min distance to an obstacle edge after handback
    s_handback1: float = None  # first handback when retake is set
    backtrack_between: float = 0.0
    note: str = ""
    messages: list = field(default_factory=list)

    @property
    def passed(self):
        if self.sc.gaze_after and (
            self.gaze_dist is None or self.gaze_dist > self.sc.goal_tolerance + 0.5
        ):
            return False
        if self.sc.expect == "return":
            return self.finished
        return (self.finished and self.backtrack <= BACKTRACK_TOL and not self.missed
                and self.backtrack_between <= BACKTRACK_TOL)


def run_scenario(sc: Scenario, timeout_after_handback=60.0) -> Result:
    W = route_points(sc)
    route = shapely.LineString(W)
    s_k = route.line_locate_point(shapely.Point(W[sc.k]))
    actions = (
        compile_points_plan(W) if sc.plan == "per_leg" else compile_single_follow(W)
    )
    if sc.gaze_after:
        poi = np.array([W[-1][0], W[-1][1], 0.0])
        actions = actions + [
            Gaze(frame="map", robot_point=poi, gaze_point=poi + [2.0, 0, 0],
                 object_id="poi")
        ]
    seq = ActionSequence(plan_id=sc.name, robot_name="spot", actions=actions)

    spot = FakeSpot(init_pose=np.array([W[0][0], W[0][1], 0.0, 0.0]))
    lag = sc.poll_lag if sc.poll_lag is not None else (0.5 if sc.lease_error else 0.0)
    lease = ScriptedLeaseClient(poll_lag=lag)
    spot.lease_client = lease
    spot.raise_on_lease_loss = sc.lease_error
    spot.take_lease = lambda: setattr(lease, "owner", AUTONOMY)

    rec = Recorder()
    refresher = None
    if sc.planner == "astar":
        omap = OccupancyMap(rec, inflate_radius_meters=0.2)
        refresher = GridRefresher(spot, omap, obstacles=sc.obstacles)
        planner = MidLevelPlanner(omap, rec, lookahead_distance_grid=50)
    else:
        planner = IdentityPlanner(rec)

    rng = np.random.default_rng(abs(hash(sc.name)) % 2**32)
    if sc.pose_fail_p > 0:
        real_get_pose = spot.get_pose

        def flaky_get_pose():
            if (sys._getframe(1).f_code.co_name == "_wait_recording_takeover"
                    and rng.random() < sc.pose_fail_p):
                raise RuntimeError("simulated robot-state read failure")
            return real_get_pose()

        spot.get_pose = flaky_get_pose

    def transform_lookup(a, b):
        if sc.tf_fail and sys._getframe(1).f_code.co_name == "_path_xy":
            raise RuntimeError("simulated TF lookup failure")
        return np.zeros(3), _Quat()

    ex = SpotExecutor(
        spot,
        detector=None,
        transform_lookup=transform_lookup,
        planner=planner,
        follower_lookahead=sc.lookahead,
        goal_tolerance=sc.goal_tolerance,
    )
    ex.initialize_lease_manager(rec)

    res = Result(sc)
    res.messages = rec.messages

    def fake_gaze(command, feedback, pick_next=False):
        x, y = spot.get_pose()[:2]
        res.gaze_dist = float(np.hypot(x - W[-1][0], y - W[-1][1]))
        return True

    ex.execute_gaze = fake_gaze
    phase = ["auto"]
    stop_logging = threading.Event()

    def log_pose():
        while not stop_logging.is_set():
            x, y = spot.get_pose()[:2]
            res.trace.append((time.time(), x, y, phase[0]))
            time.sleep(0.05)

    threading.Thread(target=log_pose, daemon=True).start()
    ex_thread = threading.Thread(
        target=ex.process_action_sequence, args=(seq, rec), daemon=True
    )
    ex_thread.start()

    def progress():
        x, y = spot.get_pose()[:2]
        return route.line_locate_point(shapely.Point(x, y))

    # 1. autonomy drives until the robot is "stuck" before waypoint k
    t0 = time.time()
    while progress() < s_k - sc.stuck_before:
        if time.time() - t0 > 60:
            res.note = "never reached takeover point"
            return _finish(res, ex, stop_logging, refresher)
        time.sleep(0.02)

    # 2. operator takes the lease on the tablet
    lease.owner = "tablet-operator"
    phase[0] = "tablet"
    time.sleep(0.5)  # executor notices, follower exits
    if refresher and sc.grid_freeze_after_handback > 0:
        refresher.frozen = True

    # 3. operator pushes the robot along the route past waypoint k
    s_goal = s_k + sc.overshoot
    s = progress()
    if sc.drive:
        pts = [spot.get_pose()[:2]] + [np.asarray(p, float) for p in sc.drive]
        dense = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / (1.5 * 0.05))))
            for f in np.linspace(0, 1, n + 1)[1:]:
                x, y = a + f * (b - a)
                dense.append((x, y))
                spot.set_pose(np.array([x, y, 0.0, 0.0]))
                time.sleep(0.05)
        s_goal = (sc.s_handback if sc.s_handback is not None
                  else track_progress(route, dense, s0=s)[-1])
        s = s_goal
    while s < s_goal:
        s = min(s + 1.5 * 0.05, s_goal)  # 1.5 m/s
        p = _point_at(route, s)
        # left-normal offset ramps in over the push
        frac = (s - (s_k - sc.stuck_before)) / (s_goal - (s_k - sc.stuck_before))
        nx, ny = _left_normal(route, s)
        x, y = p.x + sc.lateral * frac * nx, p.y + sc.lateral * frac * ny
        spot.set_pose(np.array([x, y, 0.0, 0.0]))
        time.sleep(0.05)

    # 4. operator releases the lease; LeaseManager takes it back
    res.s_handback = s_goal
    lease.owner = ""
    phase[0] = "after"
    t_hand = time.time()
    if refresher and sc.grid_freeze_after_handback > 0:
        threading.Timer(
            sc.grid_freeze_after_handback, lambda: setattr(refresher, "frozen", False)
        ).start()

    if sc.retake:
        gap, extra = sc.retake
        phase[0] = "between"
        time.sleep(gap)
        res.s_handback1 = res.s_handback
        lease.owner = "tablet-operator"
        phase[0] = "tablet"
        time.sleep(0.5)
        s = progress()
        s_goal = s + extra
        while s < s_goal:
            s = min(s + 1.5 * 0.05, s_goal)
            p = _point_at(route, s)
            spot.set_pose(np.array([p.x, p.y, 0.0, 0.0]))
            time.sleep(0.05)
        res.s_handback = s_goal
        lease.owner = ""
        phase[0] = "after"

    ex_thread.join(timeout=timeout_after_handback)
    res.finished = not ex_thread.is_alive()
    res.targets = [t for t in rec.targets if t[0] >= t_hand]
    return _finish(res, ex, stop_logging, refresher)


def _finish(res, ex, stop_logging, refresher):
    ex.keep_going = False
    time.sleep(0.3)
    stop_logging.set()
    if refresher:
        refresher.stop = True
    W = route_points(res.sc)
    # extended 20 m past the end, as in plot(), so overshoot is measurable
    d_end = (W[-1] - W[-2]) / np.linalg.norm(W[-1] - W[-2])
    route = shapely.LineString(np.vstack([W, W[-1] + 20 * d_end]))
    after = np.array([(x, y) for _, x, y, ph in res.trace if ph == "after"])
    if len(after):
        s_after = track_progress(route, after, s0=res.s_handback)
        res.backtrack = max(0.0, res.s_handback - s_after.min())
        res.final_dist = float(np.linalg.norm(after[-1] - W[-1]))
        res.finished = res.finished and res.final_dist < res.sc.goal_tolerance + 0.5
        for j in range(1, len(W) - 1):
            s_j = route.line_locate_point(shapely.Point(W[j]))
            if s_j > res.s_handback and (
                np.linalg.norm(after - W[j], axis=1).min()
                > res.sc.goal_tolerance + 1.0
            ):
                res.missed.append(j)
        if res.sc.obstacles:
            res.clearance = float(min(
                np.hypot(after[:, 0] - ox, after[:, 1] - oy).min() - r
                for ox, oy, r in res.sc.obstacles))
    between = [(x, y) for _, x, y, ph in res.trace if ph == "between"]
    if between and res.s_handback1 is not None:
        sb = track_progress(route, between, s0=res.s_handback1)
        res.backtrack_between = max(0.0, res.s_handback1 - sb.min())
    return res


class _Quat:
    x = y = z = 0.0
    w = 1.0


def _point_at(route, s):
    """Point at arc length s, extrapolating past the end along the last leg."""
    if s <= route.length:
        return route.interpolate(s)
    c = np.asarray(route.coords)
    d = c[-1] - c[-2]
    d /= np.linalg.norm(d)
    e = c[-1] + (s - route.length) * d
    return shapely.Point(e)


def _left_normal(route, s, eps=0.05):
    a = route.interpolate(max(0.0, s - eps))
    b = route.interpolate(min(route.length, s + eps))
    d = np.array([b.x - a.x, b.y - a.y])
    d /= np.linalg.norm(d) + 1e-9
    return -d[1], d[0]


# --------------------------------------------------------------------------
def plot(res: Result, out_dir):
    """Three panels: whole route, zoom on the passed waypoint, and route
    progress vs time. Waypoints are labelled W0..Wn; the passed waypoint has
    its goal-tolerance circle; the walk-back (if any) is drawn as a thick
    red arrow from the handback point to the furthest point back."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    sc = res.sc
    W = route_points(sc)
    # extend the route 20 m past its end so positions beyond the final
    # waypoint get a real distance instead of all snapping to the end
    d_end = (W[-1] - W[-2]) / np.linalg.norm(W[-1] - W[-2])
    route = shapely.LineString(np.vstack([W, W[-1] + 20 * d_end]))
    colors = {"auto": "tab:blue", "tablet": "tab:orange", "after": "tab:green"}
    phase_label = {"auto": "autonomy (before)", "tablet": "operator on tablet",
                   "after": "autonomy (after handback)"}
    tr = {ph: np.array([(t, x, y) for t, x, y, p in res.trace if p == ph])
          for ph in colors}

    # handback point and furthest point back (same metric as the PASS/FAIL)
    hand = tr["tablet"][-1, 1:] if len(tr["tablet"]) else None
    worst = None
    # auto: tracked from W0; tablet: the scripted push, linear in arc length;
    # after: tracked from the handback point
    s_by = {ph: np.zeros(0) for ph in colors}
    if len(tr["auto"]):
        s_by["auto"] = track_progress(route, tr["auto"][:, 1:])
    if len(tr["tablet"]) and sc.drive:
        s_by["tablet"] = np.array([route.line_locate_point(shapely.Point(x, y))
                                   for x, y in tr["tablet"][:, 1:]])
    elif len(tr["tablet"]):
        s0 = s_by["auto"][-1] if len(s_by["auto"]) else 0.0
        s_by["tablet"] = np.linspace(s0, res.s_handback, len(tr["tablet"]))
    if len(tr["after"]):
        s_by["after"] = track_progress(route, tr["after"][:, 1:], s0=res.s_handback)
    if len(tr["after"]):
        s_after = s_by["after"]
        i = int(np.argmin(s_after))
        if res.s_handback - s_after[i] > 0.05:
            worst = (tr["after"][i, 0], tr["after"][i, 1:], s_after[i])
    back = res.backtrack if sc.expect == "forward" else None

    fig, (ax, axz, ax2) = plt.subplots(1, 3, figsize=(18, 6),
                                       gridspec_kw={"width_ratios": [1, 1, 1.2]})

    def draw_map(a, zoom):
        a.plot(W[:, 0], W[:, 1], "k--", lw=1, zorder=1, label="planned route")
        for j, w in enumerate(W):
            a.plot(*w, "ks", ms=7, zorder=5)
            a.annotate(f"W{j}", w, xytext=(6, 6), textcoords="offset points",
                       fontsize=11, fontweight="bold")
        a.add_patch(Circle(W[sc.k], sc.goal_tolerance, fill=False, ls=":",
                           ec="r", lw=1.2, zorder=2,
                           label=f"goal tolerance {sc.goal_tolerance} m around W{sc.k}"))
        a.plot(*W[sc.k], "o", mfc="none", mec="r", ms=16, mew=2, zorder=6,
               label=f"W{sc.k} = waypoint pushed past")
        for ph, c in colors.items():
            if len(tr[ph]):
                a.plot(tr[ph][:, 1], tr[ph][:, 2], ".", ms=4 if ph != "after" else 5,
                       color=c, zorder=3 if ph != "after" else 4,
                       label=phase_label[ph])
        if len(tr["tablet"]):
            a.plot(*tr["tablet"][0, 1:], "v", color="tab:orange", ms=11,
                   mec="k", zorder=7, label="tablet takes lease")
        if hand is not None:
            a.plot(*hand, "*", color="gold", ms=18, mec="k", zorder=8,
                   label="lease handed back")
        if worst is not None:
            a.annotate("", xy=worst[1], xytext=hand, zorder=9,
                       arrowprops=dict(arrowstyle="-|>", color="red", lw=3))
            a.plot(*worst[1], "X", color="red", ms=12, mec="k", zorder=9,
                   label="furthest point walked back to")
            if zoom:
                mid = (np.asarray(hand) + worst[1]) / 2
                a.annotate(f"walked back {res.s_handback - worst[2]:.1f} m",
                           mid, xytext=(18, 0), textcoords="offset points",
                           ha="left", va="center", color="red", fontsize=12,
                           fontweight="bold")
        for j in res.missed:
            a.plot(*W[j], "X", color="red", ms=14, mec="k", zorder=9)
            a.annotate("never reached", W[j], xytext=(6, -14),
                       textcoords="offset points", color="red", fontsize=9,
                       fontweight="bold")
        a.set_aspect("equal", adjustable="datalim")

    draw_map(ax, zoom=False)
    xs, ys = W[:, 0], W[:, 1]
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    half = max(np.ptp(xs), np.ptp(ys), 8) / 2 + 3
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_title("whole route (top-down, metres)")
    ax.legend(fontsize=7, loc="upper left")

    draw_map(axz, zoom=True)
    zc = W[sc.k] if hand is None else (W[sc.k] + np.asarray(hand)) / 2
    zh = max(sc.overshoot, sc.goal_tolerance, 3) + 2.5
    axz.set_xlim(zc[0] - zh, zc[0] + zh)
    axz.set_ylim(zc[1] - zh, zc[1] + zh)
    axz.set_title(f"zoom on W{sc.k}")
    axz.grid(alpha=0.3)

    # progress along the route vs time
    if res.trace:
        t0 = res.trace[0][0]
        for ph, c in colors.items():
            if len(tr[ph]):
                P = np.column_stack([tr[ph][:, 0] - t0, s_by[ph]])
                ax2.plot(P[:, 0], P[:, 1], ".", ms=3, color=c, label=phase_label[ph])
        for j, w in enumerate(W):
            sw = route.line_locate_point(shapely.Point(w))
            ax2.axhline(sw, color="r" if j == sc.k else "0.6",
                        lw=1.2 if j == sc.k else 0.7, ls=":")
            ax2.text(0, sw, f" W{j}",
                     va="bottom", fontsize=10,
                     color="r" if j == sc.k else "0.3", fontweight="bold")
        if len(tr["tablet"]):
            th = tr["tablet"][-1, 0] - t0
            ax2.axvline(th, color="gold", lw=1.5, label="lease handed back")
            ax2.plot(th, res.s_handback, "*", color="gold", ms=16, mec="k")
        if worst is not None:
            tw = worst[0] - t0
            ax2.annotate("", xy=(tw, worst[2]), xytext=(tw, res.s_handback),
                         arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
            ax2.text(tw + 0.3, (worst[2] + res.s_handback) / 2,
                     f"walked back\n{res.s_handback - worst[2]:.1f} m",
                     color="red", fontsize=11, fontweight="bold", va="center")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("distance along planned route [m]")
        ax2.legend(fontsize=8, loc="lower right")
        ax2.set_title("progress along route (any dip after the star = walk-back)")
        ax2.grid(alpha=0.3)

    verdict = "PASS" if res.passed else "FAIL"
    detail = (f"walk-back {back:.1f} m" if back is not None
              else "must return to the final goal / POI")
    if res.missed:
        detail += "  |  never reached " + ", ".join(f"W{j}" for j in res.missed)
    action = sc.desc or f"pushed {sc.overshoot:g} m past W{sc.k}"
    fig.suptitle(f"{sc.name}  |  {verdict}  |  {detail}  |  "
                 f"{action}, goal_tolerance {sc.goal_tolerance} m", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{sc.name}.png"), dpi=100)
    plt.close(fig)


def all_scenarios(goal_tolerance):
    gt = goal_tolerance
    out = []
    # omniplanner-style plans: one Follow per leg (the reported case)
    for planner in ("identity", "astar"):
        for route in ("straight", "L", "U"):
            for d in (1.0, 4.0):
                out.append(Scenario(f"per_leg-{planner}-{route}-over{d:g}",
                                    route, "per_leg", planner, d, goal_tolerance=gt))
    out.append(Scenario("per_leg-identity-out_back-over3", "out_back", "per_leg",
                        "identity", 3.0, stuck_before=2.0, goal_tolerance=gt))
    out.append(Scenario("per_leg-identity-L-over4-lateral2", "L", "per_leg",
                        "identity", 4.0, lateral=2.0, goal_tolerance=gt))
    # same, but lease loss surfaces as LeaseUseError (real-robot path)
    for route in ("straight", "L"):
        out.append(Scenario(f"per_leg-identity-{route}-over4-leaseerror", route,
                            "per_leg", "identity", 4.0, lease_error=True,
                            goal_tolerance=gt))
    # control: whole route as one Follow
    for planner in ("identity", "astar"):
        for route in ("straight", "L", "U"):
            out.append(Scenario(f"single-{planner}-{route}-over4", route, "single",
                                planner, 4.0, goal_tolerance=gt))
    # final goal / POI: the robot must still go back to it (no next waypoint)
    out.append(Scenario("per_leg-identity-straight-pastgoal-over4", "straight",
                        "per_leg", "identity", 4.0, k=3, expect="return",
                        goal_tolerance=gt))
    out.append(Scenario("per_leg-identity-L-gaze-pastpoi-over4", "L", "per_leg",
                        "identity", 4.0, k=3, gaze_after=True, expect="return",
                        goal_tolerance=gt))
    # intermediate waypoint passed on a plan that ends in a Gaze: skip it,
    # and the Gaze still happens at the POI
    out.append(Scenario("per_leg-identity-L-gaze-over4", "L", "per_leg",
                        "identity", 4.0, k=1, gaze_after=True, goal_tolerance=gt))
    out.append(Scenario("single-astar-straight-over4-stalegrid3", "straight",
                        "single", "astar", 4.0, grid_freeze_after_handback=3.0,
                        goal_tolerance=gt))
    # paper clip, obstacle dodge: stuck 8 m into the outbound straight, the
    # operator steers 5 m toward the return straight (now nearer to it than
    # to the outbound one) and hands back there, 4 m before W1
    for plan in ("per_leg", "single"):
        for planner in ("identity", "astar"):
            out.append(Scenario(
                f"{plan}-{planner}-clip-dodge", "clip", plan, planner, -4.0,
                lateral=5.0, k=1, stuck_before=8.0, goal_tolerance=gt,
                desc="dodged 5 m toward the return straight, 4 m before W1"))
    # paper clip, pushed through the whole U-turn: past W1..W5
    for plan in ("per_leg", "single"):
        out.append(Scenario(
            f"{plan}-identity-clip-pastturn", "clip", plan, "identity", 16.0,
            k=1, stuck_before=3.0, goal_tolerance=gt,
            desc="pushed through the U-turn, past W1-W5"))
    # operator drives around an obstacle off the path, then carries on.
    # straight route, obstacle at x = 6 (between W0 and W1): around it and
    # back onto the path at x = 9, or handed back halfway round
    out.append(Scenario("per_leg-identity-straight-detour-rejoin", "straight",
                        "per_leg", "identity", 0.0, k=1, stuck_before=6.0,
                        drive=((5, 2.5), (8, 2.5), (9, 0.3)), goal_tolerance=gt,
                        desc="drove round an obstacle at x=6, back on the path"))
    out.append(Scenario("per_leg-identity-straight-detour-midway", "straight",
                        "per_leg", "identity", 0.0, k=1, stuck_before=6.0,
                        drive=((5, 2.5), (7, 2.5)), goal_tolerance=gt,
                        desc="handed back halfway round an obstacle at x=6"))
    # straight route, obstacle ON waypoint W1: wide around it (never within
    # tolerance of W1), then on along the next leg
    out.append(Scenario("per_leg-identity-straight-detour-aroundW1", "straight",
                        "per_leg", "identity", 0.0, k=1, stuck_before=4.0,
                        drive=((7, 4), (13, 4), (15, 0.5)), goal_tolerance=gt,
                        desc="drove 4 m wide round an obstacle on W1, back on the path"))
    # L route, obstacle in the corner W1: round the outside, then handed back
    # either well onto the next leg or just after getting round
    out.append(Scenario("per_leg-identity-L-detour-corner-far", "L", "per_leg",
                        "identity", 0.0, k=1, stuck_before=3.0,
                        drive=((7, -4), (14, -4), (14, 4), (10.5, 6)),
                        goal_tolerance=gt,
                        desc="round the outside of an obstacle in corner W1, on up the next leg"))
    out.append(Scenario("per_leg-identity-L-detour-corner-near", "L", "per_leg",
                        "identity", 0.0, k=1, stuck_before=3.0,
                        drive=((7, -4), (14, -4), (13, 1)),
                        goal_tolerance=gt,
                        desc="round the outside of an obstacle in corner W1, handed back just past it"))
    # many waypoints: pushed past five of them (W2..W6, 3 m apart)
    for planner in ("identity", "astar"):
        out.append(Scenario(
            f"per_leg-{planner}-many-past5", "many", "per_leg", planner, 13.0,
            k=2, stuck_before=2.0, goal_tolerance=gt,
            desc="pushed past W2-W6"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", help="'all' or substring filter")
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--goal-tolerance", type=float, default=2.8)
    args = ap.parse_args()

    scenarios = all_scenarios(args.goal_tolerance)
    if args.scenario != "all":
        scenarios = [s for s in scenarios if args.scenario in s.name]
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    # executor / FakeSpot are very chatty on stdout; keep the report on stderr
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    n_pass = 0
    try:
        for sc in scenarios:
            res = run_scenario(sc)
            n_pass += res.passed
            print(
                f"{'PASS' if res.passed else 'FAIL'}  {sc.name:42s} "
                f"backtrack={res.backtrack:5.2f} m  finished={res.finished}"
                f"{f'  gaze_dist={res.gaze_dist:.2f} m' if res.gaze_dist is not None else ''}"
                f"{'  ' + res.note if res.note else ''}",
                file=sys.stderr, flush=True,
            )
            if args.plot_dir:
                plot(res, args.plot_dir)
    finally:
        sys.stdout = real_stdout
    print(f"{n_pass}/{len(scenarios)} passed", file=sys.stderr, flush=True)
    # LeaseManager's monitor thread is non-daemon and never exits.
    os._exit(0 if n_pass == len(scenarios) else 1)


if __name__ == "__main__":
    main()
