import pytest

from controller import ChaseConfig, ChaseController, State
from detector import Detection

W, H = 1000, 1000


def ball(cx, bottom, size=40):
    """A ball centered horizontally at cx with its bottom edge at `bottom` (px)."""
    return Detection(x=cx - size // 2, y=bottom - size, w=size, h=size)


@pytest.fixture
def cfg():
    return ChaseConfig()


def test_starts_searching_and_pulses(cfg):
    c = ChaseController(cfg)
    turning = c.update([], W, H, now=0.0)
    assert turning.state == State.SEARCH
    assert turning.left_pct == cfg.search_pct and turning.right_pct == -cfg.search_pct
    paused = c.update([], W, H, now=cfg.search_turn_s + 0.01)
    assert (paused.left_pct, paused.right_pct) == (0.0, 0.0)
    again = c.update([], W, H, now=cfg.search_turn_s + cfg.search_pause_s + 0.01)
    assert again.left_pct > 0


def test_off_center_ball_turns_in_place_toward_it(cfg):
    c = ChaseController(cfg)
    right = c.update([ball(900, 400)], W, H, now=0.0)
    assert right.state == State.ALIGN
    assert right.left_pct > 0 > right.right_pct
    assert right.left_pct == -right.right_pct

    c = ChaseController(cfg)
    left = c.update([ball(100, 400)], W, H, now=0.0)
    assert left.left_pct < 0 < left.right_pct


def test_small_turn_gets_friction_floor(cfg):
    c = ChaseController(cfg)
    c.update([ball(900, 400)], W, H, now=0.0)  # into ALIGN
    # Just above approach_tol: kp*e would be ~5 %, below the floor.
    cmd = c.update([ball(500 + int(0.13 * 500), 400)], W, H, now=0.1)
    assert cmd.state == State.ALIGN
    assert cmd.left_pct == pytest.approx(cfg.min_turn_pct)


def test_centered_ball_drives_forward_with_steering(cfg):
    c = ChaseController(cfg)
    ahead = c.update([ball(500, 300)], W, H, now=0.0)
    assert ahead.state == State.APPROACH
    assert ahead.left_pct == ahead.right_pct == cfg.approach_pct

    slightly_right = c.update([ball(550, 300)], W, H, now=0.1)
    assert slightly_right.state == State.APPROACH
    assert slightly_right.left_pct > slightly_right.right_pct > 0


def test_align_hysteresis(cfg):
    c = ChaseController(cfg)
    c.update([ball(500, 300)], W, H, now=0.0)
    # Drift to e=0.2: inside align_tol, so keep approaching.
    assert c.update([ball(600, 300)], W, H, now=0.1).state == State.APPROACH
    # e=0.3 > align_tol: stop and turn.
    assert c.update([ball(650, 300)], W, H, now=0.2).state == State.ALIGN
    # Back to e=0.2: still above approach_tol, so keep turning.
    assert c.update([ball(600, 300)], W, H, now=0.3).state == State.ALIGN
    assert c.update([ball(520, 300)], W, H, now=0.4).state == State.APPROACH


def test_slows_down_as_ball_gets_close(cfg):
    c = ChaseController(cfg)
    far = c.update([ball(500, 300)], W, H, now=0.0)
    close = c.update([ball(500, 850)], W, H, now=0.1)
    assert close.state == State.APPROACH
    assert cfg.min_approach_pct <= close.left_pct < far.left_pct


def test_never_exceeds_max_pct_and_keeps_turn_ratio():
    cfg = ChaseConfig(max_pct=20.0, approach_pct=30.0)
    c = ChaseController(cfg)
    cmd = c.update([ball(560, 300)], W, H, now=0.0)
    assert max(abs(cmd.left_pct), abs(cmd.right_pct)) == pytest.approx(20.0)
    assert cmd.left_pct > cmd.right_pct


def test_arrives_when_ball_reaches_bottom(cfg):
    c = ChaseController(cfg)
    c.update([ball(500, 300)], W, H, now=0.0)
    cmd = c.update([ball(500, 950)], W, H, now=0.1)
    assert cmd.state == State.ARRIVED
    assert (cmd.left_pct, cmd.right_pct) == (0.0, 0.0)


def test_close_ball_lost_under_camera_counts_as_arrived(cfg):
    c = ChaseController(cfg)
    c.update([ball(500, 300)], W, H, now=0.0)
    last = c.update([ball(500, 800)], W, H, now=0.1)  # near=0.8 >= blind_frac
    # Brief dropout: keep pushing forward.
    blind = c.update([], W, H, now=0.2)
    assert blind.state == State.APPROACH
    assert (blind.left_pct, blind.right_pct) == (last.left_pct, last.right_pct)
    # Gone longer than lost_timeout: it's under the robot.
    assert c.update([], W, H, now=0.1 + cfg.lost_timeout_s + 0.01).state == State.ARRIVED


def test_far_ball_lost_goes_back_to_search_toward_last_side(cfg):
    c = ChaseController(cfg)
    c.update([ball(100, 300)], W, H, now=0.0)  # far, on the left
    cmd = c.update([], W, H, now=cfg.lost_timeout_s + 0.01)
    assert cmd.state == State.SEARCH
    assert cmd.left_pct < 0 < cmd.right_pct


def test_arrived_holds_until_ball_removed_then_searches(cfg):
    c = ChaseController(cfg)
    c.update([ball(500, 950)], W, H, now=0.0)
    assert c.state == State.ARRIVED
    # Ball still sitting there: keep holding, even well past the cooldown.
    assert c.update([ball(500, 950)], W, H, now=5.0).state == State.ARRIVED
    # A far ball appearing doesn't pull us away early.
    assert c.update([ball(500, 200)], W, H, now=5.5).state == State.ARRIVED
    # Close ball gone for cooldown_s: search again.
    assert c.update([], W, H, now=5.0 + cfg.cooldown_s + 0.01).state == State.SEARCH


def test_picks_closest_ball_then_sticks_with_it(cfg):
    c = ChaseController(cfg)
    first = c.update([ball(500, 300), ball(520, 500)], W, H, now=0.0)
    assert first.target.y + first.target.h == 500  # lowest in frame
    # A different ball becomes slightly closer; keep chasing the original.
    second = c.update([ball(510, 520), ball(200, 560)], W, H, now=0.1)
    assert second.target.center[0] == 510
