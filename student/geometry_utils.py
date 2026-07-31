# -*- coding: utf-8 -*-
"""Sign-independent angle repair for the GeoMathUtil degeneracy.

Platform defect (AIP_LIB/DogFightEnv/Release/src/dogfight/.../GeoMathUtil.py):
    _get_antenna_train_angle() and _get_aspect_angle() build their signed result
    as `sign * magnitude`, where `sign = np.sign(p_unit_t[2])`. When the target
    is at exactly the same altitude the vertical component is 0.0, and
    `np.sign(0.0) == 0.0`, so the WHOLE angle collapses to 0.0 regardless of the
    real geometry. An enemy sitting exactly at our six, co-altitude, is reported
    as 0 deg -- i.e. perfectly nose-on -- and a 1e-9 m altitude offset flips it
    back to +/-180.

    This is reachable at reset(): experiments/*.yaml give both aircraft the same
    `initial_scenario.altitude_m` (7000), and reset() calls get_observation().

GeoMathUtil.py is platform-owned and must not be edited (CLAUDE.md
"Hard constraints" #1), so the repair lives here and is applied inside the
student hooks only.

Scope limit -- read this before trusting the repair:
    dogfight's update_damage() calls the SAME broken helpers and gates damage on
    `|ATA| <= wez["angle_deg"] / 2.0`. Repairing our observation/reward does NOT
    repair that check. The simulator can still award damage for a co-altitude
    dead-astern geometry. We only guarantee that what the agent *observes* and
    what it is *rewarded* for are geometrically truthful.

The functions here are pure and stateless (CLAUDE.md observation gotcha #3):
build_observation() may be called several extra times per step, and with
ownship/target swapped, so nothing may be cached between calls.

State layout used below (verified against FighterSim.py:170-180):
    state[0:3] = North / East / Down,   meters
    state[3:6] = roll / pitch / yaw,    degrees
    state[6:9] = u / v / w body velocity, meters/second (NOT in StateIndex)
Literal indices are used on purpose so this module imports with no `dogfight`
dependency and can therefore be exercised anywhere.
"""
from __future__ import annotations

import math


IDX_NORTH, IDX_EAST, IDX_DOWN = 0, 1, 2
IDX_ROLL, IDX_PITCH, IDX_YAW = 3, 4, 5
IDX_U, IDX_V, IDX_W = 6, 7, 8

# |reported| below this counts as "the platform said zero".
DEGENERATE_REPORT_EPS_DEG = 1e-6
# True magnitude above this counts as "and it was wrong to say zero".
DEGENERATE_TRUE_EPS_DEG = 1e-3

VALID_KINDS = ("ata", "aa")


def _f(value) -> float:
    """float(value) or NaN -- never raises."""
    try:
        out = float(value)
    except (TypeError, ValueError, IndexError):
        return math.nan
    return out


def _nose_vector_ned(state) -> tuple[float, float, float]:
    """Unit body-x (nose) vector of `state` in the NED frame.

    Roll is irrelevant to where the nose points, so only pitch/yaw are used.
    """
    pitch = math.radians(_f(state[IDX_PITCH]))
    yaw = math.radians(_f(state[IDX_YAW]))
    cos_pitch = math.cos(pitch)
    return (cos_pitch * math.cos(yaw), cos_pitch * math.sin(yaw), -math.sin(pitch))


def _line_of_sight_ned(frm, to) -> tuple[float, float, float]:
    """Vector frm -> to in NED meters."""
    return (
        _f(to[IDX_NORTH]) - _f(frm[IDX_NORTH]),
        _f(to[IDX_EAST]) - _f(frm[IDX_EAST]),
        _f(to[IDX_DOWN]) - _f(frm[IDX_DOWN]),
    )


def _angle_between_deg(u, v) -> float:
    """Unsigned angle between two vectors, degrees in [0, 180].

    arccos of the normalized dot product -- no cross product, no vertical
    component, so no sign is involved anywhere. That is the whole point: this
    cannot reproduce the platform's sign-collapse bug.
    """
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    norm_u = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    norm_v = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if not (math.isfinite(dot) and math.isfinite(norm_u) and math.isfinite(norm_v)):
        return math.nan
    if norm_u <= 0.0 or norm_v <= 0.0:  # co-located or degenerate attitude
        return math.nan
    cosine = dot / (norm_u * norm_v)
    cosine = -1.0 if cosine < -1.0 else (1.0 if cosine > 1.0 else cosine)
    return math.degrees(math.acos(cosine))


def true_angle_magnitude_deg(ownship, target, kind: str = "ata") -> float:
    """Sign-independent |angle| in degrees, or NaN if it cannot be computed.

    kind="ata": angle between the ownship nose and the ownship->target LOS.
                0 = target dead ahead (nose-on), 180 = target dead astern.
    kind="aa":  angle between the target's tail direction and the
                target->ownship LOS.
                0 = we are at the target's six, 180 = we are on its nose.
    """
    if kind == "ata":
        return _angle_between_deg(
            _nose_vector_ned(ownship), _line_of_sight_ned(ownship, target)
        )
    if kind == "aa":
        # Tail direction is -nose, so the angle to it is 180 - angle to the nose.
        forward = _angle_between_deg(
            _nose_vector_ned(target), _line_of_sight_ned(target, ownship)
        )
        return math.nan if math.isnan(forward) else 180.0 - forward
    raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")


def repair_degenerate_angle(
    reported, ownship, target, geo_info=None, kind: str = "ata"
) -> float:
    """Return `reported` unless it is the sign-collapse artifact.

    Intervenes only in the degenerate band: |reported| ~ 0 while the true
    magnitude is not ~ 0. Everywhere else `reported` is passed through
    unchanged, sign included, so normal-geometry behaviour is bit-identical to
    the un-repaired hooks.

    On repair the value is returned as +magnitude. At exact dead-astern the
    left/right sign is genuinely undefined (that is the same singularity the
    platform trips over), so a deterministic positive is the only stable
    choice -- an arbitrary sign would make obs[6]/obs[7] jitter between
    identical geometries. A reward hook should use abs() on this value.

    `geo_info` is accepted for call-site symmetry with the platform helpers and
    is deliberately unused: the whole point is to derive the answer without
    going back through the broken code path.
    """
    value = _f(reported)
    if not math.isfinite(value):
        return value  # NaN in, NaN out -- the caller guards for this.
    if abs(value) > DEGENERATE_REPORT_EPS_DEG:
        return value  # non-degenerate: never touch it

    true_magnitude = true_angle_magnitude_deg(ownship, target, kind)
    if not math.isfinite(true_magnitude):
        return value  # cannot do better than the platform here
    if true_magnitude <= DEGENERATE_TRUE_EPS_DEG:
        return value  # genuinely ~0: the platform was right

    return true_magnitude


# The hooks call this under the leading-underscore name used in the design note.
_repair_degenerate_angle = repair_degenerate_angle


# ---------------------------------------------------------------------------
# Velocity / closure
# ---------------------------------------------------------------------------
def body_to_ned_velocity(state) -> tuple[float, float, float]:
    """Rotate state[6:9] (body u/v/w, m/s) into the NED frame.

    3-2-1 (yaw-pitch-roll) convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Consistent with _nose_vector_ned(), which is this matrix's first column.
    """
    roll = math.radians(_f(state[IDX_ROLL]))
    pitch = math.radians(_f(state[IDX_PITCH]))
    yaw = math.radians(_f(state[IDX_YAW]))
    u, v, w = _f(state[IDX_U]), _f(state[IDX_V]), _f(state[IDX_W])

    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    north = u * (cp * cy) + v * (sr * sp * cy - cr * sy) + w * (cr * sp * cy + sr * sy)
    east = u * (cp * sy) + v * (sr * sp * sy + cr * cy) + w * (cr * sp * sy - sr * cy)
    down = u * (-sp) + v * (sr * cp) + w * (cr * cp)
    return (north, east, down)


def closure_rate_mps(ownship, target) -> float:
    """Range rate along the line of sight, m/s. Positive = closing.

    closure = -(v_target - v_ownship) . r_hat,  r = target_pos - ownship_pos

    Derived from the two velocity vectors only -- never from a cached previous
    distance. That is what keeps build_observation() a pure function of its
    arguments, which observation gotcha #3 requires: with an action provider,
    _build_action_context() calls the hook step_ratio extra times per step and
    the target-provider path passes ownship/target swapped, so any
    previous-step cache would read differently in training vs local
    verification. Returns 0.0 when the two aircraft are co-located.
    """
    r_north = _f(target[IDX_NORTH]) - _f(ownship[IDX_NORTH])
    r_east = _f(target[IDX_EAST]) - _f(ownship[IDX_EAST])
    r_down = _f(target[IDX_DOWN]) - _f(ownship[IDX_DOWN])

    range_m = math.sqrt(r_north * r_north + r_east * r_east + r_down * r_down)
    if not math.isfinite(range_m) or range_m <= 0.0:
        return 0.0

    own_v = body_to_ned_velocity(ownship)
    tgt_v = body_to_ned_velocity(target)
    rel = (tgt_v[0] - own_v[0], tgt_v[1] - own_v[1], tgt_v[2] - own_v[2])

    radial = (rel[0] * r_north + rel[1] * r_east + rel[2] * r_down) / range_m
    return -radial if math.isfinite(radial) else 0.0


__all__ = [
    "repair_degenerate_angle",
    "_repair_degenerate_angle",
    "true_angle_magnitude_deg",
    "body_to_ned_velocity",
    "closure_rate_mps",
    "DEGENERATE_REPORT_EPS_DEG",
    "DEGENERATE_TRUE_EPS_DEG",
]
