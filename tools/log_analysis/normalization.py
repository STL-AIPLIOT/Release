# -*- coding: utf-8 -*-
"""end_condition / outcome / BFM 모드명 정규화.

실제 로그에서 확인된 값(2026-08-04):
    end_condition : "ownship altitude below min" (47), "target altitude below min" (10)
    outcome       : "crash" (47), "draw" (10)

termination.py:20-55 가 생성할 수 있는 문자열 전체를 alias 에 미리 넣어 두었다.
로그에 없던 값이 나오면 unknown 으로 분류하고 호출부가 별도 보고할 수 있게 한다.
"""
from __future__ import annotations

# 정규화 유형
ALTITUDE_CHECK_FAILED = "altitude_check_failed"
COLLISION = "collision"
TIMEOUT = "timeout"
WEZ_HIT = "wez_hit"
FDM_FAIL = "fdm_fail"
FUEL = "fuel"
OTHER = "other"
UNKNOWN = "unknown"

END_CONDITION_TYPES = (
    ALTITUDE_CHECK_FAILED, COLLISION, TIMEOUT, WEZ_HIT, FDM_FAIL, FUEL, OTHER, UNKNOWN,
)

# 원본 문자열(소문자) -> 정규화 유형.
# src/dogfight/envs/termination.py:20-55 에서 실제로 생성되는 값을 기준으로 작성했다.
END_CONDITION_ALIASES: dict[str, str] = {
    # 확인됨 — 현재 로그에 실제로 존재
    "ownship altitude below min": ALTITUDE_CHECK_FAILED,
    "target altitude below min": ALTITUDE_CHECK_FAILED,
    # termination.py 가 만들 수 있는 나머지
    "fdm update fail": FDM_FAIL,
    "ownship destroyed": WEZ_HIT,
    "target destroyed": WEZ_HIT,
    "fuel fail": FUEL,
    "two circle headon guard fail": OTHER,
    "max time out": TIMEOUT,
    "episode step limit": TIMEOUT,
}

# 부분 문자열 규칙. 정확히 일치하지 않을 때만 쓴다. 순서가 우선순위다.
_SUBSTRING_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("altitude below min", "altitude_check", "low altitude", "min altitude"), ALTITUDE_CHECK_FAILED),
    (("collision", "collide", "ground impact", "crash into"), COLLISION),
    (("time out", "timeout", "step limit", "draw"), TIMEOUT),
    (("destroyed", "wez", "shot", "damage"), WEZ_HIT),
    (("fdm",), FDM_FAIL),
    (("fuel",), FUEL),
    (("guard fail",), OTHER),
)

OUTCOME_ALIASES: dict[str, str] = {
    "win": "win", "loss": "loss", "lose": "loss", "defeat": "loss",
    "draw": "draw", "tie": "draw",
    "timeout": "timeout", "time_out": "timeout",
    "crash": "crash",
    "other": "other",
}

BFM_ALIASES: dict[str, str] = {
    "obfm": "OBFM", "offensive": "OBFM",
    "dbfm": "DBFM", "defensive": "DBFM",
    "habfm": "HABFM", "high_aspect": "HABFM", "headon": "HABFM",
    "scissors": "SCISSORS", "scissor": "SCISSORS",
    "none": "UNKNOWN", "": "UNKNOWN", "unknown": "UNKNOWN", "detecting": "UNKNOWN",
}


def normalize_end_condition(raw: str | None) -> str:
    """end_condition 문자열을 정규화 유형으로 바꾼다.

    매핑되지 않으면 unknown 을 돌려준다. 호출부는 원본 값을 따로 모아
    unmapped 목록으로 보고해야 한다(임의로 other 로 묻지 않는다).
    """
    if raw is None:
        return UNKNOWN
    key = str(raw).strip().lower()
    if not key:
        return UNKNOWN
    if key in END_CONDITION_ALIASES:
        return END_CONDITION_ALIASES[key]
    for needles, kind in _SUBSTRING_RULES:
        if any(nd in key for nd in needles):
            return kind
    return UNKNOWN


def is_mapped_end_condition(raw: str | None) -> bool:
    """정규화가 실제로 매핑에 걸렸는지. unknown 이면 False."""
    return normalize_end_condition(raw) != UNKNOWN


def normalize_outcome(raw: str | None) -> str:
    """outcome 문자열 정규화. 모르는 값은 원본을 소문자로 그대로 둔다."""
    if raw is None:
        return "unknown"
    key = str(raw).strip().lower()
    return OUTCOME_ALIASES.get(key, key or "unknown")


def normalize_bfm(raw: str | None) -> str:
    """BFM 모드명 정규화. 모르는 값은 UNKNOWN."""
    if raw is None:
        return "UNKNOWN"
    key = str(raw).strip().lower()
    return BFM_ALIASES.get(key, "UNKNOWN")
