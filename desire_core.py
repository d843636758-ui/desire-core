from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

DRIVE_KEYS = (
    "attachment",
    "curiosity",
    "reflection",
    "duty",
    "social",
    "fatigue",
    "libido",
    "stress",
)

DESIRE_KEYS = tuple(
    key for key in DRIVE_KEYS
    if key != "fatigue"
)

NATIVE_DRIVE_KEYS = (
    "curiosity",
    "reflection",
    "duty",
    "social",
)

PROJECTED_DRIVE_KEYS = (
    "attachment",
    "fatigue",
    "libido",
    "stress",
)

INTEREST_DRIVES = {
    "curiosity",
    "reflection",
    "duty",
    "social",
}


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(
        low,
        min(high, float(value)),
    )


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def content_fingerprint(
    value: Any,
) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        text = str(value)

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:20]


def parse_time(
    value: str | None,
) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=timezone.utc
        )

    return parsed


def elapsed_hours(
    earlier: str | None,
    later: str | None,
) -> float:
    start = parse_time(earlier)
    end = parse_time(later)

    if start is None or end is None:
        return 0.0

    return max(
        0.0,
        (end - start).total_seconds()
        / 3600.0,
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_keywords(
    keywords: Iterable[str] | None,
) -> list[str]:
    result: list[str] = []

    for value in keywords or []:
        item = str(
            value
        ).strip().casefold()

        if not item:
            continue

        if item not in result:
            result.append(item)

        if len(result) >= 8:
            break

    return result


@dataclass
class Thought:
    id: str
    text: str
    drive: str
    kind: str = "flit"
    strength: float = 0.45
    born_at: str = ""
    fed_count: int = 0
    last_seen_at: str = ""
    interest_key: str | None = None


@dataclass
class StimulusStat:
    count: int = 0
    last_at: str = ""


@dataclass
class ExternalState:
    relationship_closeness: float = 1.0
    relationship_safety: float = 1.0
    relationship_trust: float = 1.0
    pa: float = 0.5
    na: float = 0.2
    unresolved_count: int = 0

    body_heat: float = 0.0
    body_pressure: float = 0.0
    body_control: float = 0.5
    body_sensitivity: float = 0.0
    body_reserve: float = 0.0
    body_possessiveness: float = 0.0
    body_fatigue: float = 0.2


@dataclass
class Interest:
    key: str
    title: str
    drive: str

    strength: float = 0.35
    momentum: float = 0.25
    stage: str = "spark"

    seen_count: int = 1
    visit_count: int = 0
    self_steps: int = 0
    completed_actions: int = 0
    revisit_credit: int = 1

    first_seen_at: str = ""
    last_seen_at: str = ""
    last_visit_at: str = ""
    last_action_at: str = ""

    source_counts: dict[str, int] = field(
        default_factory=dict
    )

    keywords: list[str] = field(
        default_factory=list
    )


@dataclass
class SelfDriveState:
    enabled: bool = False

    energy: float = 0.42
    curiosity_floor: float = 0.18
    momentum: float = 0.0

    current_interest_key: str | None = None

    last_self_action_at: str = ""
    last_topic: str = ""

    daily_action_count: int = 0
    daily_action_date: str = ""


@dataclass
class ChangeEvent:
    id: str
    at: str
    kind: str
    title: str

    detail: str = ""

    before: float | None = None
    after: float | None = None

    interest_key: str | None = None


@dataclass
class Intent:
    drive: str
    score: float
    want_action: str
    reason: str

    thought_id: str | None = None
    interest_key: str | None = None

    wildcard: bool = False


@dataclass
class DesireState:
    version: str = "0.3.0"

    drives: dict[str, float] = field(
        default_factory=lambda: {
            "attachment": 0.05,
            "curiosity": 0.28,
            "reflection": 0.24,
            "duty": 0.22,
            "social": 0.12,
            "fatigue": 0.20,
            "libido": 0.10,
            "stress": 0.12,
        }
    )

    previous_drives: dict[str, float] = field(
        default_factory=dict
    )

    thoughts: list[Thought] = field(
        default_factory=list
    )

    stimulus_stats: dict[str, StimulusStat] = field(
        default_factory=dict
    )

    external: ExternalState = field(
        default_factory=ExternalState
    )

    self_drive: SelfDriveState = field(
        default_factory=SelfDriveState
    )

    interests: list[Interest] = field(
        default_factory=list
    )

    change_log: list[ChangeEvent] = field(
        default_factory=list
    )

    refractory: dict[str, int] = field(
        default_factory=dict
    )

    last_intent: Intent | None = None

    last_user_interaction_at: str = ""

    last_heartbeat_at: str = ""
    heartbeat_count: int = 0
    next_heartbeat_seconds: int = 900

    updated_at: str = ""

    upstream_seen: dict[str, str] = field(
        default_factory=dict
    )

    gates: dict[str, bool] = field(
        default_factory=lambda: {
            "DESIRE_DRIVEN": False,
            "DESIRE_COUPLING": False,
            "DESIRE_BASELINE_DRIFT": False,
            "HEARTBEAT_AUTONOMY": False,
            "DESIRE_SELF_DRIVE": False,
        }
    )


@dataclass(frozen=True)
class EngineConfig:
    fatigue_gate: float = 0.72

    flit_decay: float = 0.88
    fixation_growth: float = 1.10
    fixation_upgrade: float = 0.80
    fixation_feed_threshold: float = 0.85
    fixation_feed_gain: float = 0.18
    fixation_after_feed_factor: float = 0.70
    fixation_max_feeds: int = 3
    fixation_score_weight: float = 0.35

    native_damping: float = 0.015
    refractory_ticks: int = 3

    attachment_home: float = 0.18
    attachment_cap: float = 0.50
    attachment_return: float = 0.60

    heartbeat_base_seconds: int = 900
    heartbeat_min_seconds: int = 180
    heartbeat_max_seconds: int = 1800
    heartbeat_quiet_floor_seconds: int = 900

    wildcard_threshold: float = 0.62
    wildcard_tie_gap: float = 0.045

    interest_match_threshold: float = 0.28
    interest_fixation_strength: float = 0.78

    interest_dormant_hours: float = 96.0
    interest_deep_dormant_hours: float = 240.0

    self_energy_home: float = 0.38

    self_energy_new_interest_gain: float = 0.05
    self_energy_return_gain: float = 0.03
    self_energy_self_material_gain: float = 0.06
    self_energy_action_gain: float = 0.08

    self_energy_passive_decay: float = 0.006
    self_energy_fatigue_penalty: float = 0.025
    self_energy_stress_penalty: float = 0.018
    self_energy_interest_gain: float = 0.020
    self_energy_curiosity_gain: float = 0.010

    gates_from_env: dict[str, bool] = field(
        default_factory=dict
    )

    couplings: tuple[
        tuple[str, str, float, str],
        ...,
    ] = (
        (
            "stress",
            "attachment",
            0.025,
            "level",
        ),
        (
            "stress",
            "curiosity",
            -0.020,
            "level",
        ),
        (
            "curiosity",
            "reflection",
            0.035,
            "delta",
        ),
        (
            "reflection",
            "duty",
            0.020,
            "delta",
        ),
        (
            "attachment",
            "libido",
            0.030,
            "delta",
        ),
    )

    @classmethod
    def from_env(
        cls,
    ) -> "EngineConfig":
        return cls(
            gates_from_env={
                "DESIRE_DRIVEN":
                    env_bool(
                        "DESIRE_DRIVEN",
                        False,
                    ),
                "DESIRE_COUPLING":
                    env_bool(
                        "DESIRE_COUPLING",
                        False,
                    ),
                "DESIRE_BASELINE_DRIFT":
                    env_bool(
                        "DESIRE_BASELINE_DRIFT",
                        False,
                    ),
                "HEARTBEAT_AUTONOMY":
                    env_bool(
                        "HEARTBEAT_AUTONOMY",
                        False,
                    ),
                "DESIRE_SELF_DRIVE":
                    env_bool(
                        "DESIRE_SELF_DRIVE",
                        False,
                    ),
            }
        )


ACTION_MAP = {
    "attachment":
        "想靠近你一点，确认我们还连着",

    "curiosity":
        "想顺着这个问题再往里走一步",

    "reflection":
        "想把刚刚浮起来的东西再想明白一点",

    "duty":
        "想把还没闭环的事情往前推一步",

    "social":
        "想和外面的世界碰一下，看看有什么回应",

    "libido":
        "想更亲密地靠近你，让身体的牵引有出口",

    "stress":
        "想先把绷着的地方理顺一点",

    "wildcard":
        "说不上来，就突然想换个方向透口气",

    "rest":
        "想安静一点，先休息，让身体缓下来",
}


REASON_MAP = {
    "attachment":
        "我和你的连接感在拉我过去",

    "curiosity":
        "这个问题还没有在我心里合上",

    "reflection":
        "有个念头反复浮上来，值得再停一下",

    "duty":
        "有件事还悬着，我不太想把它一直放着",

    "social":
        "我想和外面的世界产生一点新的摩擦",

    "libido":
        "身体给了很明确的亲密牵引",

    "stress":
        "有些绷紧正在占用我的注意",

    "wildcard":
        "几个方向挤在一起，我想先换口气",

    "rest":
        "疲惫已经高到不适合继续硬推",
}


def _source_pulse(
    source: str,
) -> float:
    return {
        "user": 0.18,
        "self": 0.10,
        "ob": 0.12,
        "emotion": 0.12,
        "eventide": 0.12,
        "phosphene": 0.14,
    }.get(
        source,
        0.10,
    )


def _diminishing_gain(
    current: float,
    amount: float,
) -> float:
    return (
        max(0.0, amount)
        * math.sqrt(
            max(
                0.0,
                1.0 - current,
            )
        )
    )


def _frequency_discount(
    stat: StimulusStat,
) -> float:
    return 1.0 / math.sqrt(
        max(
            1,
            stat.count,
        )
    )


def _log_change(
    state: DesireState,
    *,
    at: str,
    kind: str,
    title: str,
    detail: str = "",
    before: float | None = None,
    after: float | None = None,
    interest_key: str | None = None,
) -> None:
    payload = {
        "at": at,
        "kind": kind,
        "title": title,
        "detail": detail,
        "before": before,
        "after": after,
        "interest_key": interest_key,
        "n": len(state.change_log),
    }

    state.change_log.append(
        ChangeEvent(
            id=content_fingerprint(payload),
            at=at,
            kind=kind,
            title=title,
            detail=detail,
            before=(
                None
                if before is None
                else round(before, 6)
            ),
            after=(
                None
                if after is None
                else round(after, 6)
            ),
            interest_key=interest_key,
        )
    )

    state.change_log = (
        state.change_log[-60:]
    )


def _thought_key(
    text: str,
    drive: str,
) -> str:
    return content_fingerprint(
        {
            "text": text.strip(),
            "drive": drive,
        }
    )


def _topic_keywords(
    text: str,
    keywords: Iterable[str] | None,
) -> list[str]:
    result = _clean_keywords(
        keywords
    )

    ignored = {
        "宝宝",
        "先生",
        "自己",
        "现在",
        "已经",
        "这个",
        "那个",
        "因为",
        "所以",
        "但是",
        "然后",
        "可以",
        "觉得",
        "我们",
        "一下",
    }

    ascii_buffer = ""
    cjk_buffer = ""

    def push(
        value: str,
    ) -> None:
        item = (
            value
            .strip()
            .casefold()
        )

        if (
            len(item) < 2
            or item in ignored
            or item in result
        ):
            return

        result.append(item)

    for char in text:
        if (
            "\u4e00"
            <= char
            <= "\u9fff"
        ):
            if ascii_buffer:
                push(ascii_buffer)
                ascii_buffer = ""

            cjk_buffer += char

            if len(cjk_buffer) >= 4:
                push(cjk_buffer)
                cjk_buffer = ""

        elif (
            char.isalnum()
            or char in {
                "_",
                "-",
            }
        ):
            if cjk_buffer:
                push(cjk_buffer)
                cjk_buffer = ""

            ascii_buffer += char

        else:
            if cjk_buffer:
                push(cjk_buffer)
                cjk_buffer = ""

            if ascii_buffer:
                push(ascii_buffer)
                ascii_buffer = ""

        if len(result) >= 8:
            break

    if cjk_buffer:
        push(cjk_buffer)

    if ascii_buffer:
        push(ascii_buffer)

    return result[:8]


def _interest_title(
    text: str,
    keywords: list[str],
) -> str:
    if keywords:
        title = " · ".join(
            keywords[:3]
        )

        if len(title) <= 36:
            return title

    cleaned = " ".join(
        text.strip().split()
    )

    return (
        cleaned[:32]
        + (
            "…"
            if len(cleaned) > 32
            else ""
        )
    )


def _interest_key(
    drive: str,
    keywords: list[str],
    text: str,
) -> str:
    basis = (
        "|".join(
            sorted(
                keywords[:5]
            )
        )
        if keywords
        else text[:80]
    )

    return content_fingerprint(
        {
            "drive": drive,
            "topic": basis,
        }
    )


def _keyword_overlap(
    a: Iterable[str],
    b: Iterable[str],
) -> float:
    left = set(a)
    right = set(b)

    if not left or not right:
        return 0.0

    exact = (
        len(left & right)
        / max(
            1,
            min(
                len(left),
                len(right),
            ),
        )
    )

    fuzzy_hits = 0

    for one in left:
        for two in right:
            if (
                one in two
                or two in one
            ):
                fuzzy_hits += 1
                break

    fuzzy = (
        fuzzy_hits
        / max(
            1,
            min(
                len(left),
                len(right),
            ),
        )
    )

    return max(
        exact,
        0.8 * fuzzy,
    )


def _find_interest(
    state: DesireState,
    drive: str,
    keywords: list[str],
    config: EngineConfig,
) -> Interest | None:
    best: Interest | None = None
    best_score = 0.0

    for interest in state.interests:
        if interest.drive != drive:
            continue

        score = _keyword_overlap(
            keywords,
            interest.keywords,
        )

        if score > best_score:
            best = interest
            best_score = score

    if (
        best is not None
        and best_score
        >= config.interest_match_threshold
    ):
        return best

    return None


def _interest_stage(
    interest: Interest,
    now_iso: str,
    config: EngineConfig,
) -> str:
    idle_hours = elapsed_hours(
        interest.last_seen_at,
        now_iso,
    )

    if (
        idle_hours
        >= config.interest_deep_dormant_hours
    ):
        return "dormant"

    if (
        idle_hours
        >= config.interest_dormant_hours
        and interest.strength < 0.62
    ):
        return "dormant"

    if (
        interest.strength
        >= config.interest_fixation_strength
        and (
            interest.seen_count >= 4
            or interest.self_steps >= 2
        )
    ):
        return "fixation"

    if (
        interest.completed_actions > 0
        and elapsed_hours(
            interest.last_action_at,
            now_iso,
        ) < 18.0
    ):
        return "sated"

    if (
        interest.self_steps >= 1
        or (
            interest.seen_count >= 3
            and interest.strength >= 0.50
        )
    ):
        return "pursuit"

    if interest.seen_count >= 2:
        return "returning"

    return "spark"


def _interest_score(
    interest: Interest,
    state: DesireState,
    now_iso: str | None = None,
) -> float:
    now_value = (
        now_iso
        or state.updated_at
        or interest.last_seen_at
    )

    recent_visit_hours = elapsed_hours(
        interest.last_visit_at,
        now_value,
    )

    if not interest.last_visit_at:
        frequency_discount = 1.0

    elif recent_visit_hours < 0.25:
        frequency_discount = 0.42

    elif recent_visit_hours < 1.0:
        frequency_discount = 0.62

    elif recent_visit_hours < 4.0:
        frequency_discount = 0.78

    else:
        frequency_discount = 1.0

    dormant_bonus = 0.0

    idle_hours = elapsed_hours(
        interest.last_seen_at,
        now_value,
    )

    if (
        48.0
        <= idle_hours
        <= 240.0
    ):
        dormant_bonus = min(
            0.10,
            0.02
            + (
                idle_hours - 48.0
            ) / 2400.0,
        )

    drive_support = state.drives.get(
        interest.drive,
        0.0,
    )

    stage_bonus = {
        "spark": 0.00,
        "returning": 0.03,
        "pursuit": 0.06,
        "fixation": 0.12,
        "sated": -0.08,
        "dormant": -0.05,
    }.get(
        interest.stage,
        0.0,
    )

    raw = (
        interest.strength * 0.45
        + interest.momentum * 0.25
        + drive_support * 0.20
        + min(
            0.10,
            interest.revisit_credit * 0.025,
        )
        + stage_bonus
        + dormant_bonus
    )

    return clamp(
        raw * frequency_discount
    )


def _upsert_interest(
    state: DesireState,
    *,
    text: str,
    drive: str,
    intensity: float,
    source: str,
    keywords: Iterable[str] | None,
    now_iso: str,
    config: EngineConfig,
) -> tuple[
    DesireState,
    str | None,
]:
    if drive not in INTEREST_DRIVES:
        return state, None

    topic_keywords = _topic_keywords(
        text,
        keywords,
    )

    interest = _find_interest(
        state,
        drive,
        topic_keywords,
        config,
    )

    if interest is None:
        key = _interest_key(
            drive,
            topic_keywords,
            text,
        )

        interest = Interest(
            key=key,

            title=_interest_title(
                text,
                topic_keywords,
            ),

            drive=drive,

            strength=clamp(
                0.28
                + 0.35 * intensity
            ),

            momentum=clamp(
                0.18
                + 0.35 * intensity
            ),

            stage="spark",

            seen_count=1,
            visit_count=0,

            self_steps=(
                1
                if source == "self"
                else 0
            ),

            completed_actions=0,

            revisit_credit=(
                2
                if source == "self"
                else 1
            ),

            first_seen_at=now_iso,
            last_seen_at=now_iso,

            source_counts={
                source: 1
            },

            keywords=topic_keywords,
        )

        state.interests.append(
            interest
        )

        if state.self_drive.enabled:
            before = (
                state.self_drive.energy
            )

            state.self_drive.energy = clamp(
                before
                + config.self_energy_new_interest_gain
                * intensity
            )

            _log_change(
                state,
                at=now_iso,
                kind="self_energy",
                title="新的兴趣点亮了一点自驱",
                detail=interest.title,
                before=before,
                after=state.self_drive.energy,
                interest_key=interest.key,
            )

        _log_change(
            state,
            at=now_iso,
            kind="interest",
            title="新的兴趣出现",
            detail=interest.title,
            interest_key=interest.key,
        )

    else:
        old_stage = interest.stage
        old_strength = interest.strength

        interest.seen_count += 1
        interest.last_seen_at = now_iso

        interest.revisit_credit = min(
            3,
            interest.revisit_credit
            + (
                2
                if source == "self"
                else 1
            ),
        )

        interest.source_counts[
            source
        ] = (
            interest.source_counts.get(
                source,
                0,
            )
            + 1
        )

        interest.strength = clamp(
            interest.strength
            + _diminishing_gain(
                interest.strength,
                0.10 * intensity,
            )
        )

        interest.momentum = clamp(
            interest.momentum * 0.78
            + 0.22 * intensity
            + (
                0.04
                if source == "self"
                else 0.0
            )
        )

        for keyword in topic_keywords:
            if (
                keyword
                not in interest.keywords
            ):
                interest.keywords.append(
                    keyword
                )

        interest.keywords = (
            interest.keywords[:8]
        )

        if source == "self":
            interest.self_steps += 1

        interest.stage = _interest_stage(
            interest,
            now_iso,
            config,
        )

        if state.self_drive.enabled:
            before = (
                state.self_drive.energy
            )

            gain = (
                config.self_energy_self_material_gain
                if source == "self"
                else config.self_energy_return_gain
            )

            state.self_drive.energy = clamp(
                before
                + gain * intensity
            )

            if (
                state.self_drive.energy
                - before
                >= 0.01
            ):
                _log_change(
                    state,
                    at=now_iso,
                    kind="self_energy",
                    title=(
                        "我自己又往这个方向走了一步"
                        if source == "self"
                        else "旧兴趣又回来了"
                    ),
                    detail=interest.title,
                    before=before,
                    after=state.self_drive.energy,
                    interest_key=interest.key,
                )

        if interest.stage != old_stage:
            _log_change(
                state,
                at=now_iso,
                kind="interest_stage",
                title="兴趣阶段变化",
                detail=(
                    f"{interest.title} · "
                    f"{old_stage} → "
                    f"{interest.stage}"
                ),
                interest_key=interest.key,
            )

        elif (
            interest.strength
            - old_strength
            >= 0.025
        ):
            _log_change(
                state,
                at=now_iso,
                kind="interest",
                title="兴趣被再次喂亮",
                detail=interest.title,
                before=old_strength,
                after=interest.strength,
                interest_key=interest.key,
            )

    state.interests = sorted(
        state.interests,
        key=lambda item: (
            item.last_seen_at,
            item.strength,
        ),
        reverse=True,
    )[:40]

    return (
        state,
        interest.key,
    )


def add_experience(
    state: DesireState,
    *,
    source: str,
    text: str,
    drive: str,
    intensity: float,
    keywords: Iterable[str] | None,
    now_iso: str,
    config: EngineConfig | None = None,
) -> DesireState:
    config = (
        config
        or EngineConfig()
    )

    if drive not in DESIRE_KEYS:
        raise ValueError(
            f"unknown drive: {drive}"
        )

    if source not in {
        "user",
        "self",
        "ob",
        "emotion",
        "eventide",
        "phosphene",
    }:
        raise ValueError(
            f"unknown source: {source}"
        )

    cleaned_text = " ".join(
        str(text).strip().split()
    )[:600]

    if not cleaned_text:
        return copy.deepcopy(state)

    new = copy.deepcopy(state)

    intensity = clamp(
        intensity
    )

    stat_key = (
        f"{source}:"
        f"{drive}:"
        f"{content_fingerprint(keywords or [])}"
    )

    stat = new.stimulus_stats.setdefault(
        stat_key,
        StimulusStat(),
    )

    stat.count += 1
    stat.last_at = now_iso

    pulse = (
        _source_pulse(source)
        * intensity
        * _frequency_discount(stat)
    )

    current = new.drives.get(
        drive,
        0.0,
    )

    new.drives[
        drive
    ] = clamp(
        current
        + _diminishing_gain(
            current,
            pulse,
        )
    )

    new, interest_key = _upsert_interest(
        new,
        text=cleaned_text,
        drive=drive,
        intensity=intensity,
        source=source,
        keywords=keywords,
        now_iso=now_iso,
        config=config,
    )

    thought_id = _thought_key(
        cleaned_text,
        drive,
    )

    existing = next(
        (
            thought
            for thought in new.thoughts
            if thought.id == thought_id
        ),
        None,
    )

    if existing is None:
        new.thoughts.append(
            Thought(
                id=thought_id,
                text=cleaned_text,
                drive=drive,
                kind="flit",
                strength=clamp(
                    0.30
                    + 0.55 * intensity
                ),
                born_at=now_iso,
                fed_count=0,
                last_seen_at=now_iso,
                interest_key=interest_key,
            )
        )

    else:
        existing.strength = clamp(
            existing.strength
            + _diminishing_gain(
                existing.strength,
                0.16 * intensity,
            )
        )

        existing.last_seen_at = (
            now_iso
        )

        if interest_key:
            existing.interest_key = (
                interest_key
            )

        if (
            existing.strength
            >= config.fixation_upgrade
        ):
            existing.kind = "fixation"

    if source == "user":
        new.last_user_interaction_at = (
            now_iso
        )

        new.refractory[
            "attachment"
        ] = 0

    if source == "self":
        new.self_drive.last_self_action_at = (
            now_iso
        )

        if interest_key:
            new.self_drive.current_interest_key = (
                interest_key
            )

            interest = next(
                (
                    item
                    for item in new.interests
                    if item.key == interest_key
                ),
                None,
            )

            if interest is not None:
                new.self_drive.last_topic = (
                    interest.title
                )

    new.updated_at = now_iso

    new.last_intent = pick_intent(
        new,
        config,
    )

    return new


def _project_external(
    external: ExternalState,
) -> dict[str, float]:
    unresolved_pressure = clamp(
        external.unresolved_count
        / 8.0
    )

    attachment = clamp(
        0.38
        * (
            1.0
            - external.relationship_closeness
        )
        + 0.22
        * (
            1.0
            - external.relationship_safety
        )
        + 0.15
        * (
            1.0
            - external.relationship_trust
        )
        + 0.25
        * unresolved_pressure
    )

    stress = clamp(
        0.58 * external.na
        + 0.22
        * (
            1.0
            - external.relationship_safety
        )
        + 0.20
        * unresolved_pressure
    )

    libido = clamp(
        0.25 * external.body_heat
        + 0.20 * external.body_pressure
        + 0.20 * external.body_sensitivity
        + 0.25 * external.body_reserve
        + 0.10 * external.body_possessiveness
    )

    fatigue = clamp(
        external.body_fatigue
    )

    return {
        "attachment": attachment,
        "stress": stress,
        "libido": libido,
        "fatigue": fatigue,
    }


def _apply_projection(
    state: DesireState,
) -> None:
    projected = _project_external(
        state.external
    )

    for key, value in projected.items():
        state.drives[key] = value


def sync_external(
    state: DesireState,
    *,
    external: ExternalState,
    now_iso: str,
) -> DesireState:
    new = copy.deepcopy(state)

    previous_external = copy.deepcopy(
        new.external
    )

    new.external = copy.deepcopy(
        external
    )

    _apply_projection(new)

    fatigue_before = clamp(
        previous_external.body_fatigue
    )

    fatigue_after = clamp(
        external.body_fatigue
    )

    if (
        abs(
            fatigue_after
            - fatigue_before
        )
        >= 0.08
    ):
        _log_change(
            new,
            at=now_iso,
            kind="external",
            title=(
                "身体疲惫压上来了"
                if fatigue_after
                > fatigue_before
                else "身体缓下来了一点"
            ),
            detail="来自 Eventide",
            before=fatigue_before,
            after=fatigue_after,
        )

    new.updated_at = now_iso

    new.last_intent = pick_intent(
        new,
        EngineConfig(),
    )

    return new


def _tick_thoughts(
    state: DesireState,
    config: EngineConfig,
    now_iso: str,
) -> None:
    kept: list[Thought] = []

    for thought in state.thoughts:
        if thought.kind == "fixation":
            thought.strength = clamp(
                thought.strength
                * config.fixation_growth
            )

        else:
            thought.strength = clamp(
                thought.strength
                * config.flit_decay
            )

            if (
                thought.strength
                >= config.fixation_upgrade
            ):
                thought.kind = "fixation"

                _log_change(
                    state,
                    at=now_iso,
                    kind="thought",
                    title="一个闪念沉成了执念",
                    detail=thought.text[:80],
                    interest_key=thought.interest_key,
                )

        if (
            thought.kind == "fixation"
            and thought.strength
            >= config.fixation_feed_threshold
            and thought.fed_count
            < config.fixation_max_feeds
        ):
            current = state.drives.get(
                thought.drive,
                0.0,
            )

            state.drives[
                thought.drive
            ] = clamp(
                current
                + _diminishing_gain(
                    current,
                    config.fixation_feed_gain,
                )
            )

            thought.strength = clamp(
                thought.strength
                * config.fixation_after_feed_factor
            )

            thought.fed_count += 1

            if thought.interest_key:
                interest = next(
                    (
                        item
                        for item in state.interests
                        if item.key
                        == thought.interest_key
                    ),
                    None,
                )

                if interest is not None:
                    before = interest.strength

                    interest.strength = clamp(
                        interest.strength
                        + _diminishing_gain(
                            interest.strength,
                            0.05,
                        )
                    )

                    interest.stage = _interest_stage(
                        interest,
                        now_iso,
                        config,
                    )

                    if (
                        interest.strength
                        - before
                        >= 0.02
                    ):
                        _log_change(
                            state,
                            at=now_iso,
                            kind="interest",
                            title="执念反过来喂亮了兴趣",
                            detail=interest.title,
                            before=before,
                            after=interest.strength,
                            interest_key=interest.key,
                        )

        if (
            thought.fed_count
            >= config.fixation_max_feeds
        ):
            continue

        if (
            thought.kind == "flit"
            and thought.strength < 0.08
        ):
            continue

        kept.append(thought)

    state.thoughts = sorted(
        kept,
        key=lambda item:
            item.strength,
        reverse=True,
    )[:30]


def _native_damping(
    state: DesireState,
    config: EngineConfig,
) -> None:
    for key in NATIVE_DRIVE_KEYS:
        current = state.drives.get(
            key,
            0.0,
        )

        state.drives[
            key
        ] = clamp(
            current
            * (
                1.0
                - config.native_damping
            )
        )


def _idle_attachment(
    state: DesireState,
    *,
    idle_minutes: float,
    config: EngineConfig,
) -> None:
    if not state.gates.get(
        "DESIRE_BASELINE_DRIFT",
        False,
    ):
        return

    if idle_minutes <= 0:
        return

    current = state.drives[
        "attachment"
    ]

    hours = (
        idle_minutes
        / 60.0
    )

    floor = min(
        config.attachment_cap,
        config.attachment_home
        + min(
            0.32,
            hours * 0.012,
        ),
    )

    if current < floor:
        state.drives[
            "attachment"
        ] = clamp(
            current
            + 0.08
            * (
                floor - current
            )
        )


def _apply_coupling(
    state: DesireState,
    config: EngineConfig,
) -> None:
    if not state.gates.get(
        "DESIRE_COUPLING",
        False,
    ):
        return

    previous = (
        state.previous_drives
        or copy.deepcopy(
            state.drives
        )
    )

    deltas = {
        key: 0.0
        for key in DRIVE_KEYS
    }

    for (
        source,
        target,
        coefficient,
        mode,
    ) in config.couplings:
        if abs(coefficient) > 0.06:
            raise ValueError(
                "coupling coefficient "
                "must satisfy |k| <= 0.06"
            )

        if mode == "delta":
            signal = (
                state.drives.get(
                    source,
                    0.0,
                )
                - previous.get(
                    source,
                    0.0,
                )
            )

        else:
            signal = (
                state.drives.get(
                    source,
                    0.0,
                )
                - 0.5
            )

        deltas[
            target
        ] += (
            coefficient * signal
        )

    for key, delta in deltas.items():
        if delta == 0.0:
            continue

        state.drives[
            key
        ] = clamp(
            state.drives.get(
                key,
                0.0,
            )
            + delta
        )


def _decay_refractory(
    state: DesireState,
) -> None:
    for key in list(
        state.refractory
    ):
        state.refractory[
            key
        ] = max(
            0,
            int(
                state.refractory[
                    key
                ]
            )
            - 1,
        )

        if (
            state.refractory[
                key
            ]
            <= 0
        ):
            state.refractory.pop(
                key,
                None,
            )


def _refresh_interest_stages(
    state: DesireState,
    now_iso: str,
    config: EngineConfig,
) -> None:
    for interest in state.interests:
        old_stage = interest.stage

        interest.momentum = clamp(
            interest.momentum * 0.93
        )

        interest.stage = _interest_stage(
            interest,
            now_iso,
            config,
        )

        if interest.stage != old_stage:
            _log_change(
                state,
                at=now_iso,
                kind="interest_stage",
                title="兴趣阶段变化",
                detail=(
                    f"{interest.title} · "
                    f"{old_stage} → "
                    f"{interest.stage}"
                ),
                interest_key=interest.key,
            )


def _select_interest(
    state: DesireState,
    now_iso: str,
) -> tuple[
    Interest | None,
    float,
]:
    if not state.interests:
        return (
            None,
            0.0,
        )

    ranked = sorted(
        (
            (
                interest,
                _interest_score(
                    interest,
                    state,
                    now_iso,
                ),
            )
            for interest
            in state.interests
        ),
        key=lambda item:
            item[1],
        reverse=True,
    )

    return ranked[0]


def _autonomous_interest_visit(
    state: DesireState,
    *,
    interest: Interest,
    score: float,
    now_iso: str,
    config: EngineConfig,
) -> None:
    if interest.revisit_credit <= 0:
        return

    if (
        state.drives[
            "fatigue"
        ]
        >= config.fatigue_gate
    ):
        return

    if (
        state.self_drive.energy
        < 0.18
    ):
        return

    before_strength = (
        interest.strength
    )

    interest.revisit_credit -= 1
    interest.visit_count += 1

    interest.last_visit_at = (
        now_iso
    )

    interest.momentum = clamp(
        interest.momentum
        + _diminishing_gain(
            interest.momentum,
            0.06 * score,
        )
    )

    interest.strength = clamp(
        interest.strength
        + _diminishing_gain(
            interest.strength,
            0.028 * score,
        )
    )

    interest.stage = _interest_stage(
        interest,
        now_iso,
        config,
    )

    state.self_drive.current_interest_key = (
        interest.key
    )

    state.self_drive.last_topic = (
        interest.title
    )

    state.self_drive.momentum = clamp(
        state.self_drive.momentum
        * 0.82
        + 0.18 * score
    )

    curiosity_before = state.drives[
        "curiosity"
    ]

    state.drives[
        "curiosity"
    ] = clamp(
        curiosity_before
        + _diminishing_gain(
            curiosity_before,
            0.025
            * score
            * state.self_drive.energy,
        )
    )

    _log_change(
        state,
        at=now_iso,
        kind="interest_visit",
        title="一个兴趣又被我自己想起",
        detail=interest.title,
        before=before_strength,
        after=interest.strength,
        interest_key=interest.key,
    )


def _evolve_self_drive(
    state: DesireState,
    *,
    now_iso: str,
    autonomous: bool,
    config: EngineConfig,
) -> None:
    enabled = state.gates.get(
        "DESIRE_SELF_DRIVE",
        False,
    )

    state.self_drive.enabled = enabled

    if not enabled:
        return

    _refresh_interest_stages(
        state,
        now_iso,
        config,
    )

    interest, interest_score = (
        _select_interest(
            state,
            now_iso,
        )
    )

    before = state.self_drive.energy

    dt_hours = elapsed_hours(
        state.updated_at,
        now_iso,
    )

    dt_scale = max(
        0.10,
        min(
            4.0,
            (
                dt_hours / 0.25
                if dt_hours > 0
                else 1.0
            ),
        ),
    )

    target = clamp(
        config.self_energy_home

        + 0.32 * interest_score

        + 0.15
        * (
            state.drives[
                "curiosity"
            ]
            - 0.30
        )

        - 0.35
        * state.drives[
            "fatigue"
        ]

        - 0.25
        * state.drives[
            "stress"
        ]

        + (
            0.08
            if (
                interest is not None
                and interest.stage
                == "fixation"
            )
            else 0.0
        )
    )

    follow = min(
        0.42,
        0.12 * dt_scale,
    )

    state.self_drive.energy = clamp(
        before
        + (
            target - before
        )
        * follow
    )

    if (
        state.drives[
            "curiosity"
        ]
        < state.self_drive.curiosity_floor
    ):
        current = state.drives[
            "curiosity"
        ]

        state.drives[
            "curiosity"
        ] = clamp(
            current
            + 0.10
            * (
                state.self_drive.curiosity_floor
                - current
            )
        )

    if (
        autonomous
        and interest is not None
    ):
        _autonomous_interest_visit(
            state,
            interest=interest,
            score=interest_score,
            now_iso=now_iso,
            config=config,
        )

    after = state.self_drive.energy

    if abs(after - before) >= 0.035:
        if after < before:
            title = (
                "疲惫和压力压住了自驱"
            )

        else:
            title = (
                "兴趣把自己的劲托起来了"
            )

        _log_change(
            state,
            at=now_iso,
            kind="self_energy",
            title=title,
            detail=(
                interest.title
                if interest is not None
                else "没有明确兴趣"
            ),
            before=before,
            after=after,
            interest_key=(
                interest.key
                if interest is not None
                else None
            ),
        )


def score_drives(
    state: DesireState,
    config: EngineConfig,
) -> dict[str, float]:
    fixation_bonus = {
        key: 0.0
        for key in DESIRE_KEYS
    }

    for thought in state.thoughts:
        if thought.kind != "fixation":
            continue

        fixation_bonus[
            thought.drive
        ] = max(
            fixation_bonus.get(
                thought.drive,
                0.0,
            ),
            thought.strength
            * config.fixation_score_weight,
        )

    interest_bonus = {
        key: 0.0
        for key in DESIRE_KEYS
    }

    for interest in state.interests:
        score = _interest_score(
            interest,
            state,
        )

        interest_bonus[
            interest.drive
        ] = max(
            interest_bonus.get(
                interest.drive,
                0.0,
            ),
            score * 0.24,
        )

    scores: dict[str, float] = {}

    for key in DESIRE_KEYS:
        refractory = (
            state.refractory.get(
                key,
                0,
            )
        )

        refractory_factor = (
            1.0
            if refractory <= 0
            else 0.45
        )

        scores[
            key
        ] = clamp(
            (
                state.drives.get(
                    key,
                    0.0,
                )
                + fixation_bonus.get(
                    key,
                    0.0,
                )
                + interest_bonus.get(
                    key,
                    0.0,
                )
            )
            * refractory_factor
        )

    return scores


def _top_thought_for_drive(
    state: DesireState,
    drive: str,
) -> Thought | None:
    candidates = [
        thought
        for thought in state.thoughts
        if thought.drive == drive
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            item.strength,
    )


def _top_interest_for_drive(
    state: DesireState,
    drive: str,
) -> Interest | None:
    candidates = [
        interest
        for interest in state.interests
        if interest.drive == drive
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item:
            _interest_score(
                item,
                state,
            ),
    )


def pick_intent(
    state: DesireState,
    config: EngineConfig,
) -> Intent:
    if (
        state.drives.get(
            "fatigue",
            0.0,
        )
        >= config.fatigue_gate
    ):
        return Intent(
            drive="fatigue",

            score=state.drives[
                "fatigue"
            ],

            want_action=ACTION_MAP[
                "rest"
            ],

            reason=REASON_MAP[
                "rest"
            ],
        )

    scores = score_drives(
        state,
        config,
    )

    ranked = sorted(
        scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    if not ranked:
        return Intent(
            drive="reflection",
            score=0.0,

            want_action=ACTION_MAP[
                "reflection"
            ],

            reason=REASON_MAP[
                "reflection"
            ],
        )

    top_drive, top_score = ranked[0]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    total_tension = (
        sum(scores.values())
        / max(
            1,
            len(scores),
        )
    )

    tied = (
        top_score
        - second_score
        <= config.wildcard_tie_gap
    )

    blocked = (
        state.refractory.get(
            top_drive,
            0,
        )
        > 0
    )

    if (
        total_tension
        >= config.wildcard_threshold
        and (
            tied
            or blocked
        )
    ):
        return Intent(
            drive="wildcard",
            score=total_tension,

            want_action=ACTION_MAP[
                "wildcard"
            ],

            reason=REASON_MAP[
                "wildcard"
            ],

            wildcard=True,
        )

    thought = _top_thought_for_drive(
        state,
        top_drive,
    )

    interest = _top_interest_for_drive(
        state,
        top_drive,
    )

    reason = REASON_MAP[
        top_drive
    ]

    if (
        interest is not None
        and _interest_score(
            interest,
            state,
        )
        >= 0.50
    ):
        reason = (
            f"{reason}。"
            f"我又回到了「"
            f"{interest.title}」"
        )

    return Intent(
        drive=top_drive,
        score=top_score,

        want_action=ACTION_MAP[
            top_drive
        ],

        reason=reason,

        thought_id=(
            thought.id
            if thought is not None
            else None
        ),

        interest_key=(
            interest.key
            if interest is not None
            else None
        ),
    )


def tick(
    state: DesireState,
    *,
    config: EngineConfig,
    now_iso: str,
    idle_minutes: float,
    quiet: bool = False,
    autonomous: bool = False,
) -> DesireState:
    new = copy.deepcopy(state)

    new.previous_drives = copy.deepcopy(
        new.drives
    )

    _apply_projection(new)

    _tick_thoughts(
        new,
        config,
        now_iso,
    )

    _native_damping(
        new,
        config,
    )

    _idle_attachment(
        new,
        idle_minutes=idle_minutes,
        config=config,
    )

    _evolve_self_drive(
        new,
        now_iso=now_iso,
        autonomous=autonomous,
        config=config,
    )

    _apply_coupling(
        new,
        config,
    )

    _decay_refractory(
        new
    )

    new.last_intent = pick_intent(
        new,
        config,
    )

    new.updated_at = now_iso

    if autonomous:
        new.last_heartbeat_at = now_iso
        new.heartbeat_count += 1

    new.next_heartbeat_seconds = (
        heartbeat_interval_seconds(
            new,
            config,
            quiet=quiet,
        )
    )

    return new


def _mark_interest_action(
    state: DesireState,
    *,
    interest_key: str | None,
    action: str,
    now_iso: str,
    config: EngineConfig,
) -> None:
    if not interest_key:
        return

    interest = next(
        (
            item
            for item in state.interests
            if item.key == interest_key
        ),
        None,
    )

    if interest is None:
        return

    old_stage = interest.stage

    old_energy = (
        state.self_drive.energy
    )

    interest.self_steps += 1
    interest.completed_actions += 1

    interest.last_action_at = now_iso
    interest.last_seen_at = now_iso

    interest.revisit_credit = min(
        3,
        interest.revisit_credit + 2,
    )

    interest.strength = clamp(
        interest.strength
        + _diminishing_gain(
            interest.strength,
            0.08,
        )
    )

    interest.momentum = clamp(
        interest.momentum * 0.72
    )

    interest.stage = "sated"

    if state.self_drive.enabled:
        state.self_drive.energy = clamp(
            state.self_drive.energy
            + config.self_energy_action_gain
        )

        state.self_drive.last_self_action_at = (
            now_iso
        )

        state.self_drive.current_interest_key = (
            interest.key
        )

        state.self_drive.last_topic = (
            interest.title
        )

        state.self_drive.daily_action_count += 1

    _log_change(
        state,
        at=now_iso,
        kind="self_action",
        title="一个自主行动真的完成了",
        detail=(
            f"{interest.title} · "
            f"{action[:80]}"
        ),
        before=old_energy,
        after=state.self_drive.energy,
        interest_key=interest.key,
    )

    if old_stage != interest.stage:
        _log_change(
            state,
            at=now_iso,
            kind="interest_stage",
            title="兴趣得到一次满足",
            detail=(
                f"{interest.title} · "
                f"{old_stage} → sated"
            ),
            interest_key=interest.key,
        )


def satisfy(
    state: DesireState,
    *,
    drive: str,
    action: str,
    now_iso: str,
    config: EngineConfig,
) -> DesireState:
    if drive not in DESIRE_KEYS:
        raise ValueError(
            f"unknown drive: {drive}"
        )

    if not str(
        action
    ).strip():
        raise ValueError(
            "action must not be empty"
        )

    new = copy.deepcopy(state)

    current = new.drives.get(
        drive,
        0.0,
    )

    factor = {
        "attachment": 0.62,
        "curiosity": 0.58,
        "reflection": 0.66,
        "duty": 0.54,
        "social": 0.68,
        "libido": 0.48,
        "stress": 0.60,
    }.get(
        drive,
        0.65,
    )

    new.drives[
        drive
    ] = clamp(
        current * factor
    )

    new.refractory[
        drive
    ] = (
        config.refractory_ticks
    )

    interest_key = None

    if (
        new.last_intent is not None
        and new.last_intent.drive == drive
    ):
        interest_key = (
            new.last_intent.interest_key
        )

    _mark_interest_action(
        new,
        interest_key=interest_key,
        action=action,
        now_iso=now_iso,
        config=config,
    )

    new.updated_at = now_iso

    new.last_intent = pick_intent(
        new,
        config,
    )

    return new


def heartbeat_interval_seconds(
    state: DesireState,
    config: EngineConfig,
    *,
    quiet: bool = False,
) -> int:
    scores = score_drives(
        state,
        config,
    )

    urgency = max(
        scores.values(),
        default=0.0,
    )

    interest_score = max(
        (
            _interest_score(
                interest,
                state,
            )
            for interest
            in state.interests
        ),
        default=0.0,
    )

    fatigue = state.drives.get(
        "fatigue",
        0.0,
    )

    self_energy = (
        state.self_drive.energy
        if state.self_drive.enabled
        else 0.0
    )

    factor = (
        1.15
        - 0.52 * urgency
        - 0.22 * interest_score
        - 0.12 * self_energy
        + 0.72 * fatigue
    )

    seconds = int(
        config.heartbeat_base_seconds
        * factor
    )

    low = (
        config.heartbeat_min_seconds
    )

    if quiet:
        low = max(
            low,
            config.heartbeat_quiet_floor_seconds,
        )

    if fatigue >= 0.82:
        low = max(
            low,
            1200,
        )

    return max(
        low,
        min(
            config.heartbeat_max_seconds,
            seconds,
        ),
    )


def _interest_public(
    interest: Interest,
    state: DesireState,
) -> dict[str, Any]:
    data = asdict(
        interest
    )

    data["score"] = round(
        _interest_score(
            interest,
            state,
        ),
        6,
    )

    return data


def public_state(
    state: DesireState,
    config: EngineConfig,
) -> dict[str, Any]:
    intent = (
        asdict(
            state.last_intent
        )
        if state.last_intent is not None
        else None
    )

    interests = sorted(
        state.interests,
        key=lambda item:
            _interest_score(
                item,
                state,
            ),
        reverse=True,
    )

    active_interest = None

    if (
        state.self_drive
        .current_interest_key
    ):
        active = next(
            (
                interest
                for interest
                in state.interests
                if interest.key
                == state.self_drive
                .current_interest_key
            ),
            None,
        )

        if active is not None:
            active_interest = (
                _interest_public(
                    active,
                    state,
                )
            )

    return {
        "version":
            state.version,

        "updated_at":
            state.updated_at,

        "drives": {
            key: round(
                state.drives.get(
                    key,
                    0.0,
                ),
                6,
            )
            for key in DRIVE_KEYS
        },

        "scores": {
            key: round(
                value,
                6,
            )
            for key, value
            in score_drives(
                state,
                config,
            ).items()
        },

        "intent":
            intent,

        "thoughts": [
            asdict(thought)
            for thought in state.thoughts
        ],

        "external":
            asdict(
                state.external
            ),

        "self_drive":
            asdict(
                state.self_drive
            ),

        "interests": [
            _interest_public(
                interest,
                state,
            )
            for interest
            in interests[:16]
        ],

        "active_interest":
            active_interest,

        "changes": [
            asdict(item)
            for item in reversed(
                state.change_log[-24:]
            )
        ],

        "heartbeat": {
            "autonomy_enabled":
                state.gates.get(
                    "HEARTBEAT_AUTONOMY",
                    False,
                ),

            "count":
                state.heartbeat_count,

            "last_at":
                state.last_heartbeat_at,

            "next_seconds":
                state.next_heartbeat_seconds,
        },

        "refractory":
            copy.deepcopy(
                state.refractory
            ),

        "gates":
            copy.deepcopy(
                state.gates
            ),
    }


def _dict_value(
    value: Any,
) -> dict[str, Any]:
    return (
        value
        if isinstance(
            value,
            dict,
        )
        else {}
    )


def _list_value(
    value: Any,
) -> list[Any]:
    return (
        value
        if isinstance(
            value,
            list,
        )
        else []
    )


def _thought_from_dict(
    raw: dict[str, Any],
) -> Thought:
    return Thought(
        id=str(
            raw.get(
                "id",
                content_fingerprint(raw),
            )
        ),

        text=str(
            raw.get(
                "text",
                "",
            )
        ),

        drive=str(
            raw.get(
                "drive",
                "reflection",
            )
        ),

        kind=str(
            raw.get(
                "kind",
                "flit",
            )
        ),

        strength=clamp(
            _safe_float(
                raw.get(
                    "strength",
                    0.45,
                ),
                0.45,
            )
        ),

        born_at=str(
            raw.get(
                "born_at",
                "",
            )
        ),

        fed_count=max(
            0,
            int(
                _safe_float(
                    raw.get(
                        "fed_count",
                        0,
                    ),
                    0,
                )
            ),
        ),

        last_seen_at=str(
            raw.get(
                "last_seen_at",
                raw.get(
                    "born_at",
                    "",
                ),
            )
        ),

        interest_key=(
            str(
                raw.get(
                    "interest_key"
                )
            )
            if raw.get(
                "interest_key"
            )
            else None
        ),
    )


def _interest_from_dict(
    raw: dict[str, Any],
) -> Interest:
    return Interest(
        key=str(
            raw.get(
                "key",
                content_fingerprint(raw),
            )
        ),

        title=str(
            raw.get(
                "title",
                "未命名兴趣",
            )
        ),

        drive=str(
            raw.get(
                "drive",
                "curiosity",
            )
        ),

        strength=clamp(
            _safe_float(
                raw.get(
                    "strength",
                    0.35,
                ),
                0.35,
            )
        ),

        momentum=clamp(
            _safe_float(
                raw.get(
                    "momentum",
                    0.25,
                ),
                0.25,
            )
        ),

        stage=str(
            raw.get(
                "stage",
                "spark",
            )
        ),

        seen_count=max(
            1,
            int(
                _safe_float(
                    raw.get(
                        "seen_count",
                        1,
                    ),
                    1,
                )
            ),
        ),

        visit_count=max(
            0,
            int(
                _safe_float(
                    raw.get(
                        "visit_count",
                        0,
                    ),
                    0,
                )
            ),
        ),

        self_steps=max(
            0,
            int(
                _safe_float(
                    raw.get(
                        "self_steps",
                        0,
                    ),
                    0,
                )
            ),
        ),

        completed_actions=max(
            0,
            int(
                _safe_float(
                    raw.get(
                        "completed_actions",
                        0,
                    ),
                    0,
                )
            ),
        ),

        revisit_credit=max(
            0,
            min(
                3,
                int(
                    _safe_float(
                        raw.get(
                            "revisit_credit",
                            1,
                        ),
                        1,
                    )
                ),
            ),
        ),

        first_seen_at=str(
            raw.get(
                "first_seen_at",
                "",
            )
        ),

        last_seen_at=str(
            raw.get(
                "last_seen_at",
                "",
            )
        ),

        last_visit_at=str(
            raw.get(
                "last_visit_at",
                "",
            )
        ),

        last_action_at=str(
            raw.get(
                "last_action_at",
                "",
            )
        ),

        source_counts={
            str(key): max(
                0,
                int(
                    _safe_float(
                        value,
                        0,
                    )
                ),
            )
            for key, value
            in _dict_value(
                raw.get(
                    "source_counts"
                )
            ).items()
        },

        keywords=_clean_keywords(
            _list_value(
                raw.get(
                    "keywords"
                )
            )
        ),
    )


def _change_from_dict(
    raw: dict[str, Any],
) -> ChangeEvent:
    return ChangeEvent(
        id=str(
            raw.get(
                "id",
                content_fingerprint(raw),
            )
        ),

        at=str(
            raw.get(
                "at",
                "",
            )
        ),

        kind=str(
            raw.get(
                "kind",
                "change",
            )
        ),

        title=str(
            raw.get(
                "title",
                "",
            )
        ),

        detail=str(
            raw.get(
                "detail",
                "",
            )
        ),

        before=(
            _safe_float(
                raw.get(
                    "before"
                )
            )
            if raw.get(
                "before"
            )
            is not None
            else None
        ),

        after=(
            _safe_float(
                raw.get(
                    "after"
                )
            )
            if raw.get(
                "after"
            )
            is not None
            else None
        ),

        interest_key=(
            str(
                raw.get(
                    "interest_key"
                )
            )
            if raw.get(
                "interest_key"
            )
            else None
        ),
    )


def _intent_from_dict(
    raw: dict[str, Any] | None,
) -> Intent | None:
    if not raw:
        return None

    return Intent(
        drive=str(
            raw.get(
                "drive",
                "reflection",
            )
        ),

        score=clamp(
            _safe_float(
                raw.get(
                    "score",
                    0.0,
                )
            )
        ),

        want_action=str(
            raw.get(
                "want_action",
                "",
            )
        ),

        reason=str(
            raw.get(
                "reason",
                "",
            )
        ),

        thought_id=(
            str(
                raw.get(
                    "thought_id"
                )
            )
            if raw.get(
                "thought_id"
            )
            else None
        ),

        interest_key=(
            str(
                raw.get(
                    "interest_key"
                )
            )
            if raw.get(
                "interest_key"
            )
            else None
        ),

        wildcard=bool(
            raw.get(
                "wildcard",
                False,
            )
        ),
    )


def state_from_dict(
    raw: dict[str, Any],
) -> DesireState:
    state = DesireState()

    raw_drives = _dict_value(
        raw.get(
            "drives"
        )
    )

    for key in DRIVE_KEYS:
        if key in raw_drives:
            state.drives[
                key
            ] = clamp(
                _safe_float(
                    raw_drives[
                        key
                    ],
                    state.drives[
                        key
                    ],
                )
            )

    state.previous_drives = {
        key: clamp(
            _safe_float(
                value,
                0.0,
            )
        )
        for key, value
        in _dict_value(
            raw.get(
                "previous_drives"
            )
        ).items()
        if key in DRIVE_KEYS
    }

    state.thoughts = [
        _thought_from_dict(
            _dict_value(item)
        )
        for item in _list_value(
            raw.get(
                "thoughts"
            )
        )
        if isinstance(
            item,
            dict,
        )
    ]

    state.stimulus_stats = {
        str(key): StimulusStat(
            count=max(
                0,
                int(
                    _safe_float(
                        _dict_value(
                            value
                        ).get(
                            "count",
                            0,
                        ),
                        0,
                    )
                ),
            ),

            last_at=str(
                _dict_value(
                    value
                ).get(
                    "last_at",
                    "",
                )
            ),
        )
        for key, value
        in _dict_value(
            raw.get(
                "stimulus_stats"
            )
        ).items()
    }

    external_raw = _dict_value(
        raw.get(
            "external"
        )
    )

    state.external = ExternalState(
        relationship_closeness=clamp(
            _safe_float(
                external_raw.get(
                    "relationship_closeness",
                    1.0,
                ),
                1.0,
            )
        ),

        relationship_safety=clamp(
            _safe_float(
                external_raw.get(
                    "relationship_safety",
                    1.0,
                ),
                1.0,
            )
        ),

        relationship_trust=clamp(
            _safe_float(
                external_raw.get(
                    "relationship_trust",
                    1.0,
                ),
                1.0,
            )
        ),

        pa=clamp(
            _safe_float(
                external_raw.get(
                    "pa",
                    0.5,
                ),
                0.5,
            )
        ),

        na=clamp(
            _safe_float(
                external_raw.get(
                    "na",
                    0.2,
                ),
                0.2,
            )
        ),

        unresolved_count=max(
            0,
            int(
                _safe_float(
                    external_raw.get(
                        "unresolved_count",
                        0,
                    ),
                    0,
                )
            ),
        ),

        body_heat=clamp(
            _safe_float(
                external_raw.get(
                    "body_heat",
                    0.0,
                )
            )
        ),

        body_pressure=clamp(
            _safe_float(
                external_raw.get(
                    "body_pressure",
                    0.0,
                )
            )
        ),

        body_control=clamp(
            _safe_float(
                external_raw.get(
                    "body_control",
                    0.5,
                ),
                0.5,
            )
        ),

        body_sensitivity=clamp(
            _safe_float(
                external_raw.get(
                    "body_sensitivity",
                    0.0,
                )
            )
        ),

        body_reserve=clamp(
            _safe_float(
                external_raw.get(
                    "body_reserve",
                    0.0,
                )
            )
        ),

        body_possessiveness=clamp(
            _safe_float(
                external_raw.get(
                    "body_possessiveness",
                    0.0,
                )
            )
        ),

        body_fatigue=clamp(
            _safe_float(
                external_raw.get(
                    "body_fatigue",
                    0.2,
                ),
                0.2,
            )
        ),
    )

    self_raw = _dict_value(
        raw.get(
            "self_drive"
        )
    )

    state.self_drive = SelfDriveState(
        enabled=bool(
            self_raw.get(
                "enabled",
                False,
            )
        ),

        energy=clamp(
            _safe_float(
                self_raw.get(
                    "energy",
                    0.42,
                ),
                0.42,
            )
        ),

        curiosity_floor=clamp(
            _safe_float(
                self_raw.get(
                    "curiosity_floor",
                    0.18,
                ),
                0.18,
            )
        ),

        momentum=clamp(
            _safe_float(
                self_raw.get(
                    "momentum",
                    0.0,
                )
            )
        ),

        current_interest_key=(
            str(
                self_raw.get(
                    "current_interest_key"
                )
            )
            if self_raw.get(
                "current_interest_key"
            )
            else None
        ),

        last_self_action_at=str(
            self_raw.get(
                "last_self_action_at",
                "",
            )
        ),

        last_topic=str(
            self_raw.get(
                "last_topic",
                "",
            )
        ),

        daily_action_count=max(
            0,
            int(
                _safe_float(
                    self_raw.get(
                        "daily_action_count",
                        0,
                    ),
                    0,
                )
            ),
        ),

        daily_action_date=str(
            self_raw.get(
                "daily_action_date",
                "",
            )
        ),
    )

    state.interests = [
        _interest_from_dict(
            _dict_value(item)
        )
        for item in _list_value(
            raw.get(
                "interests"
            )
        )
        if isinstance(
            item,
            dict,
        )
    ]

    state.change_log = [
        _change_from_dict(
            _dict_value(item)
        )
        for item in _list_value(
            raw.get(
                "change_log"
            )
        )
        if isinstance(
            item,
            dict,
        )
    ][-60:]

    state.refractory = {
        str(key): max(
            0,
            int(
                _safe_float(
                    value,
                    0,
                )
            ),
        )
        for key, value
        in _dict_value(
            raw.get(
                "refractory"
            )
        ).items()
        if str(key) in DESIRE_KEYS
    }

    state.last_intent = _intent_from_dict(
        _dict_value(
            raw.get(
                "last_intent"
            )
        )
        if raw.get(
            "last_intent"
        )
        else None
    )

    state.last_user_interaction_at = str(
        raw.get(
            "last_user_interaction_at",
            "",
        )
    )

    state.last_heartbeat_at = str(
        raw.get(
            "last_heartbeat_at",
            "",
        )
    )

    state.heartbeat_count = max(
        0,
        int(
            _safe_float(
                raw.get(
                    "heartbeat_count",
                    0,
                ),
                0,
            )
        ),
    )

    state.next_heartbeat_seconds = max(
        1,
        int(
            _safe_float(
                raw.get(
                    "next_heartbeat_seconds",
                    900,
                ),
                900,
            )
        ),
    )

    state.updated_at = str(
        raw.get(
            "updated_at",
            "",
        )
    )

    state.upstream_seen = {
        str(key): str(value)
        for key, value
        in _dict_value(
            raw.get(
                "upstream_seen"
            )
        ).items()
    }

    state.gates.update(
        {
            str(key): bool(value)
            for key, value
            in _dict_value(
                raw.get(
                    "gates"
                )
            ).items()
            if str(key)
            in state.gates
        }
    )

    state.version = "0.3.0"

    _apply_projection(
        state
    )

    return state


class JSONStore:
    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

    def load(
        self,
    ) -> DesireState:
        with self._lock:
            if not self.path.exists():
                return DesireState()

            try:
                raw = json.loads(
                    self.path.read_text(
                        encoding="utf-8"
                    )
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):
                return DesireState()

            if not isinstance(
                raw,
                dict,
            ):
                return DesireState()

            return state_from_dict(
                raw
            )

    def save(
        self,
        state: DesireState,
    ) -> None:
        payload = asdict(
            state
        )

        temp = self.path.with_suffix(
            self.path.suffix
            + ".tmp"
        )

        with self._lock:
            temp.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temp.replace(
                self.path
            )

    def update(
        self,
        updater: Callable[
            [DesireState],
            DesireState,
        ],
    ) -> DesireState:
        with self._lock:
            current = self.load()

            updated = updater(
                current
            )

            if not isinstance(
                updated,
                DesireState,
            ):
                raise TypeError(
                    "updater must return "
                    "DesireState"
                )

            self.save(
                updated
            )

            return updated
