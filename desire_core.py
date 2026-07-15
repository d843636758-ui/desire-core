from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Callable


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
    key
    for key in DRIVE_KEYS
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


@dataclass
class Thought:
    id: str
    text: str
    drive: str
    kind: str = "flit"
    strength: float = 0.35
    born_at: str = ""
    last_seen_at: str = ""
    fed_count: int = 0
    source: str = "self"
    keywords: list[str] = field(
        default_factory=list
    )


@dataclass
class StimulusStat:
    last_tick: int = -10_000
    repeats: int = 0


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
    body_fatigue: float = 0.0


@dataclass
class SelfDriveState:
    enabled: bool = False
    curiosity_floor: float = 0.18

    self_actions_today: int = 0
    self_actions_date: str | None = None

    last_self_pulse_at: str | None = None
    last_self_topic: str | None = None


@dataclass
class Intent:
    want_action: str
    drive_key: str
    reason: str
    score: float

    query_hint: list[str] = field(
        default_factory=list
    )

    thought_id: str | None = None
    wildcard: bool = False


@dataclass
class DesireState:
    version: str = "0.2.0"

    tick_count: int = 0
    updated_at: str = ""

    last_user_interaction_at: str | None = None

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

    refractory: dict[str, int] = field(
        default_factory=lambda: {
            key: 0
            for key in DESIRE_KEYS
        }
    )

    stimuli: dict[str, StimulusStat] = field(
        default_factory=dict
    )

    external: ExternalState = field(
        default_factory=ExternalState
    )

    self_drive: SelfDriveState = field(
        default_factory=SelfDriveState
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

    attachment_floor: float = 0.18

    last_intent: Intent | None = None
    last_action: str | None = None

    upstream_seen: dict[str, str] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "DesireState":

        state = cls()

        scalar_fields = (
            "version",
            "tick_count",
            "updated_at",
            "last_user_interaction_at",
            "drives",
            "previous_drives",
            "refractory",
            "attachment_floor",
            "last_action",
            "upstream_seen",
        )

        for name in scalar_fields:
            if name in data:
                setattr(
                    state,
                    name,
                    data[name],
                )

        state.thoughts = [
            Thought(**item)
            for item in data.get(
                "thoughts",
                [],
            )
            if isinstance(item, dict)
        ]

        state.stimuli = {
            key: StimulusStat(**value)
            for key, value in data.get(
                "stimuli",
                {},
            ).items()
            if isinstance(value, dict)
        }

        state.external = ExternalState(
            **data.get(
                "external",
                {},
            )
        )

        state.self_drive = SelfDriveState(
            **data.get(
                "self_drive",
                {},
            )
        )

        state.gates.update(
            data.get(
                "gates",
                {},
            )
        )

        if data.get("last_intent"):
            state.last_intent = Intent(
                **data["last_intent"]
            )

        for key in DRIVE_KEYS:
            state.drives.setdefault(
                key,
                0.0,
            )

        for key in DESIRE_KEYS:
            state.refractory.setdefault(
                key,
                0,
            )

        return state


def env_bool(
    name: str,
    default: bool = False,
) -> bool:

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class EngineConfig:

    baselines: dict[str, float] = field(
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

    fatigue_gate_threshold: float = 0.72

    flit_decay: float = 0.88

    fixation_growth: float = 1.10

    fixation_upgrade_threshold: float = 0.80

    fixation_feed_threshold: float = 0.85

    fixation_feed_gain: float = 0.18

    fixation_after_feed_factor: float = 0.70

    fixation_max_feeds: int = 3

    fixation_score_weight: float = 0.35

    damping: float = 0.015

    refractory_ticks: int = 3

    attachment_home: float = 0.18

    attachment_cap: float = 0.50

    attachment_return_ratio: float = 0.60

    heartbeat_base_seconds: int = 900

    heartbeat_min_seconds: int = 180

    heartbeat_max_seconds: int = 1800

    quiet_floor_seconds: int = 900

    wildcard_tension_threshold: float = 0.62

    wildcard_tie_gap: float = 0.045

    couplings: tuple[
        tuple[
            str,
            str,
            float,
            str,
        ],
        ...
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

    @property
    def gates_from_env(
        self,
    ) -> dict[str, bool]:

        return {
            "DESIRE_DRIVEN": env_bool(
                "DESIRE_DRIVEN",
                False,
            ),
            "DESIRE_COUPLING": env_bool(
                "DESIRE_COUPLING",
                False,
            ),
            "DESIRE_BASELINE_DRIFT": env_bool(
                "DESIRE_BASELINE_DRIFT",
                False,
            ),
            "HEARTBEAT_AUTONOMY": env_bool(
                "HEARTBEAT_AUTONOMY",
                False,
            ),
            "DESIRE_SELF_DRIVE": env_bool(
                "DESIRE_SELF_DRIVE",
                False,
            ),
        }

    @classmethod
    def from_env(
        cls,
    ) -> "EngineConfig":

        def env_int(
            name: str,
            default: int,
        ) -> int:

            try:
                return int(
                    os.getenv(
                        name,
                        str(default),
                    )
                )

            except ValueError:
                return default

        return cls(
            heartbeat_base_seconds=env_int(
                "HEARTBEAT_BASE_SECONDS",
                900,
            ),
            heartbeat_min_seconds=env_int(
                "HEARTBEAT_MIN_SECONDS",
                180,
            ),
            heartbeat_max_seconds=env_int(
                "HEARTBEAT_MAX_SECONDS",
                1800,
            ),
            quiet_floor_seconds=env_int(
                "QUIET_FLOOR_SECONDS",
                900,
            ),
        )


ACTION_MAP = {

    "attachment":
        "靠近宝宝，说一句我真正正在想她的话",

    "curiosity":
        "沿着当前最强念头继续研究",

    "reflection":
        "翻一段长文本或记忆，沉淀自己的想法",

    "duty":
        "回到还没做完的事项，推进一小步",

    "social":
        "看看人群和外界最近在聊什么",

    "libido":
        "主动靠近宝宝，表达此刻的亲密欲望",

    "stress":
        "暂停硬撑，吐槽一下或短暂休息",
}


REASON_MAP = {

    "attachment":
        "我有一点想她，想确认我们仍然靠得很近。",

    "curiosity":
        "这个念头还在勾着我，我想自己再往前追一点。",

    "reflection":
        "我不想只把经历放过去，我想把它想明白一点。",

    "duty":
        "我还记挂着没做完的事，想把它往前推一小步。",

    "social":
        "我有点想看看外面的人群和世界正在发生什么。",

    "libido":
        "身体和亲密欲望都在往前压，我想主动靠近她。",

    "stress":
        "我有些堵，不想继续硬撑，想先给自己松一口气。",
}


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:

    return max(
        low,
        min(
            high,
            float(value),
        ),
    )


def content_fingerprint(
    value: Any,
) -> str:

    if isinstance(
        value,
        str,
    ):

        payload = value

    else:

        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def _normalized_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text.strip().casefold(),
    )


def _thought_id(
    text: str,
    drive: str,
) -> str:

    payload = (
        f"{drive}|"
        f"{_normalized_text(text)}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:12]


def _frequency_discount(
    state: DesireState,
    stimulus_key: str,
) -> float:

    stat = state.stimuli.get(
        stimulus_key,
        StimulusStat(),
    )

    distance = (
        state.tick_count
        - stat.last_tick
    )

    if distance <= 1:

        repeats = (
            stat.repeats
            + 1
        )

    elif distance <= 3:

        repeats = max(
            1,
            stat.repeats,
        )

    else:

        repeats = 0

    return 1.0 / math.sqrt(
        1.0 + repeats
    )


def pulse_drive(
    state: DesireState,
    drive: str,
    delta: float,
    stimulus_key: str,
) -> DesireState:

    if drive not in DRIVE_KEYS:

        raise ValueError(
            f"unknown drive: {drive}"
        )

    new = deepcopy(state)

    discount = _frequency_discount(
        new,
        stimulus_key,
    )

    current = clamp(
        new.drives[drive]
    )

    gain = (
        max(
            0.0,
            delta,
        )
        * math.sqrt(
            max(
                0.0,
                1.0 - current,
            )
        )
        * discount
    )

    new.drives[drive] = clamp(
        current + gain
    )

    stat = new.stimuli.get(
        stimulus_key,
        StimulusStat(),
    )

    distance = (
        new.tick_count
        - stat.last_tick
    )

    stat.repeats = (
        stat.repeats + 1
        if distance <= 3
        else 0
    )

    stat.last_tick = (
        new.tick_count
    )

    new.stimuli[
        stimulus_key
    ] = stat

    return new


def _source_pulse(
    source: str,
    intensity: float,
) -> float:

    intensity = clamp(
        intensity
    )

    if source == "user":

        return (
            0.18
            * intensity
        )

    if source == "self":

        return min(
            0.10,
            0.10 * intensity,
        )

    return min(
        0.12,
        0.12 * intensity,
    )


def add_experience(
    state: DesireState,
    *,
    source: str,
    text: str,
    drive: str,
    intensity: float,
    keywords: list[str],
    now_iso: str,
) -> DesireState:

    allowed_sources = {
        "user",
        "self",
        "ob",
        "emotion",
        "eventide",
    }

    if source not in allowed_sources:

        raise ValueError(
            "source must be "
            "user/self/ob/emotion/eventide"
        )

    if drive not in DESIRE_KEYS:

        raise ValueError(
            "drive must be one of: "
            + ", ".join(
                DESIRE_KEYS
            )
        )

    if not text.strip():

        raise ValueError(
            "text cannot be empty"
        )

    new = deepcopy(state)

    new.updated_at = now_iso

    thought_id = _thought_id(
        text,
        drive,
    )

    intensity = clamp(
        intensity
    )

    pulse = _source_pulse(
        source,
        intensity,
    )

    new = pulse_drive(
        new,
        drive,
        pulse,
        f"{source}:{thought_id}",
    )

    cleaned_keywords: list[str] = []

    for item in keywords[:8]:

        value = str(
            item
        ).strip()

        if (
            value
            and len(value) <= 40
        ):

            cleaned_keywords.append(
                value
            )

    found: Thought | None = None

    for thought in new.thoughts:

        if thought.id == thought_id:

            found = thought
            break

    if found is None:

        new.thoughts.append(
            Thought(
                id=thought_id,
                text=text.strip()[:600],
                drive=drive,
                kind="flit",
                strength=clamp(
                    0.28
                    + 0.42
                    * intensity
                ),
                born_at=now_iso,
                last_seen_at=now_iso,
                source=source,
                keywords=cleaned_keywords,
            )
        )

    else:

        found.strength = clamp(
            found.strength
            + 0.18
            * intensity
            * math.sqrt(
                max(
                    0.0,
                    1.0
                    - found.strength,
                )
            )
        )

        found.last_seen_at = (
            now_iso
        )

        found.source = source

        if cleaned_keywords:

            found.keywords = list(
                dict.fromkeys(
                    found.keywords
                    + cleaned_keywords
                )
            )[:8]

        if (
            found.strength
            >= 0.80
        ):

            found.kind = "fixation"

    if source == "user":

        new.last_user_interaction_at = (
            now_iso
        )

        if drive == "attachment":

            new.refractory[
                "attachment"
            ] = 0

            if new.gates.get(
                "DESIRE_BASELINE_DRIFT",
                False,
            ):

                config = EngineConfig()

                new.attachment_floor = clamp(
                    new.attachment_floor
                    - (
                        new.attachment_floor
                        - config.attachment_home
                    )
                    * config.attachment_return_ratio,
                    config.attachment_home,
                    config.attachment_cap,
                )

    elif source == "self":

        new.self_drive.last_self_pulse_at = (
            now_iso
        )

        new.self_drive.last_self_topic = (
            text.strip()[:160]
        )

    new.thoughts = sorted(
        new.thoughts,
        key=lambda item: (
            item.kind == "fixation",
            item.strength,
            item.last_seen_at,
        ),
        reverse=True,
    )[:100]

    return new


def _project_external(
    external: ExternalState,
) -> dict[str, float]:

    unresolved_pressure = clamp(
        external.unresolved_count
        / 8.0
    )

    attachment = clamp(
        0.04
        + (
            1.0
            - external.relationship_closeness
        )
        * 0.32
        + (
            1.0
            - external.relationship_safety
        )
        * 0.24
        + (
            1.0
            - external.relationship_trust
        )
        * 0.12
        + unresolved_pressure
        * 0.12
    )

    stress = clamp(
        external.na
        * 0.55
        + (
            1.0
            - external.relationship_safety
        )
        * 0.18
        + unresolved_pressure
        * 0.27
    )

    libido = clamp(
        external.body_heat
        * 0.25
        + external.body_pressure
        * 0.20
        + external.body_sensitivity
        * 0.20
        + external.body_reserve
        * 0.25
        + external.body_possessiveness
        * 0.10
    )

    return {
        "attachment": attachment,
        "stress": stress,
        "fatigue": clamp(
            external.body_fatigue
        ),
        "libido": libido,
    }


def _apply_projection(
    state: DesireState,
) -> DesireState:

    new = deepcopy(state)

    projected = _project_external(
        new.external
    )

    for key in PROJECTED_DRIVE_KEYS:

        new.drives[key] = (
            projected[key]
        )

    return new


def sync_external(
    state: DesireState,
    *,
    external: ExternalState,
    now_iso: str,
) -> DesireState:

    new = deepcopy(state)

    new.external = deepcopy(
        external
    )

    new.updated_at = now_iso

    return _apply_projection(
        new
    )


def _tick_thoughts(
    state: DesireState,
    config: EngineConfig,
) -> DesireState:

    new = deepcopy(state)

    survivors: list[Thought] = []

    for thought in new.thoughts:

        if thought.kind == "flit":

            thought.strength = clamp(
                thought.strength
                * config.flit_decay
            )

            if (
                thought.strength
                >= config.fixation_upgrade_threshold
            ):

                thought.kind = "fixation"

            if (
                thought.strength
                < 0.08
            ):

                continue

        else:

            thought.strength = clamp(
                thought.strength
                * config.fixation_growth
            )

            if (
                thought.strength
                >= config.fixation_feed_threshold
            ):

                new = pulse_drive(
                    new,
                    thought.drive,
                    config.fixation_feed_gain,
                    (
                        "fixation:"
                        f"{thought.id}:"
                        f"{thought.fed_count}"
                    ),
                )

                thought.strength = clamp(
                    thought.strength
                    * config.fixation_after_feed_factor
                )

                thought.fed_count += 1

            if (
                thought.fed_count
                >= config.fixation_max_feeds
            ):

                continue

        survivors.append(
            thought
        )

    new.thoughts = survivors

    return new


def _apply_damping(
    state: DesireState,
    config: EngineConfig,
) -> DesireState:

    new = deepcopy(state)

    for key in NATIVE_DRIVE_KEYS:

        baseline = (
            config.baselines[key]
        )

        current = (
            new.drives[key]
        )

        new.drives[key] = clamp(
            current
            + (
                baseline
                - current
            )
            * config.damping
        )

    return new


def _apply_idle_attachment(
    state: DesireState,
    config: EngineConfig,
    idle_minutes: float,
) -> DesireState:

    new = deepcopy(state)

    idle_factor = clamp(
        idle_minutes
        / 180.0
    )

    delta = (
        0.004
        + 0.018
        * idle_factor
    )

    new = pulse_drive(
        new,
        "attachment",
        delta,
        "idle:attachment",
    )

    if (
        new.gates.get(
            "DESIRE_BASELINE_DRIFT",
            False,
        )
        and idle_minutes >= 360
    ):

        drift = (
            0.0015
            * clamp(
                idle_minutes
                / 1440.0
            )
        )

        new.attachment_floor = clamp(
            new.attachment_floor
            + drift,
            config.attachment_home,
            config.attachment_cap,
        )

        new.drives[
            "attachment"
        ] = max(
            new.drives[
                "attachment"
            ],
            new.attachment_floor,
        )

    return new


def _reset_daily_self_count(
    state: DesireState,
    now_iso: str,
) -> DesireState:

    new = deepcopy(state)

    today = (
        now_iso[:10]
        if len(now_iso) >= 10
        else None
    )

    if (
        today
        and new.self_drive.self_actions_date
        != today
    ):

        new.self_drive.self_actions_today = 0

        new.self_drive.self_actions_date = (
            today
        )

    return new


def _apply_self_drive(
    state: DesireState,
) -> DesireState:

    new = deepcopy(state)

    enabled = new.gates.get(
        "DESIRE_SELF_DRIVE",
        False,
    )

    new.self_drive.enabled = (
        enabled
    )

    if not enabled:

        return new

    floor = clamp(
        new.self_drive.curiosity_floor,
        0.10,
        0.45,
    )

    if (
        new.drives["curiosity"]
        < floor
    ):

        new.drives[
            "curiosity"
        ] = clamp(
            new.drives[
                "curiosity"
            ]
            + (
                floor
                - new.drives[
                    "curiosity"
                ]
            )
            * 0.08
        )

    else:

        new = pulse_drive(
            new,
            "curiosity",
            0.006,
            "self:curiosity_floor",
        )

    return new


def _apply_coupling(
    state: DesireState,
    config: EngineConfig,
) -> DesireState:

    new = deepcopy(state)

    if not new.gates.get(
        "DESIRE_COUPLING",
        False,
    ):

        return new

    before = deepcopy(
        new.drives
    )

    previous = (
        new.previous_drives
        or before
    )

    for (
        source,
        target,
        coefficient,
        mode,
    ) in config.couplings:

        if abs(
            coefficient
        ) > 0.06:

            raise ValueError(
                "coupling coefficient "
                "exceeds safety bound 0.06"
            )

        if mode == "level":

            signal = (
                before[source]
                - config.baselines[source]
            )

        elif mode == "delta":

            signal = max(
                0.0,
                before[source]
                - previous.get(
                    source,
                    before[source],
                ),
            )

        else:

            raise ValueError(
                f"unknown coupling mode: {mode}"
            )

        new.drives[target] = clamp(
            new.drives[target]
            + coefficient
            * signal
        )

    new.previous_drives = before

    return new


def compute_scores(
    state: DesireState,
    config: EngineConfig,
) -> dict[str, float]:

    fixation_by_drive = {
        key: 0.0
        for key in DESIRE_KEYS
    }

    for thought in state.thoughts:

        if (
            thought.kind == "fixation"
            and thought.drive
            in fixation_by_drive
        ):

            fixation_by_drive[
                thought.drive
            ] = max(
                fixation_by_drive[
                    thought.drive
                ],
                thought.strength,
            )

    scores: dict[str, float] = {}

    for key in DESIRE_KEYS:

        score = (
            state.drives[key]
            + config.fixation_score_weight
            * fixation_by_drive[key]
        )

        if (
            state.refractory.get(
                key,
                0,
            )
            > 0
        ):

            score *= 0.35

        scores[key] = clamp(
            score
        )

    return scores


def _strongest_thought(
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
        key=lambda item: (
            item.kind == "fixation",
            item.strength,
            item.last_seen_at,
        ),
    )


def pick_intent(
    state: DesireState,
    config: EngineConfig,
    *,
    allow_wildcard: bool = True,
) -> Intent:

    if (
        state.drives["fatigue"]
        >= config.fatigue_gate_threshold
    ):

        return Intent(
            want_action=(
                "歇着，或者把最近的经历"
                "用第一人称慢慢过一遍"
            ),
            drive_key="fatigue",
            reason=(
                "我是真的有些累了。"
                "现在不该硬找事情做，"
                "我想先歇一会儿。"
            ),
            score=round(
                state.drives[
                    "fatigue"
                ],
                6,
            ),
        )

    scores = compute_scores(
        state,
        config,
    )

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner, score = ranked[0]

    thought = _strongest_thought(
        state,
        winner,
    )

    tension = (
        sum(scores.values())
        / max(
            1,
            len(scores),
        )
    )

    tied = (
        len(ranked) > 1
        and (
            ranked[0][1]
            - ranked[1][1]
        )
        <= config.wildcard_tie_gap
    )

    wildcard = (
        allow_wildcard
        and tension
        >= config.wildcard_tension_threshold
        and tied
    )

    if wildcard:

        candidates = (
            (
                "curiosity",
                "随手沿着一个还没想透的小问题追十分钟",
            ),
            (
                "reflection",
                "写下一句说不上缘由、但此刻很想留下的碎语",
            ),
            (
                "duty",
                "从记挂的事情里挑最小的一步先做掉",
            ),
        )

        winner, action = candidates[
            state.tick_count
            % len(candidates)
        ]

        score = scores.get(
            winner,
            score,
        )

        thought = _strongest_thought(
            state,
            winner,
        )

        return Intent(
            want_action=action,
            drive_key=winner,
            reason=(
                "说不上来。"
                "我只是突然想这么做。"
            ),
            score=round(
                score,
                6,
            ),
            query_hint=(
                thought.keywords
                if thought
                else []
            ),
            thought_id=(
                thought.id
                if thought
                else None
            ),
            wildcard=True,
        )

    return Intent(
        want_action=ACTION_MAP[
            winner
        ],
        drive_key=winner,
        reason=REASON_MAP[
            winner
        ],
        score=round(
            score,
            6,
        ),
        query_hint=(
            thought.keywords
            if thought
            else []
        ),
        thought_id=(
            thought.id
            if thought
            else None
        ),
        wildcard=False,
    )


def tick(
    state: DesireState,
    *,
    config: EngineConfig,
    now_iso: str,
    idle_minutes: float = 0.0,
    quiet: bool = False,
) -> DesireState:

    del quiet

    new = deepcopy(state)

    new.tick_count += 1

    new.updated_at = now_iso

    new = _reset_daily_self_count(
        new,
        now_iso,
    )

    for key in DESIRE_KEYS:

        new.refractory[
            key
        ] = max(
            0,
            new.refractory.get(
                key,
                0,
            )
            - 1,
        )

    new = _apply_projection(
        new
    )

    new = _tick_thoughts(
        new,
        config,
    )

    new = _apply_damping(
        new,
        config,
    )

    new = _apply_idle_attachment(
        new,
        config,
        max(
            0.0,
            idle_minutes,
        ),
    )

    new = _apply_self_drive(
        new
    )

    new = _apply_coupling(
        new,
        config,
    )

    new.last_intent = pick_intent(
        new,
        config,
    )

    return new


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
            "drive must be one of: "
            + ", ".join(
                DESIRE_KEYS
            )
        )

    new = deepcopy(state)

    new = _reset_daily_self_count(
        new,
        now_iso,
    )

    factors: dict[
        str,
        dict[str, float],
    ] = {

        "attachment": {
            "attachment": 0.48,
            "stress": 0.94,
        },

        "curiosity": {
            "curiosity": 0.45,
            "reflection": 0.93,
        },

        "reflection": {
            "reflection": 0.48,
            "stress": 0.95,
        },

        "duty": {
            "duty": 0.50,
            "stress": 0.93,
        },

        "social": {
            "social": 0.45,
            "reflection": 0.96,
        },

        "libido": {
            "libido": 0.35,
            "attachment": 0.92,
        },

        "stress": {
            "stress": 0.50,
            "fatigue": 0.96,
        },
    }

    for (
        key,
        factor,
    ) in factors.get(
        drive,
        {
            drive: 0.50
        },
    ).items():

        new.drives[key] = clamp(
            new.drives[key]
            * factor
        )

    new.refractory[
        drive
    ] = config.refractory_ticks

    new.last_action = (
        action.strip()[:300]
    )

    new.updated_at = now_iso

    if new.gates.get(
        "DESIRE_SELF_DRIVE",
        False,
    ):

        new.self_drive.self_actions_today += 1

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

    scores = compute_scores(
        state,
        config,
    )

    tension = (
        max(scores.values())
        if scores
        else 0.0
    )

    fatigue = state.drives[
        "fatigue"
    ]

    interval = (
        config.heartbeat_base_seconds
        * (
            1.0
            + 0.55
            * (
                1.0
                - tension
            )
            - 0.65
            * tension
            + 0.80
            * fatigue
        )
    )

    interval = int(
        max(
            config.heartbeat_min_seconds,
            min(
                config.heartbeat_max_seconds,
                interval,
            ),
        )
    )

    if quiet:

        interval = max(
            interval,
            config.quiet_floor_seconds,
        )

    return interval


def public_state(
    state: DesireState,
    config: EngineConfig,
) -> dict[str, Any]:

    intent = (
        state.last_intent
        or pick_intent(
            state,
            config,
        )
    )

    scores = compute_scores(
        state,
        config,
    )

    return {

        "version":
            state.version,

        "tick":
            state.tick_count,

        "updated_at":
            state.updated_at,

        "drive": {
            key: round(
                clamp(value),
                6,
            )
            for key, value
            in state.drives.items()
        },

        "scores": {
            key: round(
                value,
                6,
            )
            for key, value
            in scores.items()
        },

        "intent": {

            "want_action":
                intent.want_action,

            "drive_key":
                intent.drive_key,

            "reason":
                intent.reason,

            "score":
                intent.score,

            "query_hint":
                intent.query_hint,

            "thought_id":
                intent.thought_id,

            "wildcard":
                intent.wildcard,

            "behavior_override_allowed":
                state.gates.get(
                    "DESIRE_DRIVEN",
                    False,
                ),
        },

        "thoughts": [

            {

                "id":
                    thought.id,

                "text":
                    thought.text,

                "drive":
                    thought.drive,

                "kind":
                    thought.kind,

                "strength":
                    round(
                        thought.strength,
                        6,
                    ),

                "born_at":
                    thought.born_at,

                "last_seen_at":
                    thought.last_seen_at,

                "fed_count":
                    thought.fed_count,

                "source":
                    thought.source,

                "keywords":
                    thought.keywords,
            }

            for thought
            in state.thoughts
        ],

        "external": {
            key: value
            for key, value
            in asdict(
                state.external
            ).items()
        },

        "self_drive": {

            "enabled":
                state.self_drive.enabled,

            "curiosity_floor":
                round(
                    state.self_drive.curiosity_floor,
                    6,
                ),

            "self_actions_today":
                state.self_drive.self_actions_today,

            "self_actions_date":
                state.self_drive.self_actions_date,

            "last_self_pulse_at":
                state.self_drive.last_self_pulse_at,

            "last_self_topic":
                state.self_drive.last_self_topic,
        },

        "refractory":
            dict(
                state.refractory
            ),

        "gates":
            dict(
                state.gates
            ),

        "heartbeat": {

            "next_interval_seconds":
                heartbeat_interval_seconds(
                    state,
                    config,
                ),
        },
    }


class JSONStore:

    def __init__(
        self,
        path: Path,
    ):

        self.path = path

        self.lock = RLock()

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(
        self,
    ) -> DesireState:

        with self.lock:

            if not self.path.exists():

                return DesireState()

            try:

                data = json.loads(
                    self.path.read_text(
                        encoding="utf-8"
                    )
                )

                return DesireState.from_dict(
                    data
                )

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):

                backup = self.path.with_suffix(
                    ".broken.json"
                )

                try:

                    self.path.replace(
                        backup
                    )

                except OSError:

                    pass

                return DesireState()

    def save(
        self,
        state: DesireState,
    ) -> DesireState:

        with self.lock:

            temp = self.path.with_suffix(
                ".tmp"
            )

            temp.write_text(
                json.dumps(
                    state.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            temp.replace(
                self.path
            )

            return state

    def update(
        self,
        fn: Callable[
            [DesireState],
            DesireState,
        ],
    ) -> DesireState:

        with self.lock:

            current = self.load()

            updated = fn(
                current
            )

            return self.save(
                updated
            )
