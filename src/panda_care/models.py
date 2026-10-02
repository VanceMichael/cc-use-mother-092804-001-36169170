"""跨境旅居动物照护服务的领域模型。

所有业务事实都以事件表达；事件一经入链即不可修改。
时间字段区分：
- ``occurred_at``：业务发生时间（喂养、用药等实际发生时刻，责任人填报）；
- ``recorded_at``：系统入链时间（争议回放以入链时间为准，防止倒填）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


class Side(str, Enum):
    """信息可见侧别。BOTH 表示中美双方共同可见。"""

    CN = "CN"
    US = "US"
    BOTH = "BOTH"


class Role(str, Enum):
    CN_KEEPER = "CN_KEEPER"            # 中方饲养员
    US_KEEPER = "US_KEEPER"            # 美方饲养员
    VET = "VET"                        # 普通兽医
    DESIGNATED_VET = "DESIGNATED_VET"  # 指定复核兽医
    TRANSPORT = "TRANSPORT"            # 运输承运方
    OPERATOR = "OPERATOR"              # 动物园运营人员
    AUTHORITY = "AUTHORITY"            # 双方项目管理机构


class Phase(str, Enum):
    PRE_DEPARTURE = "PRE_DEPARTURE"  # 出发前评估
    CRATING = "CRATING"              # 装运封存
    FLIGHT = "FLIGHT"                # 航程
    ADAPTATION = "ADAPTATION"        # 入园适应期


# 阶段先后顺序；门禁事件只能向前推进。
PHASE_ORDER: tuple[Phase, ...] = (
    Phase.PRE_DEPARTURE,
    Phase.CRATING,
    Phase.FLIGHT,
    Phase.ADAPTATION,
)


class EventType(str, Enum):
    CARE_PLAN_CREATED = "CARE_PLAN_CREATED"
    CARE_PLAN_REVISED = "CARE_PLAN_REVISED"
    PHASE_GATE = "PHASE_GATE"                      # 阶段门禁通过/推进
    HEALTH_ASSESSMENT = "HEALTH_ASSESSMENT"        # 出发前评估
    VET_CONCLUSION = "VET_CONCLUSION"              # 兽医结论（确认后锁定）
    CRATE_SEALED = "CRATE_SEALED"                  # 装运封存
    FLIGHT_EVENT = "FLIGHT_EVENT"                  # 航程事件（起飞/延误/到达等）
    FEEDING = "FEEDING"
    MEDICATION = "MEDICATION"
    OBSERVATION = "OBSERVATION"                    # 体重/行为/临床观察
    BATCH_DISPATCHED = "BATCH_DISPATCHED"          # 饲料/物资批次去向
    EFFECT_STARTED = "EFFECT_STARTED"              # 检疫/延误/替代/隔离开始
    EFFECT_RESOLVED = "EFFECT_RESOLVED"            # 满足恢复条件后解除
    TRANSFER_OF_CARE = "TRANSFER_OF_CARE"          # 照护责任交接
    REPORT_UPLOADED = "REPORT_UPLOADED"
    REVIEW_OPENED = "REVIEW_OPENED"
    REVIEW_RESOLVED = "REVIEW_RESOLVED"


class ObservationKind(str, Enum):
    WEIGHT = "WEIGHT"
    BEHAVIOR = "BEHAVIOR"
    CLINICAL = "CLINICAL"


class EffectKind(str, Enum):
    QUARANTINE = "QUARANTINE"                # 检疫要求
    FLIGHT_DELAY = "FLIGHT_DELAY"            # 航线延误
    FEED_SUBSTITUTION = "FEED_SUBSTITUTION"  # 饲料替代
    ISOLATION = "ISOLATION"                  # 单只暂时隔离


@dataclass(frozen=True)
class Actor:
    """责任人：带侧别与角色。示例数据不得使用真实身份信息。"""

    actor_id: str
    display_name: str
    side: Side
    roles: frozenset[Role]

    @staticmethod
    def create(actor_id: str, display_name: str, side: Side, roles: set[Role]) -> "Actor":
        return Actor(actor_id, display_name, side, frozenset(roles))

    def has_role(self, role: Role) -> bool:
        return role in self.roles


@dataclass(frozen=True)
class Scope:
    """事件影响范围。空集合表示全程/全体；否则只影响指定对象。

    - animal_ids：单只或多只动物；
    - plan_item_ids：照护计划条目（如某顿饲料安排）；
    - batch_ids：物资/饲料批次；
    - phase_limited：是否仅限当前阶段（阶段结束自动失效的语义由调用方判断）。
    """

    animal_ids: frozenset[str] = frozenset()
    plan_item_ids: frozenset[str] = frozenset()
    batch_ids: frozenset[str] = frozenset()
    phase_limited: bool = False

    @staticmethod
    def global_scope() -> "Scope":
        return Scope()

    @staticmethod
    def animals(*animal_ids: str) -> "Scope":
        return Scope(animal_ids=frozenset(animal_ids))

    @property
    def is_global(self) -> bool:
        return not (self.animal_ids or self.plan_item_ids or self.batch_ids)

    def touches_animal(self, animal_id: str) -> bool:
        return self.is_global or animal_id in self.animal_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "animal_ids": sorted(self.animal_ids),
            "plan_item_ids": sorted(self.plan_item_ids),
            "batch_ids": sorted(self.batch_ids),
            "phase_limited": self.phase_limited,
        }

    @staticmethod
    def from_dict(value: dict[str, Any]) -> "Scope":
        return Scope(
            animal_ids=frozenset(value.get("animal_ids", ())),
            plan_item_ids=frozenset(value.get("plan_item_ids", ())),
            batch_ids=frozenset(value.get("batch_ids", ())),
            phase_limited=bool(value.get("phase_limited", False)),
        )


@dataclass(frozen=True)
class Feeding:
    animal_id: str
    plan_item_id: str
    item: str
    batch_id: str
    amount_kg: float
    note: str = ""


@dataclass(frozen=True)
class Medication:
    animal_id: str
    drug: str
    dose: str
    reason: str
    per_vet_conclusion_id: str | None = None  # 用药依据的兽医结论


@dataclass(frozen=True)
class Observation:
    animal_id: str
    kind: ObservationKind
    value: str
    abnormal: bool = False
    note: str = ""


@dataclass(frozen=True)
class VetConclusion:
    """兽医结论。confirmed 结论是后续照护与复核的锚点。"""

    animal_id: str
    topic: str           # 如 fitness_to_load / diagnosis / quarantine_release
    finding: str
    confirmed: bool = True
    supersedes: str | None = None  # 修正旧结论时指向旧事件 id（旧事件保留）


@dataclass(frozen=True)
class BatchRef:
    batch_id: str
    item: str
    quantity: float
    unit: str


@dataclass(frozen=True)
class TransferOfCare:
    """照护责任交接，构成争议时的交接依据。"""

    checkpoint: str                     # 如 成都基地→承运方 / 承运方→亚特兰大园
    from_actor_id: str
    to_actor_id: str
    responsibilities: tuple[str, ...]
    acknowledged_by: frozenset[str]     # 双方确认人 actor_id
    animal_ids: frozenset[str] = frozenset()  # 交接覆盖的动物
    cross_border: bool = False
    manifest_hash: str | None = None


@dataclass(frozen=True)
class PhaseGate:
    from_phase: Phase
    to_phase: Phase
    basis_event_ids: frozenset[str]


@dataclass(frozen=True)
class SpeciesFact:
    """静态物种事实（如个体身份），与事件流分开管理。"""

    animal_id: str
    name: str
    species: str = "Ailuropoda melanoleuca"


@dataclass(frozen=True)
class CareEvent:
    """入链事件。seq/hash 由台账赋值，构造时留空。"""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    actor_id: str                       # 责任人
    phase: Phase
    body: dict[str, Any]
    scope: Scope = field(default_factory=Scope.global_scope)
    visible_to: Side = Side.BOTH
    recorded_at: datetime = field(default_factory=utc_now)
    seq: int | None = None
    prev_hash: str | None = None
    hash: str | None = None

    def sealed(self, seq: int, prev_hash: str, digest: str) -> "CareEvent":
        return replace(
            self,
            seq=seq,
            prev_hash=prev_hash,
            hash=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat(),
            "recorded_at": self.recorded_at.astimezone(timezone.utc).isoformat(),
            "actor_id": self.actor_id,
            "phase": self.phase.value,
            "scope": self.scope.to_dict(),
            "visible_to": self.visible_to.value,
            "body": self.body,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }
