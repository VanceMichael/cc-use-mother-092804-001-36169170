"""跨境旅居动物照护服务。

在只追加台账之上提供业务规则：

1. 连续档案：出发前评估 → 装运封存 → 航程事件 → 入园适应，阶段只能向前推进；
2. 责任留痕：每条记录带发生时间与责任人，换班只能追加；
3. 兽医结论锁定：确认结论不可改写，修正只能以新结论声明 supersedes；
4. 范围化事件：检疫/延误/饲料替代/单只隔离只挂接到相关动物、计划条目或批次，
   必须带恢复条件，条件满足前挂接的环节不可推进；
5. 幂等上传：同一报告重复提交返回原处理结果；
6. 矛盾复核：报告与已确认兽医结论冲突时，转指定兽医，未决前冲突结论不作为依据；
7. 侧别可见性 + 时点回放：任一时点可还原照护计划、物资去向、交接依据。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .idempotency import IdempotencyStore, UploadResult, UploadStatus
from .ledger import Ledger
from .models import (
    Actor,
    BatchRef,
    CareEvent,
    EffectKind,
    EventType,
    Feeding,
    Medication,
    Observation,
    ObservationKind,
    PHASE_ORDER,
    Phase,
    Role,
    Scope,
    Side,
    TransferOfCare,
    VetConclusion,
    SpeciesFact,
    new_event_id,
    utc_now,
)


class CareServiceError(RuntimeError):
    """业务规则冲突。"""


class PermissionDenied(CareServiceError):
    """责任人无权执行该操作。"""


class PhaseGateError(CareServiceError):
    """阶段推进条件不满足。"""


class EffectActiveError(CareServiceError):
    """存在未恢复的范围化事件挂接该环节。"""


# 各事件类型允许的角色
PERMISSIONS: dict[EventType, frozenset[Role]] = {
    EventType.CARE_PLAN_CREATED: frozenset({Role.OPERATOR}),
    EventType.CARE_PLAN_REVISED: frozenset({Role.OPERATOR, Role.DESIGNATED_VET}),
    EventType.HEALTH_ASSESSMENT: frozenset({Role.VET, Role.DESIGNATED_VET}),
    EventType.VET_CONCLUSION: frozenset({Role.VET, Role.DESIGNATED_VET}),
    EventType.CRATE_SEALED: frozenset({Role.CN_KEEPER, Role.TRANSPORT, Role.OPERATOR}),
    EventType.FLIGHT_EVENT: frozenset({Role.TRANSPORT}),
    EventType.FEEDING: frozenset({Role.CN_KEEPER, Role.US_KEEPER}),
    EventType.MEDICATION: frozenset({Role.VET, Role.DESIGNATED_VET, Role.CN_KEEPER, Role.US_KEEPER}),
    EventType.OBSERVATION: frozenset({Role.CN_KEEPER, Role.US_KEEPER, Role.VET, Role.DESIGNATED_VET}),
    EventType.BATCH_DISPATCHED: frozenset({Role.OPERATOR, Role.TRANSPORT}),
    EventType.EFFECT_STARTED: frozenset({Role.OPERATOR, Role.AUTHORITY, Role.VET, Role.DESIGNATED_VET, Role.TRANSPORT}),
    EventType.EFFECT_RESOLVED: frozenset({Role.OPERATOR, Role.AUTHORITY, Role.VET, Role.DESIGNATED_VET, Role.TRANSPORT}),
    EventType.TRANSFER_OF_CARE: frozenset({Role.OPERATOR, Role.AUTHORITY}),
    EventType.PHASE_GATE: frozenset({Role.AUTHORITY}),
    EventType.REPORT_UPLOADED: frozenset(
        {Role.CN_KEEPER, Role.US_KEEPER, Role.VET, Role.DESIGNATED_VET, Role.TRANSPORT, Role.OPERATOR, Role.AUTHORITY}
    ),
    EventType.REVIEW_OPENED: frozenset({Role.DESIGNATED_VET}),
    EventType.REVIEW_RESOLVED: frozenset({Role.DESIGNATED_VET}),
}


def _require_claims(claims: list[dict[str, str]]) -> None:
    if not claims:
        raise CareServiceError("报告至少包含一条关键结论 claim")
    for claim in claims:
        for key in ("animal_id", "topic", "finding"):
            if not str(claim.get(key, "")).strip():
                raise CareServiceError(f"报告 claim 缺少 {key}")


@dataclass(frozen=True)
class ActiveEffect:
    event_id: str
    kind: EffectKind
    scope: Scope
    recovery_condition: str
    started_at: datetime
    substitute: dict[str, Any] | None = None


@dataclass(frozen=True)
class ReplayView:
    """某一时点、某一侧别可见的照护全貌。"""

    as_of: datetime
    side: Side
    phase: Phase
    plan: dict[str, Any]
    active_effects: tuple[ActiveEffect, ...]
    latest_weight_kg: dict[str, float]
    batch_movements: tuple[dict[str, Any], ...]
    transfers: tuple[dict[str, Any], ...]
    vet_conclusions: tuple[dict[str, Any], ...]
    open_reviews: tuple[dict[str, Any], ...]
    events: tuple[CareEvent, ...]
    head_hash: str

    def find_effect(self, kind: EffectKind, animal_id: str | None = None,
                    plan_item_id: str | None = None) -> ActiveEffect | None:
        for effect in self.active_effects:
            if effect.kind != kind:
                continue
            if animal_id is not None and not effect.scope.touches_animal(animal_id):
                continue
            if plan_item_id is not None and plan_item_id not in effect.scope.plan_item_ids:
                continue
            return effect
        return None


class CareService:
    def __init__(self, ledger: Ledger | None = None,
                 idempotency: IdempotencyStore | None = None,
                 *, clock: Any = utc_now) -> None:
        self.ledger = ledger or Ledger()
        self.idempotency = idempotency or IdempotencyStore()
        self._actors: dict[str, Actor] = {}
        self._animals: dict[str, SpeciesFact] = {}
        self._designated_vet_id: str | None = None
        self._clock = clock

    # ---- 人员与动物登记 -------------------------------------------------

    def register_actor(self, actor: Actor) -> None:
        if actor.actor_id in self._actors:
            raise CareServiceError("责任人已登记")
        self._actors[actor.actor_id] = actor
        if Role.DESIGNATED_VET in actor.roles:
            if self._designated_vet_id is not None:
                raise CareServiceError("指定复核兽医只能有一位")
            self._designated_vet_id = actor.actor_id

    def register_animal(self, animal: SpeciesFact) -> None:
        if animal.animal_id in self._animals:
            raise CareServiceError("动物已登记")
        self._animals[animal.animal_id] = animal

    @property
    def animals(self) -> dict[str, SpeciesFact]:
        return dict(self._animals)

    def _require_animal(self, animal_id: str) -> SpeciesFact:
        animal = self._animals.get(animal_id)
        if animal is None:
            raise CareServiceError(f"动物未登记: {animal_id}")
        return animal

    @property
    def designated_vet_id(self) -> str | None:
        return self._designated_vet_id

    def _actor(self, actor_id: str) -> Actor:
        actor = self._actors.get(actor_id)
        if actor is None:
            raise CareServiceError(f"责任人未登记: {actor_id}")
        return actor

    def _require_permission(self, event_type: EventType, actor_id: str) -> Actor:
        actor = self._actor(actor_id)
        allowed = PERMISSIONS.get(event_type, frozenset())
        if not any(actor.has_role(role) for role in allowed):
            raise PermissionDenied(f"{actor.display_name} 无权执行 {event_type.value}")
        return actor

    # ---- 通用追加 -------------------------------------------------------

    def _append(self, event_type: EventType, actor_id: str, body: dict[str, Any],
                *, scope: Scope | None = None, visible_to: Side = Side.BOTH,
                occurred_at: datetime | None = None,
                phase: Phase | None = None) -> CareEvent:
        self._require_permission(event_type, actor_id)
        moment = occurred_at or self._clock()
        if moment.tzinfo is None:
            raise CareServiceError("事件时间必须带时区")
        event = CareEvent(
            event_id=new_event_id(),
            event_type=event_type,
            occurred_at=moment,
            recorded_at=moment if occurred_at is None else self._clock(),
            actor_id=actor_id,
            phase=phase or self.current_phase(),
            body=body,
            scope=scope or Scope.global_scope(),
            visible_to=visible_to,
        )
        self.ledger.append(event)
        return self.ledger.require(event.event_id)

    # ---- 阶段 -----------------------------------------------------------

    def current_phase(self) -> Phase:
        gates = self.ledger.filter(lambda e: e.event_type == EventType.PHASE_GATE)
        if not gates:
            return Phase.PRE_DEPARTURE
        return Phase(gates[-1].body["to_phase"])

    def advance_phase(self, authority_actor_id: str, to_phase: Phase,
                      basis_event_ids: Iterable[str], *, occurred_at: datetime | None = None) -> CareEvent:
        self._require_permission(EventType.PHASE_GATE, authority_actor_id)
        current = self.current_phase()
        expected_index = PHASE_ORDER.index(current) + 1
        target_index = PHASE_ORDER.index(to_phase)
        if target_index != expected_index:
            raise PhaseGateError(f"阶段只能向前推进一格: {current.value} → {to_phase.value}")
        basis = frozenset(basis_event_ids)
        if not basis:
            raise PhaseGateError("阶段推进必须附依据事件")
        for event_id in basis:
            if self.ledger.get(event_id) is None:
                raise PhaseGateError(f"依据事件不存在: {event_id}")

        view = self.view_at(self._clock(), Side.BOTH)
        for effect in view.active_effects:
            # 全局事件（如整体检疫要求）未恢复，门禁一律不放行
            if effect.scope.is_global:
                raise EffectActiveError(
                    f"{effect.kind.value} 尚未恢复（恢复条件: {effect.recovery_condition}），不能推进到 {to_phase.value}"
                )
            # 阶段级事件（如航线延误）未恢复，不能离开它开始时所在的阶段
            if effect.scope.phase_limited:
                started = self.ledger.require(effect.event_id)
                if started.phase == current:
                    raise EffectActiveError(
                        f"{effect.kind.value} 尚未恢复（恢复条件: {effect.recovery_condition}），"
                        f"不能离开 {current.value} 阶段"
                    )
        # 检疫有时效：任一动物检疫未解除，不得进入航程或入园阶段。
        # 注意：单只隔离只影响该动物的喂养与交接，不阻断整体阶段推进。
        if to_phase in (Phase.FLIGHT, Phase.ADAPTATION):
            for effect in view.active_effects:
                if effect.kind is EffectKind.QUARANTINE and effect.scope.animal_ids:
                    raise EffectActiveError(
                        f"动物 {sorted(effect.scope.animal_ids)} 的检疫未解除，"
                        f"恢复条件: {effect.recovery_condition}"
                    )

        from .models import PhaseGate
        gate = PhaseGate(current, to_phase, basis)
        body = {
            "from_phase": gate.from_phase.value,
            "to_phase": gate.to_phase.value,
            "basis_event_ids": sorted(gate.basis_event_ids),
        }
        return self._append(EventType.PHASE_GATE, authority_actor_id, body, occurred_at=occurred_at)

    # ---- 照护计划 -------------------------------------------------------

    def create_care_plan(self, actor_id: str, plan_id: str,
                         items: list[dict[str, Any]], *,
                         occurred_at: datetime | None = None) -> CareEvent:
        if self._latest_plan_body() is not None:
            raise CareServiceError("照护计划已存在，请使用修订")
        self._validate_plan_items(items)
        return self._append(
            EventType.CARE_PLAN_CREATED, actor_id,
            {"plan_id": plan_id, "items": items, "revision": 1},
            occurred_at=occurred_at,
        )

    def revise_care_plan(self, actor_id: str, items: list[dict[str, Any]],
                         reason: str, *, occurred_at: datetime | None = None) -> CareEvent:
        plan = self._latest_plan_body()
        if plan is None:
            raise CareServiceError("照护计划尚未建立")
        self._validate_plan_items(items)
        body = {
            "plan_id": plan["plan_id"],
            "items": items,
            "revision": plan["revision"] + 1,
            "revision_of_revision": plan["revision"],
            "reason": reason,
        }
        return self._append(EventType.CARE_PLAN_REVISED, actor_id, body, occurred_at=occurred_at)

    @staticmethod
    def _validate_plan_items(items: list[dict[str, Any]]) -> None:
        if not items:
            raise CareServiceError("照护计划条目不能为空")
        seen: set[str] = set()
        for item in items:
            for key in ("plan_item_id", "animal_id", "item", "batch_id", "amount_kg"):
                if key not in item or item[key] in (None, ""):
                    raise CareServiceError(f"计划条目缺少 {key}")
            if item["plan_item_id"] in seen:
                raise CareServiceError(f"计划条目重复: {item['plan_item_id']}")
            seen.add(item["plan_item_id"])
            if float(item["amount_kg"]) <= 0:
                raise CareServiceError("投喂量必须为正")

    def _latest_plan_body(self) -> dict[str, Any] | None:
        plan_events = self.ledger.filter(
            lambda e: e.event_type in (EventType.CARE_PLAN_CREATED, EventType.CARE_PLAN_REVISED)
        )
        return plan_events[-1].body if plan_events else None

    # ---- 评估与兽医结论 -------------------------------------------------

    def record_health_assessment(self, actor_id: str, animal_id: str, summary: str,
                                 fit: bool, *, visible_to: Side = Side.BOTH,
                                 occurred_at: datetime | None = None) -> CareEvent:
        self._require_animal(animal_id)
        return self._append(
            EventType.HEALTH_ASSESSMENT, actor_id,
            {"animal_id": animal_id, "summary": summary, "fit": fit},
            scope=Scope.animals(animal_id), visible_to=visible_to, occurred_at=occurred_at,
        )

    def record_vet_conclusion(self, actor_id: str, conclusion: VetConclusion, *,
                              visible_to: Side = Side.BOTH,
                              occurred_at: datetime | None = None) -> CareEvent:
        self._require_animal(conclusion.animal_id)
        if not conclusion.finding.strip():
            raise CareServiceError("兽医结论内容不能为空")
        body: dict[str, Any] = {
            "animal_id": conclusion.animal_id,
            "topic": conclusion.topic,
            "finding": conclusion.finding,
            "confirmed": conclusion.confirmed,
            "supersedes": conclusion.supersedes,
        }
        if conclusion.supersedes is not None:
            old = self.ledger.get(conclusion.supersedes)
            if old is None or old.event_type != EventType.VET_CONCLUSION:
                raise CareServiceError("被修正结论不存在或不是兽医结论")
            if old.body["animal_id"] != conclusion.animal_id or old.body["topic"] != conclusion.topic:
                raise CareServiceError("修正结论必须针对同一动物同一议题")
            if self._is_superseded(old):
                raise CareServiceError("被修正结论已被更新的结论取代，不能重复修正")
            body["supersedes"] = conclusion.supersedes
        return self._append(
            EventType.VET_CONCLUSION, actor_id, body,
            scope=Scope.animals(conclusion.animal_id),
            visible_to=visible_to, occurred_at=occurred_at,
        )

    def _is_conclusion_active(self, event: CareEvent) -> bool:
        if not event.body.get("confirmed", False):
            return False
        if self._is_superseded(event):
            return False
        if event.event_id in self._conclusions_under_review():
            return False  # 复核期间挂起，不作为照护依据
        return True

    def _is_superseded(self, event: CareEvent) -> bool:
        newer = self.ledger.filter(
            lambda e: e.event_type == EventType.VET_CONCLUSION and (e.seq or -1) > (event.seq or -1)
        )
        return any(e.body.get("supersedes") == event.event_id for e in newer)

    def _conclusions_under_review(self) -> frozenset[str]:
        """尚未被 REVIEW_RESOLVED 闭环的复核所挂起的兽医结论。"""
        reviews = self.ledger.filter(lambda e: e.event_type == EventType.REVIEW_OPENED)
        resolved = {
            e.body["review_event_id"]
            for e in self.ledger.filter(lambda x: x.event_type == EventType.REVIEW_RESOLVED)
        }
        suspended: set[str] = set()
        for review in reviews:
            if review.event_id in resolved:
                continue
            suspended.update(review.body["conflicts_with"])
        return frozenset(suspended)

    def active_vet_conclusions(self, animal_id: str | None = None) -> tuple[CareEvent, ...]:
        return tuple(
            e for e in self.ledger.filter(lambda x: x.event_type == EventType.VET_CONCLUSION)
            if self._is_conclusion_active(e) and (animal_id is None or e.body["animal_id"] == animal_id)
        )

    # ---- 装运与航程 -----------------------------------------------------

    def seal_crate(self, actor_id: str, animal_ids: list[str], crate_id: str,
                   seal_ids: list[str], manifest_hash: str, *,
                   occurred_at: datetime | None = None) -> CareEvent:
        if self.current_phase() != Phase.CRATING:
            raise PhaseGateError("只能在装运阶段封存笼具")
        if not animal_ids or not seal_ids:
            raise CareServiceError("封存必须列明动物与封识编号")
        for animal_id in animal_ids:
            self._require_animal(animal_id)
        return self._append(
            EventType.CRATE_SEALED, actor_id,
            {"crate_id": crate_id, "seal_ids": seal_ids,
             "animal_ids": sorted(animal_ids), "manifest_hash": manifest_hash},
            scope=Scope(animal_ids=frozenset(animal_ids)),
            occurred_at=occurred_at,
        )

    def record_flight_event(self, actor_id: str, kind: str, detail: str, *,
                            flight_no: str | None = None,
                            occurred_at: datetime | None = None) -> CareEvent:
        if self.current_phase() != Phase.FLIGHT:
            raise PhaseGateError("航程事件只能发生在航程阶段")
        return self._append(
            EventType.FLIGHT_EVENT, actor_id,
            {"kind": kind, "detail": detail, "flight_no": flight_no},
            scope=Scope(phase_limited=True), occurred_at=occurred_at,
        )

    # ---- 日常照护：喂养 / 用药 / 观察 -----------------------------------

    def record_feeding(self, actor_id: str, feeding: Feeding, *,
                       occurred_at: datetime | None = None) -> CareEvent:
        self._require_animal(feeding.animal_id)
        moment = occurred_at or self._clock()
        plan = self._plan_as_of(moment)
        item = self._resolve_plan_item(plan, feeding.plan_item_id, feeding.animal_id)
        view = self.view_at(moment, Side.BOTH)

        sub = view.find_effect(EffectKind.FEED_SUBSTITUTION, plan_item_id=feeding.plan_item_id)
        if sub is not None:
            substitute_batch = (sub.substitute or {}).get("batch_id")
            if feeding.batch_id == item["batch_id"]:
                raise EffectActiveError(
                    f"原批次 {item['batch_id']} 已停用（恢复条件: {sub.recovery_condition}），"
                    f"请使用替代批次 {substitute_batch}"
                )
            if substitute_batch and feeding.batch_id != substitute_batch:
                raise EffectActiveError(
                    f"替代期间只能使用批次 {substitute_batch}，实际 {feeding.batch_id}"
                )
        elif feeding.batch_id != item["batch_id"] or feeding.item != item["item"]:
            raise CareServiceError("喂养记录与现行照护计划不符；如需变更请登记饲料替代事件")

        isolation = view.find_effect(EffectKind.ISOLATION, animal_id=feeding.animal_id)
        body = {
            "feeding": feeding.__dict__,
            "under_isolation": isolation is not None,
            "substitute_batch": sub.substitute if sub else None,
        }
        return self._append(
            EventType.FEEDING, actor_id, body,
            scope=Scope.animals(feeding.animal_id), occurred_at=occurred_at,
        )

    def record_medication(self, actor_id: str, medication: Medication, *,
                          occurred_at: datetime | None = None) -> CareEvent:
        self._require_animal(medication.animal_id)
        actor = self._actor(actor_id)
        keeper = Role.CN_KEEPER in actor.roles or Role.US_KEEPER in actor.roles
        if keeper and medication.per_vet_conclusion_id is None:
            raise PermissionDenied("饲养员用药必须引用现行兽医结论作为用药依据")
        if medication.per_vet_conclusion_id is not None:
            order = self.ledger.get(medication.per_vet_conclusion_id)
            if order is None or order.event_type != EventType.VET_CONCLUSION:
                raise CareServiceError("用药依据不是兽医结论")
            if not self._is_conclusion_active(order):
                raise CareServiceError("用药依据的兽医结论已被修正，请等待新结论")
            if order.body["animal_id"] != medication.animal_id:
                raise CareServiceError("用药动物与医嘱动物不一致")
        body = {"medication": self._dataclass_body(medication)}
        return self._append(
            EventType.MEDICATION, actor_id, body,
            scope=Scope.animals(medication.animal_id), occurred_at=occurred_at,
        )

    def record_observation(self, actor_id: str, observation: Observation, *,
                           visible_to: Side = Side.BOTH,
                           occurred_at: datetime | None = None) -> CareEvent:
        self._require_animal(observation.animal_id)
        body: dict[str, Any] = {"observation": self._dataclass_body(observation)}
        if observation.kind == ObservationKind.WEIGHT:
            try:
                body["weight_kg"] = float(observation.value)
            except (TypeError, ValueError) as exc:
                raise CareServiceError("体重观察的值必须是数字（千克）") from exc
            if body["weight_kg"] <= 0:
                raise CareServiceError("体重必须为正")
        if observation.abnormal and not observation.note.strip():
            raise CareServiceError("行为或临床异常必须附说明")
        return self._append(
            EventType.OBSERVATION, actor_id, body,
            scope=Scope.animals(observation.animal_id),
            visible_to=visible_to, occurred_at=occurred_at,
        )

    @staticmethod
    def _resolve_plan_item(plan: dict[str, Any] | None, plan_item_id: str, animal_id: str) -> dict[str, Any]:
        if plan is None:
            raise CareServiceError("照护计划尚未建立")
        for item in plan["items"]:
            if item["plan_item_id"] == plan_item_id:
                if item["animal_id"] != animal_id:
                    raise CareServiceError("计划条目属于另一只动物")
                return item
        raise CareServiceError(f"计划条目不存在: {plan_item_id}")

    @staticmethod
    def _dataclass_body(value: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, field_value in value.__dict__.items():
            if isinstance(field_value, ObservationKind):
                result[key] = field_value.value
            else:
                result[key] = field_value
        return result

    # ---- 物资批次 -------------------------------------------------------

    def dispatch_batch(self, actor_id: str, batch: BatchRef,
                       from_location: str, to_location: str,
                       custody_actor_id: str, reason: str = "", *,
                       occurred_at: datetime | None = None) -> CareEvent:
        self._actor(custody_actor_id)
        body = {
            "batch": self._dataclass_body(batch),
            "from_location": from_location,
            "to_location": to_location,
            "custody_actor_id": custody_actor_id,
            "reason": reason,
        }
        return self._append(
            EventType.BATCH_DISPATCHED, actor_id, body,
            scope=Scope(batch_ids=frozenset({batch.batch_id})),
            occurred_at=occurred_at,
        )

    # ---- 范围化事件（检疫/延误/替代/隔离） -------------------------------

    def start_effect(self, actor_id: str, kind: EffectKind, scope: Scope,
                     recovery_condition: str, *, substitute: dict[str, Any] | None = None,
                     detail: str = "", occurred_at: datetime | None = None) -> CareEvent:
        if not recovery_condition.strip():
            raise CareServiceError("范围化事件必须写明恢复条件")
        if kind is EffectKind.FEED_SUBSTITUTION:
            if not scope.plan_item_ids:
                raise CareServiceError("饲料替代必须挂接到具体照护计划条目")
            if not substitute or not substitute.get("batch_id"):
                raise CareServiceError("饲料替代必须给出替代批次")
        if kind in (EffectKind.QUARANTINE, EffectKind.ISOLATION):
            if not scope.animal_ids:
                raise CareServiceError(f"{kind.value} 必须挂接到具体动物")
            for animal_id in scope.animal_ids:
                self._require_animal(animal_id)
        if kind is EffectKind.FLIGHT_DELAY and not scope.phase_limited and not scope.is_global:
            raise CareServiceError("航线延误应挂接航程阶段")
        body = {
            "kind": kind.value,
            "recovery_condition": recovery_condition,
            "detail": detail,
            "substitute": substitute,
        }
        return self._append(EventType.EFFECT_STARTED, actor_id, body,
                            scope=scope, occurred_at=occurred_at)

    def resolve_effect(self, actor_id: str, effect_event_id: str,
                       resolution_evidence: str, *,
                       occurred_at: datetime | None = None) -> CareEvent:
        started = self.ledger.require(effect_event_id)
        if started.event_type != EventType.EFFECT_STARTED:
            raise CareServiceError("解除目标不是范围化事件")
        if not resolution_evidence.strip():
            raise CareServiceError("解除必须留存满足恢复条件的证据")
        already = self.ledger.filter(
            lambda e: e.event_type == EventType.EFFECT_RESOLVED
            and e.body.get("effect_event_id") == effect_event_id
        )
        if already:
            raise CareServiceError("该事件已恢复，不能重复解除")
        body = {
            "effect_event_id": effect_event_id,
            "kind": started.body["kind"],
            "recovery_condition": started.body["recovery_condition"],
            "resolution_evidence": resolution_evidence,
        }
        return self._append(EventType.EFFECT_RESOLVED, actor_id, body,
                            scope=started.scope, occurred_at=occurred_at)

    # ---- 责任交接 -------------------------------------------------------

    def transfer_care(self, actor_id: str, transfer: TransferOfCare, *,
                      occurred_at: datetime | None = None) -> CareEvent:
        moment = occurred_at or self._clock()
        self._actor(transfer.from_actor_id)
        self._actor(transfer.to_actor_id)
        if not transfer.responsibilities:
            raise CareServiceError("交接必须列明移交责任")
        acknowledgers = [self._actor(aid) for aid in transfer.acknowledged_by]
        if not acknowledgers:
            raise CareServiceError("交接必须有确认人")
        if transfer.cross_border:
            sides = {a.side for a in acknowledgers}
            if Side.CN not in sides or Side.US not in sides:
                raise PermissionDenied("跨境交接必须由中美双方责任人共同确认")
        view = self.view_at(moment, Side.BOTH)
        for animal_id in transfer.animal_ids:
            self._require_animal(animal_id)
            blocking = (view.find_effect(EffectKind.ISOLATION, animal_id=animal_id)
                        or view.find_effect(EffectKind.QUARANTINE, animal_id=animal_id))
            if blocking is not None:
                raise EffectActiveError(
                    f"动物 {animal_id} 处于 {blocking.kind.value}，恢复条件未满足前不能交接: "
                    f"{blocking.recovery_condition}"
                )
        body = {
            "transfer": {
                "checkpoint": transfer.checkpoint,
                "from_actor_id": transfer.from_actor_id,
                "to_actor_id": transfer.to_actor_id,
                "responsibilities": list(transfer.responsibilities),
                "acknowledged_by": sorted(transfer.acknowledged_by),
                "animal_ids": sorted(transfer.animal_ids),
                "cross_border": transfer.cross_border,
                "manifest_hash": transfer.manifest_hash,
            }
        }
        return self._append(
            EventType.TRANSFER_OF_CARE, actor_id, body,
            scope=Scope(animal_ids=transfer.animal_ids) if transfer.animal_ids else Scope.global_scope(),
            occurred_at=occurred_at,
        )

    # ---- 报告上传：幂等 + 矛盾复核 --------------------------------------

    def upload_report(self, uploader_actor_id: str, title: str, content: bytes | str,
                      claims: list[dict[str, str]], *,
                      occurred_at: datetime | None = None) -> UploadResult:
        """上传检疫/健康/航程报告。

        claims 为报告内的关键结论：[{"animal_id", "topic", "finding"}]。
        与任一现行兽医结论矛盾时，转指定兽医复核，未决前冲突结论挂起。
        """
        self._require_permission(EventType.REPORT_UPLOADED, uploader_actor_id)
        _require_claims(claims)
        if isinstance(content, str):
            content = content.encode("utf-8")
        content_hash = hashlib.sha256(content).hexdigest()

        cached = self.idempotency.get(uploader_actor_id, content_hash)
        if cached is not None:
            if cached.status is UploadStatus.REVIEW_PENDING:
                status = UploadStatus.REVIEWING
                message = "报告已在指定兽医复核中，返回原处理中的结果"
            else:
                status = cached.status
                message = f"报告已处理过，返回原结果（{cached.status.value}）"
            return UploadResult(
                status=status, event_id=cached.event_id,
                review_event_id=cached.review_event_id, duplicate=True,
                message=message,
            )

        conflicts = self._detect_conflicts(claims)
        if conflicts and self._designated_vet_id is None:
            raise CareServiceError("尚未指定复核兽医，无法处理矛盾报告")
        report_event = self._append(
            EventType.REPORT_UPLOADED, uploader_actor_id,
            {"title": title, "content_hash": content_hash, "claims": claims,
             "conflicts_with": sorted(conflicts)},
            occurred_at=occurred_at,
        )

        if conflicts:
            review_body = {
                "report_event_id": report_event.event_id,
                "content_hash": content_hash,
                "conflicts_with": sorted(conflicts),
                "claims": claims,
                "status": "OPEN",
                "designated_vet_id": self._designated_vet_id,
            }
            review_event = self._append(EventType.REVIEW_OPENED, self._designated_vet_id, review_body)
            result = UploadResult(
                status=UploadStatus.REVIEW_PENDING, event_id=report_event.event_id,
                review_event_id=review_event.event_id, duplicate=False,
                message="报告与已确认兽医结论矛盾，已转指定兽医复核；冲突结论在复核期间挂起",
            )
        else:
            result = UploadResult(
                status=UploadStatus.ACCEPTED, event_id=report_event.event_id,
                review_event_id=None, duplicate=False,
                message="报告已归档",
            )
        self.idempotency.remember(uploader_actor_id, content_hash, result)
        return result

    def resolve_review(self, vet_actor_id: str, review_event_id: str,
                       decision: str, new_finding: str | None = None, *,
                       occurred_at: datetime | None = None) -> CareEvent:
        """指定兽医复核结论。

        decision="ACCEPT_REPORT"：以新兽医结论取代冲突结论（旧结论保留在链上）。
        decision="REJECT_REPORT"：维持原结论，报告归档但不作为照护依据。
        """
        actor = self._require_permission(EventType.REVIEW_RESOLVED, vet_actor_id)
        review = self.ledger.require(review_event_id)
        if review.event_type != EventType.REVIEW_OPENED:
            raise CareServiceError("目标不是复核事件")
        if review.body["designated_vet_id"] != actor.actor_id:
            raise PermissionDenied("只有被指派的指定兽医可以作出复核结论")
        already = self.ledger.filter(
            lambda e: e.event_type == EventType.REVIEW_RESOLVED
            and e.body.get("review_event_id") == review_event_id
        )
        if already:
            raise CareServiceError("该复核已作出结论（结论只追加，不可改写）")
        if decision not in ("ACCEPT_REPORT", "REJECT_REPORT"):
            raise CareServiceError("复核决定必须是 ACCEPT_REPORT 或 REJECT_REPORT")

        superseded_events: list[str] = []
        new_conclusion_event_id: str | None = None
        if decision == "ACCEPT_REPORT":
            if not new_finding:
                raise CareServiceError("采纳报告必须给出新的兽医结论文本")
            # 每条冲突结论对应一个同议题的新结论
            report = self.ledger.require(review.body["report_event_id"])
            claims_by_key = {
                (c["animal_id"], c["topic"]): c["finding"] for c in report.body["claims"]
            }
            for old_id in review.body["conflicts_with"]:
                old = self.ledger.require(old_id)
                finding = new_finding if len(review.body["conflicts_with"]) == 1 else claims_by_key.get(
                    (old.body["animal_id"], old.body["topic"]), new_finding
                )
                new_conclusion = self.record_vet_conclusion(
                    vet_actor_id,
                    VetConclusion(
                        animal_id=old.body["animal_id"], topic=old.body["topic"],
                        finding=finding, confirmed=True, supersedes=old_id,
                    ),
                    occurred_at=occurred_at,
                )
                superseded_events.append(old_id)
                new_conclusion_event_id = new_conclusion.event_id

        resolved = self._append(
            EventType.REVIEW_RESOLVED, vet_actor_id,
            {"review_event_id": review_event_id,
             "report_event_id": review.body["report_event_id"],
             "decision": decision,
             "superseded_conclusions": superseded_events,
             "new_conclusion_event_id": new_conclusion_event_id},
            occurred_at=occurred_at,
        )
        # 更新幂等表：此后重复上传该报告得到复核终态（仍不产生新事件）
        report = self.ledger.require(review.body["report_event_id"])
        uploader = report.actor_id
        content_hash = report.body["content_hash"]
        self.idempotency.replace(
            uploader, content_hash,
            UploadResult(
                status=UploadStatus.REVIEW_ACCEPTED if decision == "ACCEPT_REPORT"
                else UploadStatus.REVIEW_REJECTED,
                event_id=report.event_id, review_event_id=review_event_id,
                duplicate=False,
                message="复核完成：报告被采纳，原结论已被新结论修正"
                if decision == "ACCEPT_REPORT" else "复核完成：报告被驳回，维持原兽医结论",
            ),
        )
        return resolved

    def _detect_conflicts(self, claims: list[dict[str, str]]) -> frozenset[str]:
        conflicts: set[str] = set()
        for claim in claims:
            animal_id, topic, finding = claim["animal_id"], claim["topic"], claim["finding"]
            for conclusion in self.active_vet_conclusions(animal_id):
                if conclusion.body["topic"] != topic:
                    continue
                if conclusion.body["finding"].strip() != finding.strip():
                    conflicts.add(conclusion.event_id)
        return frozenset(conflicts)

    # ---- 侧别视图与时点回放 ---------------------------------------------

    def view_at(self, moment: datetime, side: Side) -> ReplayView:
        events = self.ledger.as_of(moment)
        visible = tuple(e for e in events if self._visible(e, side))

        phase = Phase.PRE_DEPARTURE
        plan: dict[str, Any] | None = None
        active: dict[str, CareEvent] = {}
        weights: dict[str, float] = {}
        batches: list[dict[str, Any]] = []
        transfers: list[dict[str, Any]] = []
        conclusions: list[dict[str, Any]] = []
        reviews: dict[str, CareEvent] = {}

        for event in visible:
            t = event.event_type
            if t == EventType.PHASE_GATE:
                phase = Phase(event.body["to_phase"])
            elif t in (EventType.CARE_PLAN_CREATED, EventType.CARE_PLAN_REVISED):
                plan = event.body
            elif t == EventType.EFFECT_STARTED:
                active[event.event_id] = event
            elif t == EventType.EFFECT_RESOLVED:
                active.pop(event.body["effect_event_id"], None)
            elif t == EventType.OBSERVATION and "weight_kg" in event.body:
                weights[event.body["observation"]["animal_id"]] = event.body["weight_kg"]
            elif t == EventType.BATCH_DISPATCHED:
                batches.append({"seq": event.seq, "at": event.occurred_at, **event.body})
            elif t == EventType.TRANSFER_OF_CARE:
                transfers.append({"seq": event.seq, "at": event.occurred_at, **event.body})
            elif t == EventType.VET_CONCLUSION:
                conclusions.append({"seq": event.seq, **event.body})
            elif t == EventType.REVIEW_OPENED:
                reviews[event.event_id] = event
            elif t == EventType.REVIEW_RESOLVED:
                reviews.pop(event.body["review_event_id"], None)

        active_effects = tuple(
            ActiveEffect(
                event_id=e.event_id,
                kind=EffectKind(e.body["kind"]),
                scope=e.scope,
                recovery_condition=e.body["recovery_condition"],
                started_at=e.occurred_at,
                substitute=e.body.get("substitute"),
            )
            for e in active.values()
        )
        head_hash = visible[-1].hash if visible else ("0" * 64)
        return ReplayView(
            as_of=moment.astimezone(), side=side, phase=phase,
            plan=plan or {"plan_id": None, "items": [], "revision": 0},
            active_effects=active_effects,
            latest_weight_kg=dict(weights),
            batch_movements=tuple(batches),
            transfers=tuple(transfers),
            vet_conclusions=tuple(conclusions),
            open_reviews=tuple(
                {"seq": e.seq, **e.body} for e in reviews.values()
            ),
            events=visible,
            head_hash=head_hash,
        )

    @staticmethod
    def _visible(event: CareEvent, side: Side) -> bool:
        if side is Side.BOTH or event.visible_to is Side.BOTH:
            return True
        return event.visible_to is side

    def _plan_as_of(self, moment: datetime) -> dict[str, Any] | None:
        return self.view_at(moment, Side.BOTH).plan

    # ---- 争议导出 -------------------------------------------------------

    def dispute_export(self, moment: datetime) -> dict[str, Any]:
        """生成争议时点证据包：中美双方拿到同一规范文件。

        包含双方可见事件全集（带侧别标注）、链校验摘要、
        当时的照护计划、未恢复事件、物资去向与交接依据。
        """
        events = self.ledger.as_of(moment)
        cn_view = self.view_at(moment, Side.CN)
        us_view = self.view_at(moment, Side.US)
        head_hash = events[-1].hash if events else "0" * 64
        return {
            "domain": "panda-transit-care",
            "as_of": moment.astimezone().isoformat(),
            "phase": self.view_at(moment, Side.BOTH).phase.value,
            "head_hash": head_hash,
            "event_count": len(events),
            "events": [
                {
                    "seq": e.seq,
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "occurred_at": e.occurred_at.isoformat(),
                    "recorded_at": e.recorded_at.isoformat(),
                    "actor_id": e.actor_id,
                    "phase": e.phase.value,
                    "visible_to": e.visible_to.value,
                    "body": e.body,
                    "hash": e.hash,
                    "prev_hash": e.prev_hash,
                }
                for e in events
            ],
            "snapshots": {
                "BOTH": self._snapshot(self.view_at(moment, Side.BOTH)),
                "CN": self._snapshot(cn_view),
                "US": self._snapshot(us_view),
            },
        }

    @staticmethod
    def _snapshot(view: ReplayView) -> dict[str, Any]:
        return {
            "phase": view.phase.value,
            "plan_revision": view.plan.get("revision", 0),
            "plan_items": view.plan.get("items", []),
            "active_effects": [
                {"event_id": e.event_id, "kind": e.kind.value,
                 "scope": e.scope.to_dict(), "recovery_condition": e.recovery_condition}
                for e in view.active_effects
            ],
            "latest_weight_kg": view.latest_weight_kg,
            "batch_movements": [
                {"seq": b["seq"], "batch_id": b["batch"]["batch_id"],
                 "from_location": b["from_location"], "to_location": b["to_location"],
                 "custody_actor_id": b["custody_actor_id"]}
                for b in view.batch_movements
            ],
            "transfers": [
                {"seq": t["seq"], "checkpoint": t["transfer"]["checkpoint"],
                 "from_actor_id": t["transfer"]["from_actor_id"],
                 "to_actor_id": t["transfer"]["to_actor_id"],
                 "acknowledged_by": t["transfer"]["acknowledged_by"],
                 "manifest_hash": t["transfer"]["manifest_hash"]}
                for t in view.transfers
            ],
        }

    def verify_integrity(self) -> None:
        """对外暴露的链完整性校验入口。"""
        self.ledger.verify()
