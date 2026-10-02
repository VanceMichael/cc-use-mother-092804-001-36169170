"""跨境旅居动物照护档案：只追加记录、兽医结论锁定、作用域偶发事件。

设计要点：

- ``CareLedger`` 保存的每一条记录都带发生时间、记录时间与责任人，并通过
  前驱哈希串成只追加链；任何改写都会在 :meth:`CareLedger.verify_chain` 时暴露。
- 喂养、用药、体重、行为异常是普通照护观察；兽医结论须由兽医角色建立，
  确认后只能由兽医以"纠正记录"追加取代，原结论永远保留。
- 检疫扣留、航线延误、饲料替代、单只隔离是带作用域（动物 × 环节）和恢复
  条件的偶发事件，只影响命中的环节，并在恢复条件满足后关闭。
- 报告按内容哈希幂等受理：重复提交返回原结果；与已确认兽医结论矛盾的报告
  进入指定兽医的复核队列，不改变既有结论。
- 中方、美方各有可见范围；争议时由共同管理机构按记录时点回放档案。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class Side(str, Enum):
    CN = "CN"
    US = "US"
    JOINT = "JOINT"  # 共同管理机构，争议回放时可见全部


class Role(str, Enum):
    OPERATIONS = "动物园运营"
    KEEPER = "饲养员"
    VET = "兽医"
    CARRIER = "承运方"
    AUTHORITY = "管理机构"


class Stage(str, Enum):
    PRE_DEPARTURE = "出发前评估"
    CRATING = "装运封存"
    TRANSIT = "航程"
    QUARANTINE = "入境检疫"
    ADAPTATION = "入园适应"
    RESIDENCE = "十年旅居"


class RecordType(str, Enum):
    FEEDING = "喂养"
    MEDICATION = "用药"
    WEIGHT = "体重"
    BEHAVIOR = "行为"
    VET_CONCLUSION = "兽医结论"
    VET_CORRECTION = "兽医纠正"
    REPORT = "报告受理"
    REVIEW = "复核裁决"
    CONTINGENCY = "偶发事件"
    CONTINGENCY_CLOSE = "事件恢复"
    BATCH_RECEIPT = "饲料批次入库"
    PLAN = "照护计划"
    HANDOFF = "交接"


class Visibility(str, Enum):
    SHARED = "中美共享"
    CN_ONLY = "仅中方"
    US_ONLY = "仅美方"


class ContingencyKind(str, Enum):
    QUARANTINE_HOLD = "检疫扣留"
    FLIGHT_DELAY = "航线延误"
    FEED_SUBSTITUTION = "饲料替代"
    ISOLATION = "单只隔离"


class ReportStatus(str, Enum):
    ACCEPTED = "已受理"
    PENDING_REVIEW = "待兽医复核"
    REJECTED = "复核驳回"


# ---------------------------------------------------------------------------
# 基础对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Actor:
    id: str
    name: str
    side: Side
    roles: frozenset[Role]


@dataclass(frozen=True)
class Animal:
    id: str
    name: str


@dataclass(frozen=True)
class CareRecord:
    seq: int
    type: RecordType
    occurred_at: datetime
    recorded_at: datetime
    actor_id: str
    animal_ids: tuple[str, ...]
    stage: Stage | None
    visibility: Visibility
    payload: Mapping[str, Any]
    prev_hash: str
    hash: str


@dataclass(frozen=True)
class SubmitResult:
    record: CareRecord
    status: ReportStatus
    reviewer_id: str | None = None
    duplicate: bool = False  # 重复提交时为 True，record 指向首次受理记录


@dataclass(frozen=True)
class Snapshot:
    """某个记录时点（含该时点之前）的档案回放结果。"""

    as_of: datetime
    records: tuple[CareRecord, ...]
    plans: Mapping[Stage, CareRecord]
    active_contingencies: tuple[CareRecord, ...]
    batch_balances: Mapping[str, float]
    active_conclusions: Mapping[str, CareRecord]
    open_reviews: tuple[CareRecord, ...]
    chain_ok: bool


def _aware(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label}必须带时区")
    return value


def _digest(record_material: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        record_material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda obj: obj.isoformat() if isinstance(obj, datetime) else obj,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 档案
# ---------------------------------------------------------------------------


class CareLedger:
    def __init__(self) -> None:
        self.actors: dict[str, Actor] = {}
        self.animals: dict[str, Animal] = {}
        self.records: list[CareRecord] = []
        self._reviewers: dict[str, str] = {}  # 动物编号（"*" 为默认）→ 兽医编号
        self._report_index: dict[str, int] = {}  # 报告内容哈希 → 记录序号
        self._review_queue: dict[int, str] = {}  # 报告记录序号 → 指定复核兽医

    # -- 参与方 -------------------------------------------------------------

    def register_actor(
        self, actor_id: str, name: str, side: Side, roles: Iterable[Role]
    ) -> Actor:
        roles = frozenset(roles)
        if not roles:
            raise ValueError("参与方至少需要一个角色")
        actor = Actor(actor_id, name, side, frozenset(roles))
        self.actors[actor_id] = actor
        return actor

    def register_animal(self, animal_id: str, name: str) -> Animal:
        animal = Animal(animal_id, name)
        self.animals[animal_id] = animal
        return animal

    def designate_reviewer(self, vet_id: str, animal_id: str = "*") -> None:
        self._require_role(vet_id, Role.VET)
        self._reviewers[animal_id] = vet_id

    def _actor(self, actor_id: str) -> Actor:
        try:
            return self.actors[actor_id]
        except KeyError:
            raise ValueError(f"未知责任人：{actor_id}") from None

    def _require_role(self, actor_id: str, role: Role) -> Actor:
        actor = self._actor(actor_id)
        if role not in actor.roles:
            raise PermissionError(f"{actor.name}不具备{role.value}角色")
        return actor

    def _animals(self, animal_ids: Iterable[str], allow_empty: bool = False) -> tuple[str, ...]:
        ids = tuple(animal_ids)
        if not ids and not allow_empty:
            raise ValueError("记录必须关联至少一只动物")
        for animal_id in ids:
            if animal_id not in self.animals:
                raise ValueError(f"未知动物：{animal_id}")
        if len(set(ids)) != len(ids):
            raise ValueError("动物作用域重复")
        return ids

    # -- 追加 ---------------------------------------------------------------

    def _append(
        self,
        type_: RecordType,
        occurred_at: datetime,
        actor_id: str,
        animal_ids: tuple[str, ...],
        stage: Stage | None,
        visibility: Visibility,
        payload: Mapping[str, Any],
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        _aware(occurred_at, "发生时间")
        # 现场同步采集时记录时间默认等于发生时间；补录/迟报必须显式传入 recorded_at。
        recorded_at = _aware(recorded_at or occurred_at, "记录时间")
        self._actor(actor_id)
        seq = len(self.records) + 1
        prev_hash = self.records[-1].hash if self.records else ""
        material: dict[str, Any] = {
            "seq": seq,
            "type": type_.value,
            "occurred_at": occurred_at,
            "recorded_at": recorded_at,
            "actor_id": actor_id,
            "animal_ids": list(animal_ids),
            "stage": stage.value if stage else None,
            "visibility": visibility.value,
            "payload": dict(payload),
            "prev_hash": prev_hash,
        }
        record = CareRecord(
            seq=seq,
            type=type_,
            occurred_at=occurred_at,
            recorded_at=recorded_at,
            actor_id=actor_id,
            animal_ids=animal_ids,
            stage=stage,
            visibility=visibility,
            payload=MappingProxyType(dict(payload)),
            prev_hash=prev_hash,
            hash=_digest(material),
        )
        self.records.append(record)
        return record

    # -- 照护计划与物资 ------------------------------------------------------

    def publish_plan(
        self,
        stage: Stage,
        foods: Iterable[str],
        items: Iterable[str],
        actor_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        """发布某环节的照护计划；同环节新版本自然取代旧版本，旧版保留。"""
        foods, items = list(foods), list(items)
        if not items:
            raise ValueError("照护计划内容为空")
        actor = self._actor(actor_id)
        return self._append(
            RecordType.PLAN,
            occurred_at,
            actor_id,
            (),
            stage,
            Visibility.SHARED,
            {"foods": foods, "items": items, "publisher_side": actor.side.value},
            recorded_at,
        )

    def receive_batch(
        self,
        batch_id: str,
        food: str,
        quantity: float,
        actor_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        actor = self._actor(actor_id)
        if not {Role.OPERATIONS, Role.CARRIER, Role.KEEPER} & actor.roles:
            raise PermissionError("饲料批次入库须由运营、承运或饲养角色办理")
        if quantity <= 0:
            raise ValueError("入库数量必须为正")
        if any(r.payload.get("batch_id") == batch_id for r in self._by_type(RecordType.BATCH_RECEIPT)):
            raise ValueError(f"批次号重复：{batch_id}")
        return self._append(
            RecordType.BATCH_RECEIPT,
            occurred_at,
            actor_id,
            (),
            None,
            Visibility.SHARED,
            {"batch_id": batch_id, "food": food, "quantity": float(quantity)},
            recorded_at,
        )

    # -- 日常照护观察 --------------------------------------------------------

    def record_feeding(
        self,
        animal_id: str,
        stage: Stage,
        food: str,
        amount: float,
        actor_id: str,
        occurred_at: datetime,
        batch_id: str | None = None,
        visibility: Visibility = Visibility.SHARED,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_any_role(actor_id, (Role.KEEPER, Role.VET, Role.OPERATIONS))
        animal_ids = self._animals([animal_id])
        if amount <= 0:
            raise ValueError("喂养数量必须为正")
        payload: dict[str, Any] = {"food": food, "amount": float(amount)}
        if batch_id:
            payload["batch_id"] = batch_id
            self._validate_batch_use(batch_id, food, amount, animal_id, stage, occurred_at)
        self._validate_diet(animal_ids[0], stage, food, occurred_at)
        return self._append(
            RecordType.FEEDING, occurred_at, actor_id, animal_ids, stage, visibility, payload, recorded_at
        )

    def record_medication(
        self,
        animal_id: str,
        stage: Stage,
        medication: str,
        dose: str,
        actor_id: str,
        occurred_at: datetime,
        reason: str = "",
        visibility: Visibility = Visibility.SHARED,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_any_role(actor_id, (Role.KEEPER, Role.VET))
        animal_ids = self._animals([animal_id])
        if not medication or not dose:
            raise ValueError("用药记录缺少药品或剂量")
        return self._append(
            RecordType.MEDICATION,
            occurred_at,
            actor_id,
            animal_ids,
            stage,
            visibility,
            {"medication": medication, "dose": dose, "reason": reason},
            recorded_at,
        )

    def record_weight(
        self,
        animal_id: str,
        stage: Stage,
        kg: float,
        actor_id: str,
        occurred_at: datetime,
        visibility: Visibility = Visibility.SHARED,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_any_role(actor_id, (Role.KEEPER, Role.VET, Role.OPERATIONS))
        animal_ids = self._animals([animal_id])
        if kg <= 0:
            raise ValueError("体重数据无效")
        return self._append(
            RecordType.WEIGHT,
            occurred_at,
            actor_id,
            animal_ids,
            stage,
            visibility,
            {"kg": float(kg)},
            recorded_at,
        )

    def record_behavior(
        self,
        animal_ids: Iterable[str],
        stage: Stage,
        note: str,
        anomaly: bool,
        actor_id: str,
        occurred_at: datetime,
        visibility: Visibility = Visibility.SHARED,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_any_role(actor_id, (Role.KEEPER, Role.VET, Role.OPERATIONS))
        ids = self._animals(animal_ids)
        if not note:
            raise ValueError("行为记录内容为空")
        return self._append(
            RecordType.BEHAVIOR,
            occurred_at,
            actor_id,
            ids,
            stage,
            visibility,
            {"note": note, "anomaly": bool(anomaly)},
            recorded_at,
        )

    def _require_any_role(self, actor_id: str, roles: tuple[Role, ...]) -> None:
        actor = self._actor(actor_id)
        if not set(roles) & actor.roles:
            names = "、".join(r.value for r in roles)
            raise PermissionError(f"{actor.name}需要以下角色之一：{names}")

    # -- 兽医结论 -----------------------------------------------------------

    def add_vet_conclusion(
        self,
        animal_id: str,
        stage: Stage,
        summary: str,
        findings: Mapping[str, Any],
        vet_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_role(vet_id, Role.VET)
        animal_ids = self._animals([animal_id])
        if not summary:
            raise ValueError("兽医结论为空")
        return self._append(
            RecordType.VET_CONCLUSION,
            occurred_at,
            vet_id,
            animal_ids,
            stage,
            Visibility.SHARED,
            {"summary": summary, "findings": dict(findings), "status": "已确认"},
            recorded_at,
        )

    def correct_conclusion(
        self,
        conclusion_seq: int,
        summary: str,
        findings: Mapping[str, Any],
        reason: str,
        vet_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        """兽医以追加纠正的方式取代已确认结论；原记录保留且不可改写。"""
        self._require_role(vet_id, Role.VET)
        original = self._record(conclusion_seq)
        if original.type is not RecordType.VET_CONCLUSION:
            raise ValueError("只能纠正兽医结论")
        if self._is_superseded(conclusion_seq):
            raise ValueError("该结论已被后续纠正取代")
        if not reason:
            raise ValueError("纠正必须说明理由")
        return self._append(
            RecordType.VET_CORRECTION,
            occurred_at,
            vet_id,
            original.animal_ids,
            original.stage,
            Visibility.SHARED,
            {
                "supersedes": conclusion_seq,
                "summary": summary,
                "findings": dict(findings),
                "reason": reason,
                "status": "已确认",
            },
            recorded_at,
        )

    # -- 偶发事件（作用域 + 恢复条件）----------------------------------------

    def open_contingency(
        self,
        kind: ContingencyKind,
        segments: Iterable[Stage],
        reason: str,
        recovery_condition: str,
        actor_id: str,
        occurred_at: datetime,
        animal_ids: Iterable[str] = (),
        allowed_foods: Iterable[str] = (),
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._actor(actor_id)
        segments = tuple(segments)
        if not segments:
            raise ValueError("偶发事件必须限定受影响环节")
        if not reason or not recovery_condition:
            raise ValueError("偶发事件必须说明原因与恢复条件")
        ids = tuple(animal_ids)
        if kind is ContingencyKind.ISOLATION and not ids:
            raise ValueError("单只隔离必须指明动物")
        ids = self._animals(ids, allow_empty=True)
        payload: dict[str, Any] = {
            "kind": kind.value,
            "segments": [s.value for s in segments],
            "reason": reason,
            "recovery_condition": recovery_condition,
            "status": "生效中",
        }
        if ids:
            payload["scope_animal_ids"] = list(ids)
        allowed = list(allowed_foods)
        if kind is ContingencyKind.FEED_SUBSTITUTION:
            if not allowed:
                raise ValueError("饲料替代必须列出可接受替代饲料")
            payload["allowed_foods"] = allowed
        return self._append(
            RecordType.CONTINGENCY, occurred_at, actor_id, ids, segments[0], Visibility.SHARED, payload, recorded_at
        )

    def close_contingency(
        self,
        contingency_seq: int,
        resolution: str,
        actor_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._actor(actor_id)
        event = self._record(contingency_seq)
        if event.type is not RecordType.CONTINGENCY:
            raise ValueError("目标记录不是偶发事件")
        if any(c.payload.get("closes") == contingency_seq for c in self._by_type(RecordType.CONTINGENCY_CLOSE)):
            raise ValueError("偶发事件已恢复，不能重复关闭")
        if not resolution:
            raise ValueError("关闭事件必须记录恢复条件的落实情况")
        # 状态变更同样以追加方式记录，不回写原事件（原事件永久保留"生效中"历史状态）。
        return self._append(
            RecordType.CONTINGENCY_CLOSE,
            occurred_at,
            actor_id,
            event.animal_ids,
            event.stage,
            Visibility.SHARED,
            {"closes": contingency_seq, "resolution": resolution},
            recorded_at,
        )

    # -- 报告受理：幂等 + 矛盾复核 -------------------------------------------

    def submit_report(
        self,
        title: str,
        content_hash: str,
        actor_id: str,
        occurred_at: datetime,
        animal_id: str | None = None,
        findings: Mapping[str, Any] | None = None,
        recorded_at: datetime | None = None,
    ) -> SubmitResult:
        self._actor(actor_id)
        if not title or not content_hash:
            raise ValueError("报告缺少标题或内容哈希")
        findings = dict(findings or {})
        animal_ids = self._animals([animal_id]) if animal_id else ()

        # 同一份报告（内容哈希相同）永远返回首次受理结果（含后续裁决）。
        if content_hash in self._report_index:
            original = self._record(self._report_index[content_hash])
            return SubmitResult(
                record=original,
                status=self.report_status(original.seq),
                reviewer_id=self._review_queue.get(original.seq),
                duplicate=True,
            )

        contradiction = self._find_contradiction(animal_ids[0] if animal_ids else None, findings)
        status = ReportStatus.PENDING_REVIEW if contradiction else ReportStatus.ACCEPTED
        payload: dict[str, Any] = {
            "title": title,
            "content_hash": content_hash,
            "findings": findings,
            "status": status.value,
        }
        if contradiction:
            payload["contradicts_conclusion"] = contradiction.seq
        record = self._append(
            RecordType.REPORT,
            occurred_at,
            actor_id,
            animal_ids,
            None,
            Visibility.SHARED,
            payload,
            recorded_at,
        )
        self._report_index[content_hash] = record.seq

        reviewer_id = None
        if contradiction:
            reviewer_id = self._reviewer_for(animal_ids[0] if animal_ids else None)
            self._review_queue[record.seq] = reviewer_id
        return SubmitResult(record=record, status=status, reviewer_id=reviewer_id)

    def resolve_review(
        self,
        report_seq: int,
        approve: bool,
        note: str,
        vet_id: str,
        occurred_at: datetime,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        self._require_role(vet_id, Role.VET)
        report = self._record(report_seq)
        if report.type is not RecordType.REPORT:
            raise ValueError("目标记录不是报告")
        if report.payload["status"] != ReportStatus.PENDING_REVIEW.value:
            raise ValueError("该报告不在复核队列")
        if self._review_queue.get(report_seq) != vet_id:
            raise PermissionError("该报告已指派给其他指定兽医复核")
        if not note:
            raise ValueError("复核裁决必须说明依据")

        new_status = ReportStatus.ACCEPTED if approve else ReportStatus.REJECTED
        resolution = self._append(
            RecordType.REVIEW,
            occurred_at,
            vet_id,
            report.animal_ids,
            report.stage,
            Visibility.SHARED,
            {"report_seq": report_seq, "verdict": new_status.value, "note": note},
            recorded_at,
        )
        # 裁决同样不回写报告：以追加的 REVIEW 记录表达最终状态。
        self._review_queue.pop(report_seq, None)
        if approve and report.animal_ids and report.payload.get("findings"):
            contradicted = report.payload.get("contradicts_conclusion")
            if contradicted:
                self.correct_conclusion(
                    contradicted,
                    summary=f"采纳报告《{report.payload['title']}》后的兽医结论",
                    findings=report.payload["findings"],
                    reason=f"复核采纳：{note}",
                    vet_id=vet_id,
                    occurred_at=occurred_at,
                    recorded_at=recorded_at,
                )
        return resolution

    def report_status(self, report_seq: int) -> ReportStatus:
        report = self._record(report_seq)
        if report.type is not RecordType.REPORT:
            raise ValueError("目标记录不是报告")
        latest = report.payload["status"]
        for review in self._by_type(RecordType.REVIEW):
            if review.payload["report_seq"] == report_seq:
                latest = review.payload["verdict"]
        return ReportStatus(latest)

    # -- 交接 ---------------------------------------------------------------

    def handoff(
        self,
        from_actor_id: str,
        to_actor_id: str,
        stage: Stage,
        animal_ids: Iterable[str],
        plan_seq: int,
        batch_ids: Iterable[str] = (),
        note: str = "",
        occurred_at: datetime | None = None,
        recorded_at: datetime | None = None,
    ) -> CareRecord:
        """记录换班/换方交接；交接依据（计划、批次、记录区间）被固定下来。"""
        if occurred_at is None:
            raise ValueError("交接必须带发生时间")
        self._actor(from_actor_id)
        self._actor(to_actor_id)
        ids = self._animals(animal_ids)
        plan = self._record(plan_seq)
        if plan.type is not RecordType.PLAN:
            raise ValueError("交接依据必须指向照护计划记录")
        known_batches = {r.payload["batch_id"] for r in self._by_type(RecordType.BATCH_RECEIPT)}
        for batch_id in batch_ids:
            if batch_id not in known_batches:
                raise ValueError(f"交接引用了未知饲料批次：{batch_id}")
        payload = {
            "from_actor_id": from_actor_id,
            "to_actor_id": to_actor_id,
            "plan_seq": plan_seq,
            "batch_ids": list(batch_ids),
            "basis_seq_from": 1,
            "basis_seq_to": len(self.records),
            "note": note,
        }
        return self._append(
            RecordType.HANDOFF, occurred_at, from_actor_id, ids, stage, Visibility.SHARED, payload, recorded_at
        )

    # -- 查询 / 视图 / 回放 --------------------------------------------------

    def view_for(self, side: Side) -> tuple[CareRecord, ...]:
        """按参与方返回可见记录；JOINT 管理机构在争议时可见全部。"""
        if side is Side.JOINT:
            return tuple(self.records)
        own = Visibility.CN_ONLY if side is Side.CN else Visibility.US_ONLY
        return tuple(
            r for r in self.records if r.visibility in (Visibility.SHARED, own)
        )

    def verify_chain(self) -> bool:
        prev_hash = ""
        for i, record in enumerate(self.records, start=1):
            if record.seq != i or record.prev_hash != prev_hash:
                return False
            material = {
                "seq": record.seq,
                "type": record.type.value,
                "occurred_at": record.occurred_at,
                "recorded_at": record.recorded_at,
                "actor_id": record.actor_id,
                "animal_ids": list(record.animal_ids),
                "stage": record.stage.value if record.stage else None,
                "visibility": record.visibility.value,
                "payload": dict(record.payload),
                "prev_hash": record.prev_hash,
            }
            if _digest(material) != record.hash:
                return False
            prev_hash = record.hash
        return True

    def snapshot(self, as_of: datetime, viewer: Side = Side.JOINT) -> Snapshot:
        """还原某记录时点的照护计划、物资去向、事件状态与交接依据。"""
        _aware(as_of, "回放时点")
        visible = [r for r in self.view_for(viewer) if r.recorded_at <= as_of]
        plans: dict[Stage, CareRecord] = {}
        balances: dict[str, float] = {}
        contingencies: dict[int, CareRecord] = {}
        conclusions: dict[str, CareRecord] = {}
        superseded: set[int] = set()
        open_reviews: list[CareRecord] = []

        for r in visible:
            if r.type is RecordType.PLAN and r.stage is not None:
                plans[r.stage] = r
            elif r.type is RecordType.BATCH_RECEIPT:
                balances[r.payload["batch_id"]] = balances.get(r.payload["batch_id"], 0.0) + float(
                    r.payload["quantity"]
                )
            elif r.type is RecordType.FEEDING and "batch_id" in r.payload:
                balances[r.payload["batch_id"]] -= float(r.payload["amount"])
            elif r.type is RecordType.CONTINGENCY:
                contingencies[r.seq] = r
            elif r.type is RecordType.CONTINGENCY_CLOSE:
                contingencies.pop(r.payload["closes"], None)
            elif r.type is RecordType.VET_CONCLUSION:
                for animal_id in r.animal_ids:
                    conclusions[animal_id] = r
            elif r.type is RecordType.VET_CORRECTION:
                superseded.add(r.payload["supersedes"])
                for animal_id in r.animal_ids:
                    conclusions[animal_id] = r
            elif r.type is RecordType.REPORT and r.payload["status"] == ReportStatus.PENDING_REVIEW.value:
                if not any(
                    rv.payload["report_seq"] == r.seq
                    for rv in visible
                    if rv.type is RecordType.REVIEW
                ):
                    open_reviews.append(r)

        active_conclusions = {a: r for a, r in conclusions.items() if r.seq not in superseded}
        chain_ok = all(
            _digest({
                "seq": r.seq,
                "type": r.type.value,
                "occurred_at": r.occurred_at,
                "recorded_at": r.recorded_at,
                "actor_id": r.actor_id,
                "animal_ids": list(r.animal_ids),
                "stage": r.stage.value if r.stage else None,
                "visibility": r.visibility.value,
                "payload": dict(r.payload),
                "prev_hash": r.prev_hash,
            })
            == r.hash
            for r in visible
        )
        return Snapshot(
            as_of=as_of,
            records=tuple(visible),
            plans=MappingProxyType(plans),
            active_contingencies=tuple(contingencies.values()),
            batch_balances=MappingProxyType(balances),
            active_conclusions=MappingProxyType(active_conclusions),
            open_reviews=tuple(open_reviews),
            chain_ok=chain_ok,
        )

    def contingency_active(
        self, kind: ContingencyKind, segment: Stage, animal_id: str | None, at: datetime
    ) -> CareRecord | None:
        """查询某动物在某环节、某时点是否命中生效中的偶发事件。"""
        for event in self._by_type(RecordType.CONTINGENCY):
            if event.payload["kind"] != kind.value or event.payload["status"] != "生效中":
                continue
            if segment.value not in event.payload["segments"]:
                continue
            scope = event.payload.get("scope_animal_ids")
            if scope is not None and animal_id not in scope:
                continue
            if any(
                c.payload.get("closes") == event.seq
                for c in self._by_type(RecordType.CONTINGENCY_CLOSE)
            ):
                continue
            if event.occurred_at <= at:
                return event
        return None

    # -- 内部辅助 ------------------------------------------------------------

    def _by_type(self, type_: RecordType) -> list[CareRecord]:
        return [r for r in self.records if r.type is type_]

    def _record(self, seq: int) -> CareRecord:
        if not isinstance(seq, int) or seq < 1 or seq > len(self.records):
            raise ValueError(f"记录序号不存在：{seq}")
        return self.records[seq - 1]

    def _is_superseded(self, conclusion_seq: int) -> bool:
        return any(
            r.type is RecordType.VET_CORRECTION and r.payload["supersedes"] == conclusion_seq
            for r in self.records
        )

    def _reviewer_for(self, animal_id: str | None) -> str:
        if animal_id and animal_id in self._reviewers:
            return self._reviewers[animal_id]
        if "*" in self._reviewers:
            return self._reviewers["*"]
        raise ValueError("尚未指定复核兽医")

    def _find_contradiction(self, animal_id: str | None, findings: Mapping[str, Any]) -> CareRecord | None:
        if not animal_id or not findings:
            return None
        active = self._active_conclusion(animal_id)
        if active is None:
            return None
        for key, value in findings.items():
            if key in active.payload["findings"] and active.payload["findings"][key] != value:
                return active
        return None

    def _active_conclusion(self, animal_id: str) -> CareRecord | None:
        latest: CareRecord | None = None
        for r in self.records:
            if animal_id not in r.animal_ids:
                continue
            if r.type is RecordType.VET_CONCLUSION:
                latest = r
            elif r.type is RecordType.VET_CORRECTION:
                latest = r
        if latest is not None and latest.type is RecordType.VET_CORRECTION:
            return latest  # 纠正记录同样携带 findings
        return latest

    def _batch(self, batch_id: str) -> CareRecord:
        for r in self._by_type(RecordType.BATCH_RECEIPT):
            if r.payload["batch_id"] == batch_id:
                return r
        raise ValueError(f"未知饲料批次：{batch_id}")

    def _batch_balance(self, batch_id: str) -> float:
        receipt = self._batch(batch_id)
        used = sum(
            float(r.payload["amount"])
            for r in self._by_type(RecordType.FEEDING)
            if r.payload.get("batch_id") == batch_id
        )
        return float(receipt.payload["quantity"]) - used

    def _validate_batch_use(
        self,
        batch_id: str,
        food: str,
        amount: float,
        animal_id: str,
        stage: Stage,
        at: datetime,
    ) -> None:
        receipt = self._batch(batch_id)
        if receipt.payload["food"] != food:
            raise ValueError("喂养饲料与批次登记品类不一致")
        if self._batch_balance(batch_id) < amount:
            raise ValueError(f"批次{batch_id}结余不足")

    def _validate_diet(self, animal_id: str, stage: Stage, food: str, at: datetime) -> None:
        substitution = self.contingency_active(
            ContingencyKind.FEED_SUBSTITUTION, stage, animal_id, at
        )
        if substitution is not None:
            if food not in substitution.payload["allowed_foods"]:
                raise ValueError("替代饲料不在事件批准清单内")
            return
        plan = self._effective_plan(stage, at)
        if plan is not None and plan.payload["foods"] and food not in plan.payload["foods"]:
            raise ValueError("当前环节照护计划不含该饲料，如需替代须先开启饲料替代事件")

    def _effective_plan(self, stage: Stage, at: datetime) -> CareRecord | None:
        candidates = [
            r
            for r in self._by_type(RecordType.PLAN)
            if r.stage is stage and r.occurred_at <= at
        ]
        return candidates[-1] if candidates else None
