"""跨境旅居动物照护服务的全链路规则测试。"""

from __future__ import annotations

import json
import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.panda_care import (
    Actor,
    ActiveEffect,
    BatchRef,
    CareService,
    CareServiceError,
    EffectActiveError,
    EffectKind,
    Feeding,
    IdempotencyStore,
    Ledger,
    LedgerError,
    Medication,
    Observation,
    ObservationKind,
    PermissionDenied,
    Phase,
    PhaseGateError,
    Role,
    Scope,
    Side,
    SpeciesFact,
    TransferOfCare,
    UploadStatus,
    VetConclusion,
)

PP = "an_pp"  # 平平
FS = "an_fs"  # 福双


class FakeClock:
    """可控时钟：每次 now() 前进一秒，保证 recorded_at 严格有序。"""

    def __init__(self, start: datetime | None = None) -> None:
        self.t = start or datetime(2026, 9, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: int = 1) -> datetime:
        self.t += timedelta(seconds=seconds)
        return self.t


class CareServiceTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.svc = CareService(clock=self.clock)
        # 责任人（示例标识，不含真实身份信息）
        self.authority = Actor.create("actor_auth", "双方项目管理机构", Side.BOTH, {Role.AUTHORITY})
        self.operator = Actor.create("actor_op", "亚特兰大动物园运营", Side.US, {Role.OPERATOR})
        self.cn_keeper = Actor.create("actor_cnk", "中方饲养员", Side.CN, {Role.CN_KEEPER})
        self.us_keeper = Actor.create("actor_usk", "美方饲养员", Side.US, {Role.US_KEEPER})
        self.cn_vet = Actor.create("actor_cnvet", "中方兽医", Side.CN, {Role.VET})
        self.us_vet = Actor.create("actor_usvet", "美方兽医", Side.US, {Role.VET})
        self.d_vet = Actor.create("actor_dvet", "指定复核兽医", Side.BOTH, {Role.DESIGNATED_VET})
        self.carrier = Actor.create("actor_carrier", "承运方", Side.BOTH, {Role.TRANSPORT})
        for actor in (
            self.authority, self.operator, self.cn_keeper, self.us_keeper,
            self.cn_vet, self.us_vet, self.d_vet, self.carrier,
        ):
            self.svc.register_actor(actor)
        self.svc.register_animal(SpeciesFact(PP, "平平"))
        self.svc.register_animal(SpeciesFact(FS, "福双"))

    def plan_items(self) -> list[dict]:
        return [
            {"plan_item_id": "pp_bam", "animal_id": PP, "item": "竹笋",
             "batch_id": "batch_01", "amount_kg": 8.0},
            {"plan_item_id": "fs_bam", "animal_id": FS, "item": "竹笋",
             "batch_id": "batch_01", "amount_kg": 7.0},
        ]

    def create_plan(self) -> None:
        self.svc.create_care_plan(self.operator.actor_id, "plan_2026", self.plan_items())

    def reach_phase(self, phase: Phase) -> dict[str, object]:
        """按合法门禁依次推进到目标阶段，返回关键事件，供各测试用例复用。"""
        refs: dict[str, object] = {}
        self.create_plan()
        refs["assessment"] = self.svc.record_health_assessment(
            self.cn_vet.actor_id, PP, "出发前体检适飞", fit=True)
        refs["conclusion"] = self.svc.record_vet_conclusion(
            self.cn_vet.actor_id, VetConclusion(PP, "fitness_to_load", "适飞"))
        self.clock.advance()
        self.svc.advance_phase(
            self.authority.actor_id, Phase.CRATING, [refs["conclusion"].event_id])
        refs["seal"] = self.svc.seal_crate(
            self.cn_keeper.actor_id, [PP, FS], "crate-1", ["seal-a", "seal-b"], "manifest-1")
        self.clock.advance()
        self.svc.advance_phase(
            self.authority.actor_id, Phase.FLIGHT, [refs["seal"].event_id])
        refs["departure"] = self.svc.record_flight_event(
            self.carrier.actor_id, "DEPARTURE", "起飞", flight_no="CA8888")
        self.clock.advance()
        refs["arrival"] = self.svc.record_flight_event(
            self.carrier.actor_id, "ARRIVAL", "落地")
        if phase is Phase.FLIGHT:
            return refs
        self.clock.advance()
        self.svc.advance_phase(
            self.authority.actor_id, Phase.ADAPTATION, [refs["arrival"].event_id])
        return refs


class PhaseAndArchiveTest(CareServiceTestBase):
    def test_phase_starts_at_pre_departure_and_advances_one_step(self) -> None:
        self.assertEqual(self.svc.current_phase(), Phase.PRE_DEPARTURE)
        self.create_plan()
        conclusion = self.svc.record_vet_conclusion(
            self.cn_vet.actor_id, VetConclusion(PP, "fitness_to_load", "适飞"))
        self.clock.advance()
        self.svc.advance_phase(self.authority.actor_id, Phase.CRATING, [conclusion.event_id])
        self.assertEqual(self.svc.current_phase(), Phase.CRATING)
        # 不能跨格推进
        with self.assertRaises(PhaseGateError):
            self.svc.advance_phase(self.authority.actor_id, Phase.ADAPTATION, [conclusion.event_id])

    def test_gate_requires_authority_and_real_basis(self) -> None:
        self.create_plan()
        with self.assertRaises(PermissionDenied):
            self.svc.advance_phase(self.operator.actor_id, Phase.CRATING, ["evt_none"])
        with self.assertRaises(PhaseGateError):
            self.svc.advance_phase(self.authority.actor_id, Phase.CRATING, [])
        with self.assertRaises(PhaseGateError):
            self.svc.advance_phase(self.authority.actor_id, Phase.CRATING, ["evt_missing"])

    def test_quarantine_blocks_flight_for_scoped_animal_only(self) -> None:
        refs = self.reach_phase(Phase.FLIGHT)
        self.clock.advance()
        quarantine = self.svc.start_effect(
            self.authority.actor_id, EffectKind.QUARANTINE, Scope.animals(PP),
            recovery_condition="检疫期满且检测阴性")
        # PP 检疫未解除不能入园
        with self.assertRaises(EffectActiveError):
            self.svc.advance_phase(
                self.authority.actor_id, Phase.ADAPTATION, [refs["arrival"].event_id])
        # 另一只动物的交接不受影响
        self.svc.transfer_care(
            self.operator.actor_id,
            TransferOfCare("承运→园区(福双)", self.carrier.actor_id, self.us_keeper.actor_id,
                           ("途中喂养",), frozenset({self.carrier.actor_id, self.us_keeper.actor_id}),
                           frozenset({FS})))
        self.clock.advance()
        self.svc.resolve_effect(self.authority.actor_id, quarantine.event_id, "检测阴性，检疫期满")
        self.clock.advance()
        self.svc.advance_phase(
            self.authority.actor_id, Phase.ADAPTATION, [refs["arrival"].event_id])
        self.assertEqual(self.svc.current_phase(), Phase.ADAPTATION)

    def test_flight_delay_blocks_leaving_flight_phase(self) -> None:
        refs = self.reach_phase(Phase.FLIGHT)
        self.clock.advance()
        delay = self.svc.start_effect(
            self.carrier.actor_id, EffectKind.FLIGHT_DELAY, Scope(phase_limited=True),
            recovery_condition="重新放行并落地")
        with self.assertRaises(EffectActiveError):
            self.svc.advance_phase(
                self.authority.actor_id, Phase.ADAPTATION, [refs["arrival"].event_id])
        self.clock.advance()
        self.svc.resolve_effect(self.carrier.actor_id, delay.event_id, "航班已落地")
        self.clock.advance()
        self.svc.advance_phase(
            self.authority.actor_id, Phase.ADAPTATION, [refs["arrival"].event_id])

    def test_isolation_does_not_block_phase_but_marks_feeding_and_blocks_handover(self) -> None:
        refs = self.reach_phase(Phase.ADAPTATION)
        self.clock.advance()
        isolation = self.svc.start_effect(
            self.d_vet.actor_id, EffectKind.ISOLATION, Scope.animals(FS),
            recovery_condition="连续48小时体温正常")
        self.clock.advance()
        feeding = self.svc.record_feeding(
            self.us_keeper.actor_id, Feeding(FS, "fs_bam", "竹笋", "batch_01", 7.0))
        self.assertTrue(feeding.body["under_isolation"])
        # PP 的喂养不带隔离标记
        feeding_pp = self.svc.record_feeding(
            self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_01", 8.0))
        self.assertFalse(feeding_pp.body["under_isolation"])
        # 隔离动物不能交接
        with self.assertRaises(EffectActiveError):
            self.svc.transfer_care(
                self.operator.actor_id,
                TransferOfCare("夜间班", self.us_keeper.actor_id, self.cn_keeper.actor_id,
                               ("夜间观察",), frozenset({self.us_keeper.actor_id}),
                               frozenset({FS})))
        self.clock.advance()
        self.svc.resolve_effect(self.d_vet.actor_id, isolation.event_id, "48小时体温正常")


class AppendOnlyAndIntegrityTest(CareServiceTestBase):
    def test_events_carry_time_actor_and_hash_chain(self) -> None:
        self.create_plan()
        event = self.svc.ledger.events[-1]
        self.assertEqual(event.seq, 0)
        self.assertIsNotNone(event.hash)
        self.assertEqual(event.prev_hash, "0" * 64)
        self.assertEqual(event.actor_id, self.operator.actor_id)
        self.assertIsNotNone(event.occurred_at.tzinfo)

    def test_chain_extends_and_verifies(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.svc.verify_integrity()  # 不抛异常即通过
        events = self.svc.ledger.events
        for previous, current in zip(events, events[1:]):
            self.assertEqual(current.prev_hash, previous.hash)
            self.assertEqual(current.seq, previous.seq + 1)

    def test_sealed_event_cannot_be_appended_again(self) -> None:
        self.create_plan()
        sealed = self.svc.ledger.events[-1]
        with self.assertRaises(LedgerError):
            self.svc.ledger.append(sealed)

    def test_tampered_snapshot_fails_verification(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        path = Path("/tmp/panda-care-ledger-test.json")
        self.svc.ledger.save_json(str(path))
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw[1]["body"]["summary"] = "被篡改"
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(LedgerError):
            Ledger.from_json(str(path))

    def test_snapshot_roundtrip_preserves_chain(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        path = Path("/tmp/panda-care-ledger-ok.json")
        self.svc.ledger.save_json(str(path))
        restored = Ledger.from_json(str(path))
        self.assertEqual(len(restored.events), len(self.svc.ledger.events))
        self.assertEqual(restored.head_hash, self.svc.ledger.head_hash)

    def test_snapshot_matches_event_contract(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        schema = json.loads(
            Path("contracts/care-events.schema.json").read_text(encoding="utf-8"))
        event_types = set(schema["properties"]["event_type"]["enum"])
        phases = set(schema["properties"]["phase"]["enum"])
        hash_re = re.compile(schema["properties"]["hash"]["pattern"])
        id_re = re.compile(schema["properties"]["event_id"]["pattern"])
        required = set(schema["required"])
        for item in self.svc.ledger.snapshot():
            self.assertEqual(set(item), required)
            self.assertIn(item["event_type"], event_types)
            self.assertIn(item["phase"], phases)
            self.assertTrue(hash_re.fullmatch(item["hash"]))
            self.assertTrue(hash_re.fullmatch(item["prev_hash"]))
            self.assertTrue(id_re.fullmatch(item["event_id"]))
            self.assertEqual(
                set(item["scope"]),
                {"animal_ids", "plan_item_ids", "batch_ids", "phase_limited"})


class DailyCareTest(CareServiceTestBase):
    def test_feeding_must_match_current_plan(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        # 未登记替代时换批次被拒
        with self.assertRaises(CareServiceError):
            self.svc.record_feeding(
                self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_99", 8.0))
        with self.assertRaises(CareServiceError):
            self.svc.record_feeding(
                self.us_keeper.actor_id, Feeding(PP, "pp_bam", "苹果", "batch_01", 8.0))
        # 条目属于另一只动物
        with self.assertRaises(CareServiceError):
            self.svc.record_feeding(
                self.us_keeper.actor_id, Feeding(FS, "pp_bam", "竹笋", "batch_01", 8.0))

    def test_feed_substitution_scoped_to_plan_item(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.clock.advance()
        sub = self.svc.start_effect(
            self.operator.actor_id, EffectKind.FEED_SUBSTITUTION,
            Scope(plan_item_ids=frozenset({"pp_bam"})),
            recovery_condition="原批次复检合格",
            substitute={"batch_id": "batch_02", "item": "竹笋"})
        # PP 的原批次被停用
        with self.assertRaises(EffectActiveError):
            self.svc.record_feeding(
                self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_01", 8.0))
        # 替代批次之外的批次也不允许
        with self.assertRaises(EffectActiveError):
            self.svc.record_feeding(
                self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_03", 8.0))
        # 替代批次可投喂，并在记录中留痕
        feeding = self.svc.record_feeding(
            self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_02", 8.0))
        self.assertEqual(feeding.body["substitute_batch"]["batch_id"], "batch_02")
        # FS 的同批次条目不受影响
        self.svc.record_feeding(
            self.us_keeper.actor_id, Feeding(FS, "fs_bam", "竹笋", "batch_01", 7.0))
        # 恢复条件不满足不能解除
        with self.assertRaises(CareServiceError):
            self.svc.resolve_effect(self.operator.actor_id, sub.event_id, "  ")
        self.clock.advance()
        self.svc.resolve_effect(self.operator.actor_id, sub.event_id, "batch_01 复检合格")
        # 解除后恢复原批次
        self.svc.record_feeding(
            self.us_keeper.actor_id, Feeding(PP, "pp_bam", "竹笋", "batch_01", 8.0))

    def test_effect_requires_recovery_condition_and_cannot_double_resolve(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        with self.assertRaises(CareServiceError):
            self.svc.start_effect(
                self.operator.actor_id, EffectKind.FEED_SUBSTITUTION,
                Scope(plan_item_ids=frozenset({"pp_bam"})),
                recovery_condition=" ", substitute={"batch_id": "batch_02"})
        with self.assertRaises(CareServiceError):
            self.svc.start_effect(
                self.operator.actor_id, EffectKind.FEED_SUBSTITUTION,
                Scope(plan_item_ids=frozenset({"pp_bam"})),
                recovery_condition="有替代")
        effect = self.svc.start_effect(
            self.operator.actor_id, EffectKind.ISOLATION, Scope.animals(PP),
            recovery_condition="观察结束")
        self.clock.advance()
        self.svc.resolve_effect(self.d_vet.actor_id, effect.event_id, "观察期满")
        with self.assertRaises(CareServiceError):
            self.svc.resolve_effect(self.d_vet.actor_id, effect.event_id, "再次解除")

    def test_weight_and_abnormal_observation_rules(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.svc.record_observation(
            self.us_keeper.actor_id, Observation(PP, ObservationKind.WEIGHT, "102.5"))
        with self.assertRaises(CareServiceError):
            self.svc.record_observation(
                self.us_keeper.actor_id, Observation(PP, ObservationKind.WEIGHT, "很重"))
        with self.assertRaises(CareServiceError):
            self.svc.record_observation(
                self.us_keeper.actor_id,
                Observation(PP, ObservationKind.BEHAVIOR, "异常踱步", abnormal=True))
        self.svc.record_observation(
            self.us_keeper.actor_id,
            Observation(PP, ObservationKind.BEHAVIOR, "异常踱步", abnormal=True, note="持续30分钟"))

    def test_keeper_medication_requires_active_vet_order(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.clock.advance()
        order = self.svc.record_vet_conclusion(
            self.us_vet.actor_id, VetConclusion(PP, "parasite_control", "驱虫药一次"))
        with self.assertRaises(PermissionDenied):
            self.svc.record_medication(
                self.cn_keeper.actor_id, Medication(PP, "驱虫药", "1袋", "例行"))
        # 引用现行医嘱可以执行
        self.svc.record_medication(
            self.cn_keeper.actor_id,
            Medication(PP, "驱虫药", "1袋", "例行", per_vet_conclusion_id=order.event_id))
        # 兽医本人可直接开方用药
        self.svc.record_medication(
            self.us_vet.actor_id, Medication(FS, "益生菌", "5g", "调理"))
        # 动物不匹配被拒
        with self.assertRaises(CareServiceError):
            self.svc.record_medication(
                self.cn_keeper.actor_id,
                Medication(FS, "驱虫药", "1袋", "例行", per_vet_conclusion_id=order.event_id))


class BatchAndTransferTest(CareServiceTestBase):
    def test_batch_movements_and_cross_border_transfer(self) -> None:
        refs = self.reach_phase(Phase.FLIGHT)
        self.svc.dispatch_batch(
            self.operator.actor_id, BatchRef("batch_01", "竹笋", 300.0, "kg"),
            "成都基地冷库", "机坪冷链仓", self.carrier.actor_id, "随包机运输")
        # 跨境交接必须中美双方共同确认
        with self.assertRaises(PermissionDenied):
            self.svc.transfer_care(
                self.operator.actor_id,
                TransferOfCare("出境交接", self.cn_keeper.actor_id, self.carrier.actor_id,
                               ("途中照护",), frozenset({self.cn_keeper.actor_id}),
                               frozenset({PP, FS}), cross_border=True,
                               manifest_hash=refs["seal"].body["manifest_hash"]))
        transfer = self.svc.transfer_care(
            self.operator.actor_id,
            TransferOfCare("出境交接", self.cn_keeper.actor_id, self.carrier.actor_id,
                           ("途中照护", "冷链温控"),
                           frozenset({self.cn_keeper.actor_id, self.us_keeper.actor_id}),
                           frozenset({PP, FS}), cross_border=True,
                           manifest_hash=refs["seal"].body["manifest_hash"]))
        self.assertEqual(transfer.visible_to, Side.BOTH)
        view = self.svc.view_at(self.clock(), Side.BOTH)
        self.assertEqual(view.batch_movements[-1]["batch"]["batch_id"], "batch_01")
        self.assertEqual(view.transfers[-1]["transfer"]["checkpoint"], "出境交接")


class VetConclusionTest(CareServiceTestBase):
    def test_conclusion_supersede_keeps_old_record(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        first = self.svc.record_vet_conclusion(
            self.cn_vet.actor_id, VetConclusion(PP, "diagnosis", "轻度应激"))
        self.clock.advance()
        second = self.svc.record_vet_conclusion(
            self.d_vet.actor_id,
            VetConclusion(PP, "diagnosis", "恢复良好", supersedes=first.event_id))
        # 旧事件仍在链上
        self.assertIsNotNone(self.svc.ledger.get(first.event_id))
        active = {e.event_id for e in self.svc.active_vet_conclusions(PP)}
        self.assertIn(second.event_id, active)
        self.assertNotIn(first.event_id, active)
        # 不能重复修正同一旧结论
        with self.assertRaises(CareServiceError):
            self.svc.record_vet_conclusion(
                self.d_vet.actor_id,
                VetConclusion(PP, "diagnosis", "第三次结论", supersedes=first.event_id))
        # 修正必须同动物同议题
        with self.assertRaises(CareServiceError):
            self.svc.record_vet_conclusion(
                self.d_vet.actor_id,
                VetConclusion(FS, "diagnosis", "别的动物", supersedes=first.event_id))


class ReportAndReviewTest(CareServiceTestBase):
    REPORT_BYTES = b"health-report-content-v1"
    CLAIMS = [{"animal_id": PP, "topic": "fitness_to_load", "finding": "不适飞"}]

    def test_clean_report_is_accepted_and_idempotent(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        first = self.svc.upload_report(
            self.cn_vet.actor_id, "健康报告", b"clean-content",
            [{"animal_id": PP, "topic": "fitness_to_load", "finding": "适飞"}])
        self.assertEqual(first.status, UploadStatus.ACCEPTED)
        self.assertFalse(first.duplicate)
        # 换提交者重复上传同一份内容，仍返回原结果，不产生新事件
        count_before = len(self.svc.ledger.events)
        again = self.svc.upload_report(
            self.us_keeper.actor_id, "重复健康报告", b"clean-content",
            [{"animal_id": PP, "topic": "fitness_to_load", "finding": "适飞"}])
        self.assertTrue(again.duplicate)
        self.assertEqual(again.event_id, first.event_id)
        self.assertEqual(len(self.svc.ledger.events), count_before)

    def test_conflicting_report_opens_review_and_suspends_conclusion(self) -> None:
        refs = self.reach_phase(Phase.ADAPTATION)
        order = refs["conclusion"]
        first = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.assertEqual(first.status, UploadStatus.REVIEW_PENDING)
        self.assertIsNotNone(first.review_event_id)
        # 复核中重复上传 → REVIEWING，同一报告事件
        again = self.svc.upload_report(
            self.carrier.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.assertEqual(again.status, UploadStatus.REVIEWING)
        self.assertTrue(again.duplicate)
        self.assertEqual(again.event_id, first.event_id)
        # 冲突结论在复核期间挂起，不能作为用药依据
        with self.assertRaises(CareServiceError):
            self.svc.record_medication(
                self.cn_keeper.actor_id,
                Medication(PP, "药", "1", "依据挂起结论",
                           per_vet_conclusion_id=order.event_id))
        view = self.svc.view_at(self.clock(), Side.BOTH)
        self.assertEqual(len(view.open_reviews), 1)

    def test_only_assigned_designated_vet_can_resolve(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        result = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        with self.assertRaises(PermissionDenied):
            self.svc.resolve_review(self.us_vet.actor_id, result.review_event_id,
                                    "REJECT_REPORT")

    def test_accept_report_supersedes_and_duplicate_returns_terminal(self) -> None:
        refs = self.reach_phase(Phase.ADAPTATION)
        old = refs["conclusion"]
        result = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.clock.advance()
        resolved = self.svc.resolve_review(
            self.d_vet.actor_id, result.review_event_id,
            "ACCEPT_REPORT", new_finding="暂缓运输，治疗后复查")
        self.assertEqual(resolved.body["superseded_conclusions"], [old.event_id])
        # 旧结论失效，新结论生效
        active = {e.event_id for e in self.svc.active_vet_conclusions(PP)}
        self.assertNotIn(old.event_id, active)
        self.assertEqual(len(active & {old.event_id}), 0)
        self.assertEqual(len(self.svc.active_vet_conclusions(PP)), 1)
        # 复核已闭环
        self.assertEqual(self.svc.view_at(self.clock(), Side.BOTH).open_reviews, ())
        # 重复上传得到终态
        again = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.assertTrue(again.duplicate)
        self.assertEqual(again.status, UploadStatus.REVIEW_ACCEPTED)
        # 复核结论只能有一次
        with self.assertRaises(CareServiceError):
            self.svc.resolve_review(
                self.d_vet.actor_id, result.review_event_id, "REJECT_REPORT")

    def test_reject_report_keeps_original_conclusion(self) -> None:
        refs = self.reach_phase(Phase.ADAPTATION)
        old = refs["conclusion"]
        result = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.clock.advance()
        self.svc.resolve_review(
            self.d_vet.actor_id, result.review_event_id, "REJECT_REPORT")
        active = {e.event_id for e in self.svc.active_vet_conclusions(PP)}
        self.assertIn(old.event_id, active)
        again = self.svc.upload_report(
            self.cn_vet.actor_id, "检疫报告", self.REPORT_BYTES, self.CLAIMS)
        self.assertEqual(again.status, UploadStatus.REVIEW_REJECTED)

    def test_conflict_without_designated_vet_is_rejected_before_appending(self) -> None:
        bare = CareService(clock=self.clock)
        authority = Actor.create("auth2", "机构", Side.BOTH, {Role.AUTHORITY})
        vet = Actor.create("vet2", "兽医", Side.CN, {Role.VET})
        keeper = Actor.create("k2", "饲养员", Side.CN, {Role.CN_KEEPER})
        operator = Actor.create("op2", "运营", Side.US, {Role.OPERATOR})
        for actor in (authority, vet, keeper, operator):
            bare.register_actor(actor)
        bare.register_animal(SpeciesFact(PP, "平平"))
        bare.create_care_plan(
            "op2", "p",
            [{"plan_item_id": "x", "animal_id": PP, "item": "笋",
              "batch_id": "b", "amount_kg": 1.0}])
        bare.record_vet_conclusion("vet2", VetConclusion(PP, "t", "结论A"))
        before = len(bare.ledger.events)
        with self.assertRaises(CareServiceError):
            bare.upload_report(
                "k2", "报告", b"c",
                [{"animal_id": PP, "topic": "t", "finding": "结论B"}])
        # 报告事件未入链
        self.assertEqual(len(bare.ledger.events), before)

    def test_empty_claims_rejected(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        with self.assertRaises(CareServiceError):
            self.svc.upload_report(self.cn_vet.actor_id, "空", b"x", [])


class VisibilityAndReplayTest(CareServiceTestBase):
    def test_side_visibility_filters_events(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.clock.advance()
        cn_only = self.svc.record_health_assessment(
            self.cn_vet.actor_id, FS, "中方内部观察", fit=False, visible_to=Side.CN)
        us_only = self.svc.record_health_assessment(
            self.us_vet.actor_id, PP, "美方内部观察", fit=False, visible_to=Side.US)
        both_view = self.svc.view_at(self.clock(), Side.BOTH)
        cn_view = self.svc.view_at(self.clock(), Side.CN)
        us_view = self.svc.view_at(self.clock(), Side.US)
        ids_both = {e.event_id for e in both_view.events}
        ids_cn = {e.event_id for e in cn_view.events}
        ids_us = {e.event_id for e in us_view.events}
        self.assertIn(cn_only.event_id, ids_both)
        self.assertIn(cn_only.event_id, ids_cn)
        self.assertNotIn(cn_only.event_id, ids_us)
        self.assertIn(us_only.event_id, ids_us)
        self.assertNotIn(us_only.event_id, ids_cn)

    def test_point_in_time_replay_reconstructs_plan_weight_and_phase(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.svc.record_observation(
            self.us_keeper.actor_id, Observation(PP, ObservationKind.WEIGHT, "100.0"))
        cut = self.clock()
        self.clock.advance(10)
        self.svc.record_observation(
            self.us_keeper.actor_id, Observation(PP, ObservationKind.WEIGHT, "101.0"))
        self.svc.revise_care_plan(
            self.operator.actor_id,
            [{**self.plan_items()[0], "amount_kg": 9.0}, self.plan_items()[1]],
            reason="换季调整")
        past = self.svc.view_at(cut, Side.BOTH)
        self.assertEqual(past.phase, Phase.ADAPTATION)
        self.assertEqual(past.plan["revision"], 1)
        self.assertEqual(past.latest_weight_kg[PP], 100.0)
        now_view = self.svc.view_at(self.clock(), Side.BOTH)
        self.assertEqual(now_view.plan["revision"], 2)
        self.assertEqual(now_view.latest_weight_kg[PP], 101.0)

    def test_dispute_export_contains_common_basis_and_per_side_views(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        self.svc.dispatch_batch(
            self.operator.actor_id, BatchRef("batch_01", "竹笋", 300.0, "kg"),
            "成都", "亚特兰大冷库", self.us_keeper.actor_id)
        self.svc.record_health_assessment(
            self.cn_vet.actor_id, FS, "内部", fit=True, visible_to=Side.CN)
        moment = self.clock()
        package = self.svc.dispute_export(moment)
        self.assertEqual(package["domain"], "panda-transit-care")
        self.assertEqual(package["phase"], "ADAPTATION")
        # 证据包头部哈希与双方各自可见视图一致可核验
        self.assertEqual(len(package["events"]), package["event_count"])
        for event in package["events"]:
            self.assertIn(event["visible_to"], ("CN", "US", "BOTH"))
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", event["hash"]))
        cn_plan_items = package["snapshots"]["CN"]["plan_items"]
        us_plan_items = package["snapshots"]["US"]["plan_items"]
        # 共同照护计划双方都能还原
        self.assertEqual(len(cn_plan_items), 2)
        self.assertEqual(len(us_plan_items), 2)
        # 物资去向双方可追溯
        self.assertEqual(package["snapshots"]["BOTH"]["batch_movements"][-1]["to_location"],
                         "亚特兰大冷库")
        # 交接依据在双方快照里一致
        self.assertEqual(
            package["snapshots"]["CN"]["transfers"],
            package["snapshots"]["US"]["transfers"])
        # 头哈希可与重放对齐
        self.assertEqual(package["head_hash"], package["events"][-1]["hash"])

    def test_active_effects_reconstruct_at_replay_time(self) -> None:
        self.reach_phase(Phase.ADAPTATION)
        isolation = self.svc.start_effect(
            self.d_vet.actor_id, EffectKind.ISOLATION, Scope.animals(FS),
            recovery_condition="48小时体温正常")
        view = self.svc.view_at(self.clock(), Side.BOTH)
        found = view.find_effect(EffectKind.ISOLATION, animal_id=FS)
        self.assertIsInstance(found, ActiveEffect)
        self.assertEqual(found.event_id, isolation.event_id)
        self.assertIsNone(view.find_effect(EffectKind.ISOLATION, animal_id=PP))
        self.clock.advance()
        self.svc.resolve_effect(self.d_vet.actor_id, isolation.event_id, "体温正常")
        view2 = self.svc.view_at(self.clock(), Side.BOTH)
        self.assertIsNone(view2.find_effect(EffectKind.ISOLATION, animal_id=FS))


if __name__ == "__main__":
    unittest.main()
