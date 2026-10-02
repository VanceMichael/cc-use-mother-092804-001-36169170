"""跨境照护档案领域规则测试。"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.transit_care import (
    Actor,
    CareLedger,
    ContingencyKind,
    ReportStatus,
    Role,
    Side,
    Stage,
    Visibility,
)

CN_TZ = timezone(timedelta(hours=8))
US_TZ = timezone(timedelta(hours=-4))
T0 = datetime(2026, 10, 1, 8, 0, tzinfo=CN_TZ)


def build_journey() -> tuple[CareLedger, dict[str, Actor]]:
    """搭好一段完整旅程：参与方、动物、计划、批次、复核兽医。"""
    ledger = CareLedger()
    a = {
        "ops": ledger.register_actor("ops-1", "运营周倩", Side.JOINT, {Role.OPERATIONS, Role.AUTHORITY}),
        "cn_keeper": ledger.register_actor("k-cn", "饲养员李航", Side.CN, {Role.KEEPER}),
        "us_keeper": ledger.register_actor("k-us", "饲养员Maria", Side.US, {Role.KEEPER}),
        "cn_vet": ledger.register_actor("v-cn", "兽医陈睿", Side.CN, {Role.VET}),
        "us_vet": ledger.register_actor("v-us", "兽医Dr. Owen", Side.US, {Role.VET}),
        "carrier": ledger.register_actor("air-1", "包机承运方", Side.JOINT, {Role.CARRIER}),
    }
    ledger.register_animal("pingping", "平平")
    ledger.register_animal("fushuang", "福双")
    ledger.designate_reviewer("v-us")
    ledger.publish_plan(
        Stage.TRANSIT,
        foods=["华西箭竹", "冷箭竹", "熊猫饼干"],
        items=["饮水20L", "温控12-18℃"],
        actor_id="ops-1",
        occurred_at=T0,
    )
    ledger.receive_batch("B2026-01", "华西箭竹", 60.0, "air-1", T0 + timedelta(hours=1))
    ledger.receive_batch("B2026-02", "熊猫饼干", 10.0, "air-1", T0 + timedelta(hours=1))
    return ledger, a


class ContinuousFileTest(unittest.TestCase):
    def test_full_journey_forms_tamper_evident_chain(self) -> None:
        ledger, a = build_journey()
        ledger.record_feeding("pingping", Stage.TRANSIT, "华西箭竹", 4.0, "k-cn", T0 + timedelta(hours=3), "B2026-01")
        ledger.record_weight("pingping", Stage.TRANSIT, 102.5, "k-cn", T0 + timedelta(hours=4))
        ledger.record_medication(
            "fushuang", Stage.TRANSIT, "电解质溶液", "50ml", "v-cn", T0 + timedelta(hours=5), reason="航程补水"
        )
        ledger.record_behavior(
            ("pingping",), Stage.TRANSIT, "落地后持续走动、嗅闻", anomaly=True, actor_id="k-us",
            occurred_at=T0 + timedelta(hours=20),
        )
        ledger.add_vet_conclusion(
            "fushuang", Stage.TRANSIT, "航程状态平稳", {"脱水": "无"}, "v-cn", T0 + timedelta(hours=6)
        )
        ledger.handoff(
            "k-cn", "k-us", Stage.QUARANTINE, ("pingping", "fushuang"), plan_seq=1,
            batch_ids=["B2026-01", "B2026-02"], occurred_at=T0 + timedelta(hours=22),
        )
        self.assertTrue(ledger.verify_chain())
        # 计划1 + 批次2 + 喂养/体重/用药/行为/结论/交接6 条
        self.assertEqual(len(ledger.records), 9)

    def test_observations_require_time_responsible_and_animal(self) -> None:
        ledger, a = build_journey()
        with self.assertRaisesRegex(ValueError, "时区"):
            ledger.record_weight(
                "pingping", Stage.TRANSIT, 100.0, "k-cn", datetime(2026, 10, 1, 12, 0)
            )
        with self.assertRaisesRegex(ValueError, "未知责任人"):
            ledger.record_weight("pingping", Stage.TRANSIT, 100.0, "nobody", T0)
        with self.assertRaisesRegex(ValueError, "未知动物"):
            ledger.record_weight("ruyi", Stage.TRANSIT, 100.0, "k-cn", T0)
        with self.assertRaises(PermissionError):
            ledger.record_medication("pingping", Stage.TRANSIT, "X", "1", "air-1", T0)

    def test_tampering_with_confirmed_record_breaks_chain(self) -> None:
        ledger, a = build_journey()
        ledger.add_vet_conclusion("pingping", Stage.TRANSIT, "健康", {"发热": "否"}, "v-cn", T0)
        self.assertTrue(ledger.verify_chain())
        conclusion = ledger.records[-1]
        # 记录为不可变对象；强行外部篡改会被哈希链发现。
        object.__setattr__(conclusion, "payload", {"summary": "被改写"})
        self.assertFalse(ledger.verify_chain())


class VetConclusionTest(unittest.TestCase):
    def test_confirmed_conclusion_can_only_be_corrected_by_append(self) -> None:
        ledger, a = build_journey()
        original = ledger.add_vet_conclusion(
            "pingping", Stage.QUARANTINE, "体温正常，可进入适应期", {"发热": "否"}, "v-cn",
            T0 + timedelta(days=2),
        )
        # 换班饲养员不能改写兽医结论。
        with self.assertRaises(PermissionError):
            ledger.correct_conclusion(
                original.seq, "体温异常", {"发热": "是"}, "换班误判", "k-us", T0 + timedelta(days=3)
            )
        with self.assertRaisesRegex(ValueError, "只能纠正兽医结论"):
            ledger.correct_conclusion(1, "x", {}, "误指向计划", "v-cn", T0)

        correction = ledger.correct_conclusion(
            original.seq, "复查为低热，延长观察", {"发热": "是", "体温": "37.9℃"},
            "晨间复测确认低热", "v-us", T0 + timedelta(days=3),
        )
        # 原结论保留，活跃结论变为纠正记录；同一结论不能被二次纠正。
        self.assertEqual(ledger.records[original.seq - 1].payload["summary"], "体温正常，可进入适应期")
        with self.assertRaisesRegex(ValueError, "已被后续纠正取代"):
            ledger.correct_conclusion(original.seq, "再次修改", {}, "重复纠正", "v-cn", T0 + timedelta(days=4))
        snap = ledger.snapshot(T0 + timedelta(days=5))
        self.assertIs(snap.active_conclusions["pingping"], correction)
        self.assertIn(original, snap.records)


class ContingencyScopeTest(unittest.TestCase):
    def test_isolation_of_one_animal_does_not_affect_the_other(self) -> None:
        ledger, a = build_journey()
        ledger.open_contingency(
            ContingencyKind.ISOLATION, [Stage.QUARANTINE, Stage.ADAPTATION],
            "平平粪便检查复查", "连续三次粪检阴性", "v-us", T0 + timedelta(days=1),
            animal_ids=["pingping"],
        )
        at = T0 + timedelta(days=2)
        isolation_seq = 4  # 计划1 + 批次2 之后的第一条事件
        hit = ledger.contingency_active(ContingencyKind.ISOLATION, Stage.QUARANTINE, "pingping", at)
        self.assertIsNotNone(hit)
        self.assertIsNone(ledger.contingency_active(ContingencyKind.ISOLATION, Stage.QUARANTINE, "fushuang", at))
        # 环节外也不受影响。
        self.assertIsNone(ledger.contingency_active(ContingencyKind.ISOLATION, Stage.RESIDENCE, "pingping", at))

        ledger.close_contingency(isolation_seq, "连续三次粪检阴性，恢复合群", "v-us", T0 + timedelta(days=10))
        self.assertIsNone(
            ledger.contingency_active(ContingencyKind.ISOLATION, Stage.QUARANTINE, "pingping",
                                      T0 + timedelta(days=11))
        )
        with self.assertRaisesRegex(ValueError, "不能重复关闭"):
            ledger.close_contingency(isolation_seq, "再次关闭", "v-us", T0 + timedelta(days=12))

    def test_quarantine_hold_and_flight_delay_are_segment_scoped(self) -> None:
        ledger, a = build_journey()
        ledger.open_contingency(
            ContingencyKind.QUARANTINE_HOLD, [Stage.QUARANTINE],
            "卫生证书编号待复核", "海关放行通知下达", "ops-1", T0 + timedelta(days=1),
        )
        ledger.open_contingency(
            ContingencyKind.FLIGHT_DELAY, [Stage.TRANSIT],
            "阿拉斯加上空航路天气备降", "抵达亚特兰大并完成交接", "air-1", T0 + timedelta(hours=10),
        )
        at = T0 + timedelta(days=1, hours=12)
        self.assertIsNotNone(
            ledger.contingency_active(ContingencyKind.QUARANTINE_HOLD, Stage.QUARANTINE, None, at)
        )
        self.assertIsNone(
            ledger.contingency_active(ContingencyKind.QUARANTINE_HOLD, Stage.ADAPTATION, None, at)
        )
        self.assertIsNotNone(
            ledger.contingency_active(ContingencyKind.FLIGHT_DELAY, Stage.TRANSIT, "pingping", at)
        )

    def test_feed_substitution_only_allows_declared_alternatives_until_recovery(self) -> None:
        ledger, a = build_journey()
        # 计划外饲料在正常情况下被拒绝。
        with self.assertRaisesRegex(ValueError, "照护计划不含该饲料"):
            ledger.record_feeding("pingping", Stage.TRANSIT, "本地竹笋", 2.0, "k-cn", T0 + timedelta(hours=3))

        event = ledger.open_contingency(
            ContingencyKind.FEED_SUBSTITUTION, [Stage.TRANSIT, Stage.ADAPTATION],
            "华西箭竹受潮不可食用", "新批次箭竹完成入场检验",
            "ops-1", T0 + timedelta(hours=3), allowed_foods=["冷箭竹", "本地竹笋"],
        )
        ledger.record_feeding("pingping", Stage.TRANSIT, "本地竹笋", 2.0, "k-cn", T0 + timedelta(hours=4))
        with self.assertRaisesRegex(ValueError, "不在事件批准清单内"):
            ledger.record_feeding("pingping", Stage.TRANSIT, "苹果", 1.0, "k-cn", T0 + timedelta(hours=5))

        ledger.close_contingency(event.seq, "B2026-03批次检验合格入库", "ops-1", T0 + timedelta(days=3))
        ledger.receive_batch("B2026-03", "华西箭竹", 80.0, "air-1", T0 + timedelta(days=3))
        # 恢复后重新受计划约束。
        with self.assertRaisesRegex(ValueError, "照护计划不含该饲料"):
            ledger.record_feeding(
                "pingping", Stage.TRANSIT, "本地竹笋", 2.0, "k-cn", T0 + timedelta(days=4)
            )
        ledger.record_feeding(
            "pingping", Stage.TRANSIT, "华西箭竹", 5.0, "k-us", T0 + timedelta(days=4), "B2026-03"
        )

    def test_isolation_must_target_an_animal_and_other_events_need_segments(self) -> None:
        ledger, a = build_journey()
        with self.assertRaisesRegex(ValueError, "单只隔离必须指明动物"):
            ledger.open_contingency(
                ContingencyKind.ISOLATION, [Stage.QUARANTINE], "原因", "条件", "v-us", T0
            )
        with self.assertRaisesRegex(ValueError, "受影响环节"):
            ledger.open_contingency(
                ContingencyKind.QUARANTINE_HOLD, [], "原因", "条件", "ops-1", T0
            )
        with self.assertRaisesRegex(ValueError, "可接受替代饲料"):
            ledger.open_contingency(
                ContingencyKind.FEED_SUBSTITUTION, [Stage.TRANSIT], "原因", "条件", "ops-1", T0
            )


class ReportIdempotencyTest(unittest.TestCase):
    def test_duplicate_report_returns_original_acceptance(self) -> None:
        ledger, a = build_journey()
        first = ledger.submit_report(
            "航程温控记录", "hash-temp-001", "air-1", T0 + timedelta(hours=12),
            findings={"舱温均值": "15.2℃"},
        )
        again = ledger.submit_report(
            "航程温控记录（重传）", "hash-temp-001", "air-1", T0 + timedelta(days=1),
            findings={"舱温均值": "15.2℃"},
        )
        self.assertTrue(again.duplicate)
        self.assertEqual(again.record.seq, first.record.seq)
        self.assertIs(again.status, ReportStatus.ACCEPTED)
        self.assertEqual(len([r for r in ledger.records if r.type.value == "报告受理"]), 1)

    def test_contradictory_report_goes_to_designated_vet(self) -> None:
        ledger, a = build_journey()
        ledger.add_vet_conclusion(
            "fushuang", Stage.QUARANTINE, "检疫未见异常", {"寄生虫": "阴性"}, "v-cn",
            T0 + timedelta(days=2),
        )
        result = ledger.submit_report(
            "中方复检粪检报告", "hash-lab-09", "v-cn", T0 + timedelta(days=4),
            animal_id="fushuang", findings={"寄生虫": "阳性"},
        )
        self.assertIs(result.status, ReportStatus.PENDING_REVIEW)
        self.assertEqual(result.reviewer_id, "v-us")
        contradicted = result.record.payload["contradicts_conclusion"]
        self.assertEqual(contradicted, 4)  # 计划1 + 批次2 + 结论1

        # 非指定兽医不能裁决；裁决必须写依据。
        with self.assertRaises(PermissionError):
            ledger.resolve_review(result.record.seq, False, "驳回", "v-cn", T0 + timedelta(days=5))
        with self.assertRaisesRegex(ValueError, "必须说明依据"):
            ledger.resolve_review(result.record.seq, False, "", "v-us", T0 + timedelta(days=5))

        ledger.resolve_review(
            result.record.seq, False, "复核采样两份均阴性，原报告样本污染", "v-us",
            T0 + timedelta(days=5),
        )
        self.assertIs(ledger.report_status(result.record.seq), ReportStatus.REJECTED)
        # 驳回不改原结论；重复上传仍返回原报告与最终裁决。
        repeat = ledger.submit_report(
            "中方复检粪检报告", "hash-lab-09", "v-cn", T0 + timedelta(days=6),
            animal_id="fushuang", findings={"寄生虫": "阳性"},
        )
        self.assertTrue(repeat.duplicate)
        self.assertIs(repeat.status, ReportStatus.REJECTED)
        snap = ledger.snapshot(T0 + timedelta(days=7))
        self.assertEqual(snap.active_conclusions["fushuang"].payload["findings"]["寄生虫"], "阴性")

    def test_approved_contradiction_appends_vet_correction(self) -> None:
        ledger, a = build_journey()
        conclusion = ledger.add_vet_conclusion(
            "pingping", Stage.ADAPTATION, "适应良好", {"采食量": "正常"}, "v-us", T0 + timedelta(days=8)
        )
        result = ledger.submit_report(
            "采食观察周报", "hash-feed-77", "k-cn", T0 + timedelta(days=10),
            animal_id="pingping", findings={"采食量": "下降30%"},
        )
        self.assertIs(result.status, ReportStatus.PENDING_REVIEW)
        ledger.resolve_review(
            result.record.seq, True, "现场监控与称重佐证，采纳", "v-us", T0 + timedelta(days=11)
        )
        self.assertIs(ledger.report_status(result.record.seq), ReportStatus.ACCEPTED)
        snap = ledger.snapshot(T0 + timedelta(days=12))
        active = snap.active_conclusions["pingping"]
        self.assertEqual(active.type.value, "兽医纠正")
        self.assertEqual(active.payload["supersedes"], conclusion.seq)
        self.assertEqual(active.payload["findings"]["采食量"], "下降30%")

    def test_review_requires_designation(self) -> None:
        ledger, a = build_journey()
        ledger2 = CareLedger()
        vet = ledger2.register_actor("v1", "兽医", Side.US, {Role.VET})
        ledger2.register_animal("a1", "一")
        ledger2.add_vet_conclusion("a1", Stage.QUARANTINE, "结论", {"x": 1}, "v1", T0)
        with self.assertRaisesRegex(ValueError, "尚未指定复核兽医"):
            ledger2.submit_report("r", "h", "v1", T0, animal_id="a1", findings={"x": 2})


class BatchAndHandoffTest(unittest.TestCase):
    def test_batch_overdraw_category_and_duplicate_are_rejected(self) -> None:
        ledger, a = build_journey()
        with self.assertRaisesRegex(ValueError, "结余不足"):
            ledger.record_feeding(
                "pingping", Stage.TRANSIT, "华西箭竹", 100.0, "k-cn", T0 + timedelta(hours=3), "B2026-01"
            )
        with self.assertRaisesRegex(ValueError, "品类不一致"):
            ledger.record_feeding(
                "pingping", Stage.TRANSIT, "冷箭竹", 1.0, "k-cn", T0 + timedelta(hours=3), "B2026-01"
            )
        with self.assertRaisesRegex(ValueError, "批次号重复"):
            ledger.receive_batch("B2026-01", "华西箭竹", 5.0, "air-1", T0 + timedelta(hours=2))

    def test_snapshot_reconstructs_material_flows_and_handoff_basis(self) -> None:
        ledger, a = build_journey()
        ledger.record_feeding("pingping", Stage.TRANSIT, "华西箭竹", 6.0, "k-cn",
                              T0 + timedelta(hours=3), "B2026-01")
        ledger.record_feeding("fushuang", Stage.TRANSIT, "华西箭竹", 4.0, "k-cn",
                              T0 + timedelta(hours=4), "B2026-01")
        handoff = ledger.handoff(
            "k-cn", "k-us", Stage.QUARANTINE, ("pingping", "fushuang"), plan_seq=1,
            batch_ids=["B2026-01", "B2026-02"], occurred_at=T0 + timedelta(hours=22),
        )
        snap = ledger.snapshot(T0 + timedelta(days=1))
        self.assertAlmostEqual(snap.batch_balances["B2026-01"], 50.0)
        self.assertAlmostEqual(snap.batch_balances["B2026-02"], 10.0)
        self.assertEqual(handoff.payload["basis_seq_from"], 1)
        self.assertEqual(handoff.payload["basis_seq_to"], handoff.seq - 1)

        # 交接不能引用不存在的计划或批次。
        with self.assertRaisesRegex(ValueError, "未知饲料批次"):
            ledger.handoff("k-us", "k-cn", Stage.ADAPTATION, ("pingping",), 1,
                           batch_ids=["B-NOPE"], occurred_at=T0 + timedelta(days=2))
        with self.assertRaisesRegex(ValueError, "照护计划记录"):
            ledger.handoff("k-us", "k-cn", Stage.ADAPTATION, ("pingping",), 2,
                           occurred_at=T0 + timedelta(days=2))


class VisibilityAndReplayTest(unittest.TestCase):
    def test_each_side_sees_only_its_scope_plus_shared(self) -> None:
        ledger, a = build_journey()
        ledger.add_vet_conclusion("pingping", Stage.QUARANTINE, "共享结论", {"x": 1}, "v-us", T0)
        ledger.record_medication(
            "pingping", Stage.QUARANTINE, "美方内部用药试验", "1ml", "v-us", T0,
            visibility=Visibility.US_ONLY,
        )
        ledger.record_behavior(
            ["pingping"], Stage.PRE_DEPARTURE, "中方内部行为基线备注", anomaly=False, actor_id="k-cn",
            occurred_at=T0, visibility=Visibility.CN_ONLY,
        )
        cn = ledger.view_for(Side.CN)
        us = ledger.view_for(Side.US)
        self.assertTrue(any("内部用药" in r.payload.get("medication", "") for r in us))
        self.assertFalse(any("内部用药" in r.payload.get("medication", "") for r in cn))
        self.assertTrue(any("内部行为基线" in r.payload.get("note", "") for r in cn))
        self.assertFalse(any("内部行为基线" in r.payload.get("note", "") for r in us))
        # 共享记录双方都能看到。
        self.assertTrue(any(r.type.value == "兽医结论" for r in cn))
        self.assertEqual(len(ledger.view_for(Side.JOINT)), len(ledger.records))

    def test_as_of_replay_reproduces_state_at_dispute_point(self) -> None:
        ledger, a = build_journey()
        t1 = T0 + timedelta(hours=2)
        t2 = T0 + timedelta(days=2)
        t3 = T0 + timedelta(days=6)
        ledger.publish_plan(
            Stage.ADAPTATION, foods=["华西箭竹"], items=["丰容每日2次"],
            actor_id="ops-1", occurred_at=t2,
        )
        conclusion = ledger.add_vet_conclusion(
            "pingping", Stage.ADAPTATION, "适应良好", {"采食量": "正常"}, "v-us", t2 + timedelta(hours=6)
        )
        ledger.open_contingency(
            ContingencyKind.ISOLATION, [Stage.ADAPTATION], "粪检复查", "三次阴性",
            "v-us", t2 + timedelta(days=1), animal_ids=["pingping"],
        )
        ledger.correct_conclusion(
            conclusion.seq, "采食下降需干预", {"采食量": "下降30%"}, "周报佐证",
            "v-us", t3,
        )

        # 争议时点 1：适应期计划尚未发布。
        early = ledger.snapshot(t1)
        self.assertNotIn(Stage.ADAPTATION, early.plans)
        self.assertEqual(early.active_contingencies, ())
        self.assertEqual(early.active_conclusions, {})
        self.assertTrue(early.chain_ok)

        # 争议时点 2：原始结论仍有效、隔离生效中。
        mid = ledger.snapshot(t2 + timedelta(days=2))
        self.assertEqual(mid.active_conclusions["pingping"].seq, conclusion.seq)
        self.assertEqual(len(mid.active_contingencies), 1)
        self.assertEqual(mid.active_contingencies[0].payload["kind"], "单只隔离")

        # 争议时点 3：纠正已生效。
        later = ledger.snapshot(t3 + timedelta(days=1))
        self.assertEqual(later.active_conclusions["pingping"].payload["findings"]["采食量"], "下降30%")

        # 美方视图的时点回放看不到仅中方记录，反之亦然。
        ledger.record_behavior(
            ["pingping"], Stage.ADAPTATION, "中方内部备注", False, "k-cn", t2,
            visibility=Visibility.CN_ONLY,
        )
        self.assertFalse(
            any("中方内部备注" in r.payload.get("note", "")
                for r in ledger.snapshot(t3, viewer=Side.US).records)
        )
        self.assertTrue(
            any("中方内部备注" in r.payload.get("note", "")
                for r in ledger.snapshot(t3, viewer=Side.CN).records)
        )

    def test_timestamps_cross_timezones_are_ordered_correctly(self) -> None:
        ledger, a = build_journey()
        # 美方当地时间晚上 20 点（UTC-4）与中方次日早 8 点（UTC+8）为同一时刻。
        us_local = datetime(2026, 10, 2, 20, 0, tzinfo=US_TZ)
        cn_local = us_local.astimezone(CN_TZ)
        ledger.record_weight("pingping", Stage.QUARANTINE, 101.0, "k-us", us_local)
        ledger.record_weight("fushuang", Stage.QUARANTINE, 98.0, "k-cn", cn_local)
        times = [r.occurred_at for r in ledger.records if r.type.value == "体重"]
        self.assertEqual(times[0].astimezone(timezone.utc), times[1].astimezone(timezone.utc))


if __name__ == "__main__":
    unittest.main()
