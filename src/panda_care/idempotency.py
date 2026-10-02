"""报告上传幂等存储。

去重键是报告内容哈希：任何提交者重复上传同一份报告，都返回原始处理结果——
- 首次直接归档：重放 ACCEPTED；
- 首次提交开启复核但尚未出结论：重放 REVIEWING；
- 复核已结束：重放终态 REVIEW_ACCEPTED / REVIEW_REJECTED。
任何情况下重复提交都不产生新事件。矛盾判定与复核开启由 CareService 负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UploadStatus(str, Enum):
    ACCEPTED = "ACCEPTED"                    # 无矛盾，已归档
    REVIEW_PENDING = "REVIEW_PENDING"        # 首次提交已转指定兽医，等待结论
    REVIEWING = "REVIEWING"                  # 重复提交，且复核仍在进行
    REVIEW_ACCEPTED = "REVIEW_ACCEPTED"      # 复核采纳报告
    REVIEW_REJECTED = "REVIEW_REJECTED"      # 复核驳回报告


PENDING_STATUSES = frozenset({UploadStatus.REVIEW_PENDING, UploadStatus.REVIEWING})


@dataclass(frozen=True)
class UploadResult:
    status: UploadStatus
    event_id: str | None
    review_event_id: str | None
    duplicate: bool
    message: str


class IdempotencyStore:
    """以报告内容哈希为去重键的内存幂等表。"""

    def __init__(self) -> None:
        self._results: dict[str, UploadResult] = {}

    def get(self, uploader_actor_id: str, content_hash: str) -> UploadResult | None:
        del uploader_actor_id  # 去重只看内容，不看提交者
        return self._results.get(content_hash)

    def remember(self, uploader_actor_id: str, content_hash: str, result: UploadResult) -> None:
        del uploader_actor_id
        if result.duplicate:
            raise ValueError("幂等表不能记录重复结果本身")
        self._results[content_hash] = result

    def replace(self, uploader_actor_id: str, content_hash: str, result: UploadResult) -> None:
        """复核结束后更新原结果（保持同一幂等键）。"""
        del uploader_actor_id
        if result.duplicate:
            raise ValueError("幂等表不能记录重复结果本身")
        self._results[content_hash] = result
