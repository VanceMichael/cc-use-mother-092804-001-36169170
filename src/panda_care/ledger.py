"""只追加哈希链台账。

- 事件只能追加，不能修改或删除；
- 每条事件携带序号、前一条哈希与自身哈希，形成可独立校验的链；
- 校验在争议时点回放时重放全部哈希，任一记录被篡改即失败。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from .models import CareEvent

GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    """台账完整性被破坏。"""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_event(event: CareEvent, prev_hash: str) -> str:
    """计算事件哈希。不含 seq/prev_hash/hash 字段本身，但包含 prev_hash。"""
    payload = {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "occurred_at": event.occurred_at.astimezone(timezone.utc).isoformat(),
        "recorded_at": event.recorded_at.astimezone(timezone.utc).isoformat(),
        "actor_id": event.actor_id,
        "phase": event.phase.value,
        "scope": event.scope.to_dict(),
        "visible_to": event.visible_to.value,
        "body": event.body,
        "prev_hash": prev_hash,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class Ledger:
    """内存态只追加台账；持久化由调用方通过快照接口处理。"""

    def __init__(self, events: Iterable[CareEvent] | None = None) -> None:
        self._events: list[CareEvent] = []
        if events is not None:
            for event in events:
                self._restore(event)

    # ---- 写入 -----------------------------------------------------------

    def append(self, event: CareEvent) -> CareEvent:
        if event.seq is not None or event.hash is not None:
            raise LedgerError("事件已经入链，不能重复追加")
        if any(existing.event_id == event.event_id for existing in self._events):
            raise LedgerError(f"事件标识重复: {event.event_id}")
        if event.occurred_at.tzinfo is None or event.recorded_at.tzinfo is None:
            raise LedgerError("事件时间必须带时区")
        prev_hash = self._events[-1].hash if self._events else GENESIS_HASH
        seq = len(self._events)
        digest = digest_event(event, prev_hash)
        sealed = event.sealed(seq, prev_hash, digest)
        self._events.append(sealed)
        return sealed

    # ---- 读取 -----------------------------------------------------------

    @property
    def events(self) -> tuple[CareEvent, ...]:
        return tuple(self._events)

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_HASH

    def get(self, event_id: str) -> CareEvent | None:
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def require(self, event_id: str) -> CareEvent:
        event = self.get(event_id)
        if event is None:
            raise LedgerError(f"事件不存在: {event_id}")
        return event

    def filter(self, predicate: Callable[[CareEvent], bool]) -> tuple[CareEvent, ...]:
        return tuple(event for event in self._events if predicate(event))

    def as_of(self, moment: datetime) -> tuple[CareEvent, ...]:
        """返回某一时点（以入链时间 recorded_at 为准）已存在的事件。"""
        if moment.tzinfo is None:
            raise LedgerError("回放时点必须带时区")
        moment = moment.astimezone(timezone.utc)
        return tuple(
            event for event in self._events
            if event.recorded_at.astimezone(timezone.utc) <= moment
        )

    # ---- 完整性 ---------------------------------------------------------

    def verify(self) -> None:
        """重放整条链。发现任何删改即抛出 LedgerError。"""
        prev_hash = GENESIS_HASH
        for index, event in enumerate(self._events):
            if event.seq != index:
                raise LedgerError(f"序号断裂: 期望 {index}，实际 {event.seq}")
            if event.prev_hash != prev_hash:
                raise LedgerError(f"前序哈希断裂: 事件 {event.event_id}")
            expected = digest_event(event, prev_hash)
            if event.hash != expected:
                raise LedgerError(f"事件内容哈希不匹配: {event.event_id}")
            prev_hash = event.hash

    # ---- 快照 -----------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        self.verify()
        return [event.to_dict() for event in self._events]

    def save_json(self, path: str) -> None:
        """把整链（含哈希）写入 JSON，供跨进程恢复与争议归档。"""
        import json
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: str) -> "Ledger":
        """从快照恢复并立即重放校验；任何篡改都会抛出 LedgerError。"""
        import json
        from pathlib import Path

        from .models import EventType, Phase, Scope, Side

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        events: list[CareEvent] = []
        for item in raw:
            events.append(
                CareEvent(
                    event_id=item["event_id"],
                    event_type=EventType(item["event_type"]),
                    occurred_at=datetime.fromisoformat(item["occurred_at"]),
                    actor_id=item["actor_id"],
                    phase=Phase(item["phase"]),
                    body=item["body"],
                    scope=Scope.from_dict(item["scope"]),
                    visible_to=Side(item["visible_to"]),
                    recorded_at=datetime.fromisoformat(item["recorded_at"]),
                    seq=item["seq"],
                    prev_hash=item["prev_hash"],
                    hash=item["hash"],
                )
            )
        ledger = cls(events)
        ledger.verify()
        return ledger

    def _restore(self, event: CareEvent) -> None:
        """从快照恢复（不重新计算，交由 verify 校验）。"""
        if event.seq != len(self._events):
            raise LedgerError("快照序号不连续")
        self._events.append(event)


def append_only_ledger() -> Ledger:
    """创建空台账的工厂函数。"""
    return Ledger()


def record_event(ledger: Ledger, event: CareEvent) -> CareEvent:
    """追加事件并立即校验链完整性。"""
    sealed = ledger.append(event)
    ledger.verify()
    return sealed
