"""SSE queue helpers — thread-safe queues for worker-thread agent + async SSE."""

from __future__ import annotations

import asyncio
import json
import queue
from typing import Any, Dict

update_queues: Dict[str, queue.Queue] = {}


def send_update_sync(session_id: str, data: Any) -> None:
    """Push an update from the agent worker thread (or async handler)."""
    q = update_queues.get(session_id)
    if q is None:
        q = queue.Queue()
        update_queues[session_id] = q
    q.put(data)


async def send_update(session_id: str, data: Any) -> None:
    send_update_sync(session_id, data)


async def event_generator(session_id: str):
    q = update_queues.get(session_id)
    if q is None:
        q = queue.Queue()
        update_queues[session_id] = q
    try:
        while True:
            data = await asyncio.to_thread(q.get)
            if isinstance(data, dict) and (
                "final_output" in data or "end_output" in data
            ):
                yield {"event": "message", "data": json.dumps(data)}
                break
            yield {"event": "message", "data": json.dumps({"update": data})}
    finally:
        update_queues.pop(session_id, None)
