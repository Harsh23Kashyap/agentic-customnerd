import asyncio
import json
import sse_utils


def test_send_update_creates_queue():
    sse_utils.update_queues.clear()
    sse_utils.send_update_sync("x", "hello")
    assert sse_utils.update_queues["x"].get_nowait() == "hello"


def test_event_generator_wraps_progress_and_finishes():
    async def run():
        sse_utils.update_queues.clear()
        sse_utils.send_update_sync("x", "working")
        sse_utils.send_update_sync("x", {"end_output": "done"})
        out = []
        async for event in sse_utils.event_generator("x"):
            out.append(json.loads(event["data"]))
        return out
    assert asyncio.run(run()) == [{"update": "working"}, {"end_output": "done"}]
    assert "x" not in sse_utils.update_queues
