"""app.render_slot: a /render request that gets no slot within its wait budget
is refused with 503 instead of queueing past its caller's timeout.

Regression for the 2026-09-02 box1 incident: pollers gave up after 45-60 s but
their requests kept their place in the semaphore queue and were rendered
hours later, output discarded (meso-render reached a 27.7 h backlog; render-1
drained a 14-21 h backlog at the month-end shaper lift and helped trip the
provider's CPU limiter).

Offline: exercises the context manager and the header parser directly, with a
fresh 1-slot semaphore, no network, no satellite data.

Run: python -m unittest tests.test_render_queue_deadline
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app  # noqa: E402
from fastapi import HTTPException  # noqa: E402


class _Req:
    def __init__(self, headers=None):
        self.headers = headers or {}


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class RenderSlot(unittest.TestCase):
    def setUp(self):
        self._sem = app.render_semaphore
        app.render_semaphore = asyncio.Semaphore(1)
        app.render_queue_depth = 0

    def tearDown(self):
        app.render_semaphore = self._sem

    def test_second_waiter_gets_503_with_retry_after_and_no_slot_leaks(self):
        async def scenario():
            events = []

            async def holder():
                async with app.render_slot(5.0):
                    events.append("hold")
                    await asyncio.sleep(0.6)
                events.append("released")

            async def waiter():
                await asyncio.sleep(0.05)
                try:
                    async with app.render_slot(0.2):
                        events.append("waiter-got-slot")   # must not happen
                except HTTPException as e:
                    events.append(("503", e.status_code, e.headers.get("Retry-After"), e.detail))

            await asyncio.gather(holder(), waiter())
            # after the holder releases, a fresh acquire must succeed at once:
            # the timed-out waiter did not consume the slot.
            async with app.render_slot(0.1):
                events.append("fresh-acquire")
            return events

        ev = _run(scenario())
        self.assertEqual(ev[0], "hold")
        self.assertEqual(ev[1][0], "503")
        self.assertEqual(ev[1][1], 503)
        self.assertEqual(ev[1][2], "1")            # int(max(1, min(0.2, 60)))
        self.assertIn("no slot within 0.2s", ev[1][3])
        self.assertNotIn("waiter-got-slot", ev)
        self.assertEqual(ev[-1], "fresh-acquire")
        self.assertEqual(app.render_queue_depth, 0)

    def test_slot_released_when_body_raises(self):
        async def scenario():
            with self.assertRaises(RuntimeError):
                async with app.render_slot(1.0):
                    raise RuntimeError("render blew up")
            async with app.render_slot(0.1):
                return "ok"
        self.assertEqual(_run(scenario()), "ok")
        self.assertEqual(app.render_queue_depth, 0)

    def test_waiter_that_frees_in_time_gets_the_slot(self):
        async def scenario():
            got = []

            async def holder():
                async with app.render_slot(5.0):
                    await asyncio.sleep(0.1)

            async def waiter():
                await asyncio.sleep(0.02)
                async with app.render_slot(2.0):
                    got.append("late-but-in-time")

            await asyncio.gather(holder(), waiter())
            return got
        self.assertEqual(_run(scenario()), ["late-but-in-time"])

    def test_retry_after_floor_and_cap(self):
        self.assertEqual(app.retry_after_s(0.2), "1")
        self.assertEqual(app.retry_after_s(40), "40")
        self.assertEqual(app.retry_after_s(59.9), "59")
        self.assertEqual(app.retry_after_s(300), "60")

    def test_queue_depth_counts_only_waiters(self):
        async def scenario():
            seen = []

            async def holder():
                async with app.render_slot(5.0):
                    await asyncio.sleep(0.15)
                    seen.append(("holding", app.render_queue_depth))

            async def waiter():
                await asyncio.sleep(0.05)
                try:
                    async with app.render_slot(0.05):
                        pass
                except HTTPException:
                    seen.append(("refused", app.render_queue_depth))

            await asyncio.gather(holder(), waiter())
            return seen
        seen = _run(scenario())
        self.assertIn(("refused", 0), seen)         # the refused waiter left the count
        self.assertIn(("holding", 0), seen)         # a slot holder is not "waiting"
        self.assertEqual(app.render_queue_depth, 0)


class RequestedWait(unittest.TestCase):
    def test_default_when_no_header(self):
        self.assertEqual(app.requested_wait_s(_Req()), app.RENDER_QUEUE_WAIT_S)
        self.assertEqual(app.requested_wait_s(None), app.RENDER_QUEUE_WAIT_S)

    def test_header_honored_and_capped(self):
        self.assertEqual(app.requested_wait_s(_Req({"x-render-wait-s": "120"})), 120.0)
        self.assertEqual(app.requested_wait_s(_Req({"x-render-wait-s": "9999"})), app.RENDER_QUEUE_WAIT_MAX_S)

    def test_garbage_and_nonpositive_fall_back(self):
        for raw in ("abc", "", "0", "-5", "nan"):
            self.assertEqual(app.requested_wait_s(_Req({"x-render-wait-s": raw})),
                             app.RENDER_QUEUE_WAIT_S, raw)


if __name__ == "__main__":
    unittest.main()
