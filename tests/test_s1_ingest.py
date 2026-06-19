#!/usr/bin/env python3
"""Integration tests for the S1 ingest worker -- the never-miss CONTROL logic
pixel-diff cannot exercise (SATELLITE-REARCH §9.x): completeness gate, ledger,
backfill reconcile (dropped event -> enqueue; duplicate -> no-op PUT),
idempotent-key derivation, delete-after-PUT, DLQ-after-maxReceiveCount, and
cold-start bootstrap. Real SQS semantics via moto (incl. redrive/DLQ); R2 is an
in-memory faithful S3 stand-in (the worker only put/head/delete/list).
"""
import datetime as dt
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

import s1_slots as S  # noqa: E402
import s1_ingest as W  # noqa: E402

UTC = dt.timezone.utc

C13_A = ("ABI-L2-CMIPM/2026/169/21/"
         "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692100572_e20261692101042_c20261692101089.nc")
C13_B = ("ABI-L2-CMIPM/2026/169/21/"
         "OR_ABI-L2-CMIPM2-M6C13_G19_s20261692101572_e20261692102042_c20261692102092.nc")
M1_C13 = ("ABI-L2-CMIPM/2026/169/21/"
          "OR_ABI-L2-CMIPM1-M6C13_G19_s20261692107272_e20261692107342_c20261692107379.nc")
C10 = ("ABI-L2-CMIPM/2026/169/21/"
       "OR_ABI-L2-CMIPM2-M6C10_G19_s20261692108572_e20261692109043_c20261692109091.nc")
RADC = ("ABI-L1b-RadC/2026/169/21/"
        "OR_ABI-L1b-RadC-M6C08_G19_s20261692106194_e20261692108567_c20261692109023.nc")

BBOX = [100.0, 20.0, 112.0, 32.0]


def raw_event(key):
    return json.dumps({"Records": [{"s3": {"object": {"key": key}}}]})


def sns_envelope(key):
    inner = raw_event(key)
    return json.dumps({"Type": "Notification", "TopicArn": "t", "Message": inner})


class RenderStub:
    """Configurable stub for render_fn(bbox, time_iso) -> (bytes, headers)."""
    def __init__(self, mode="ok", xscan_override=None):
        self.mode = mode
        self.xscan_override = xscan_override
        self.calls = []

    def __call__(self, bbox, time_iso):
        self.calls.append((tuple(bbox), time_iso))
        if self.mode == "render_error":
            raise W.RenderError("boom")
        if self.mode == "skip":
            raise W.RenderSkip("422 off-disk")
        # Echo the requested scan time as X-Scan-Time (resolved == requested),
        # unless overridden to simulate the s3fs-listing-not-current case.
        d = dt.datetime.fromisoformat(time_iso)
        stamp = d.astimezone(UTC).strftime(S.STAMP_FMT)
        headers = {"Content-Type": "image/webp"}
        if self.mode != "no_xscan":   # simulate a render with no X-Scan-Time
            headers["X-Scan-Time"] = (self.xscan_override
                                      or d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        return (b"WEBPDATA-" + stamp.encode(), headers)


class FakeR2:
    """In-memory faithful S3 stand-in (R2 is S3-compatible; worker uses only
    put_bytes/put_json/head/delete/list_keys)."""
    def __init__(self):
        self.store = {}

    def put_bytes(self, key, data, content_type, cache):
        self.store[key] = data
        return True

    def put_json(self, key, obj, cache):
        self.store[key] = json.dumps(obj).encode()
        return True

    def head(self, key):
        return key in self.store

    def delete(self, keys):
        for k in keys:
            self.store.pop(k, None)

    def list_keys(self, prefix):
        return [k for k in self.store if k.startswith(prefix)]


class FailingPutR2(FakeR2):
    def put_bytes(self, key, data, content_type, cache):
        if key.endswith(".webp"):
            return False           # simulate an R2 frame-PUT failure
        return super().put_bytes(key, data, content_type, cache)


class FailingProdPutR2(FakeR2):
    """Shadow PUTs succeed; the PROD meso key PUT fails (the cutover never-miss
    edge: prod failure must NOT ack the slot)."""
    def put_bytes(self, key, data, content_type, cache):
        if key.startswith(S.S1_PROD_PRODUCT_PATH):
            return False
        return super().put_bytes(key, data, content_type, cache)


@mock_aws
class S1IngestTest(unittest.TestCase):
    def setUp(self):
        self.sqs_client = boto3.client("sqs", region_name="us-east-1")
        self.dlq_url = self.sqs_client.create_queue(QueueName="s1-dlq")["QueueUrl"]
        dlq_arn = self.sqs_client.get_queue_attributes(
            QueueUrl=self.dlq_url, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        self.q_url = self.sqs_client.create_queue(
            QueueName="s1-main",
            Attributes={
                "VisibilityTimeout": "0",   # immediate redelivery for the DLQ test
                "RedrivePolicy": json.dumps(
                    {"deadLetterTargetArn": dlq_arn, "maxReceiveCount": 2}),
            },
        )["QueueUrl"]

    def _send(self, body):
        self.sqs_client.send_message(QueueUrl=self.q_url, MessageBody=body)

    def _q_count(self, url):
        a = self.sqs_client.get_queue_attributes(
            QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages"])
        return int(a["Attributes"]["ApproximateNumberOfMessages"])

    def _worker(self, render=None, r2=None, ground_truth=None, extent=None,
                clock=None, prod_write=False):
        return W.S1Ingest(
            r2=r2 or FakeR2(),
            sqs=W.SQS(self.q_url),
            render_fn=render or RenderStub(),
            extent_fn=extent or (lambda k: list(BBOX)),
            ground_truth_fn=ground_truth or (lambda lookback: []),
            prefix="shadow",
            clock=clock or W.utcnow,
            prod_write=prod_write,
        )

    # -- happy path: render + publish + delete -------------------------------
    def test_consume_renders_and_publishes_and_deletes(self):
        self._send(raw_event(C13_A))
        r2 = FakeR2()
        render = RenderStub()
        w = self._worker(render=render, r2=r2)
        n = w.consume_once(wait_s=1)
        self.assertEqual(n, 1)
        self.assertEqual(w.stats.rendered, 1)
        key = S.shadow_frame_key("shadow", "20260618T210057Z")
        self.assertIn(key, r2.store)                       # frame PUT
        self.assertIn(S.latest_times_key("shadow"), r2.store)  # SSOT written
        lt = json.loads(r2.store[S.latest_times_key("shadow")])
        self.assertEqual(lt["latest"], "20260618T210057Z")
        self.assertEqual(self._q_count(self.q_url), 0)     # message deleted
        # rendered with the slot's bbox + specific (non-"latest") time
        self.assertEqual(render.calls[0][0], tuple(BBOX))
        self.assertEqual(render.calls[0][1], "2026-06-18T21:00:57+00:00")

    def test_sns_envelope_unwrapped_and_rendered(self):
        self._send(sns_envelope(C13_A))
        r2 = FakeR2()
        w = self._worker(r2=r2)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 1)
        self.assertIn(S.shadow_frame_key("shadow", "20260618T210057Z"), r2.store)

    # -- non-S1 messages are acked, never rendered ---------------------------
    def test_non_s1_acked_not_rendered(self):
        for k in (RADC, M1_C13, C10):
            self._send(raw_event(k))
        render = RenderStub()
        w = self._worker(render=render)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertEqual(render.calls, [])               # never rendered
        self.assertEqual(w.stats.not_ours, 3)
        self.assertEqual(self._q_count(self.q_url), 0)   # all acked

    # -- idempotency ----------------------------------------------------------
    def test_duplicate_event_no_double_put(self):
        self._send(raw_event(C13_A))
        self._send(raw_event(C13_A))                     # same slot twice
        render = RenderStub()
        w = self._worker(render=render)
        w.consume_once(wait_s=1)
        self.assertEqual(len(render.calls), 1)           # rendered once only
        self.assertEqual(w.stats.rendered, 1)
        self.assertEqual(w.stats.duplicates, 1)

    def test_already_in_r2_skips_render(self):
        r2 = FakeR2()
        r2.store[S.shadow_frame_key("shadow", "20260618T210057Z")] = b"existing"
        self._send(raw_event(C13_A))
        render = RenderStub()
        w = self._worker(render=render, r2=r2)
        w.consume_once(wait_s=1)
        self.assertEqual(render.calls, [])               # head() hit -> no render
        self.assertEqual(w.stats.duplicates, 1)
        self.assertEqual(self._q_count(self.q_url), 0)   # still acked

    # -- delete-after-PUT: a render/PUT failure must NOT ack -----------------
    def test_render_failure_does_not_delete(self):
        self._send(raw_event(C13_A))
        w = self._worker(render=RenderStub(mode="render_error"))
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertGreaterEqual(w.stats.render_fail, 1)
        # VisibilityTimeout=0 -> the undeleted message is immediately visible again.
        self.assertEqual(self._q_count(self.q_url), 1)

    def test_put_failure_does_not_delete(self):
        self._send(raw_event(C13_A))
        w = self._worker(r2=FailingPutR2())
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.put_fail, 1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertEqual(self._q_count(self.q_url), 1)   # NOT acked

    def test_xscan_mismatch_retries_not_acked(self):
        # Render resolved a DIFFERENT scan (s3fs listing not current) -> must not
        # publish/ack; SQS redelivers later when the listing refreshes.
        self._send(raw_event(C13_A))
        r2 = FakeR2()
        w = self._worker(render=RenderStub(xscan_override="2026-06-18T20:59:57Z"),
                         r2=r2)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertNotIn(S.shadow_frame_key("shadow", "20260618T210057Z"), r2.store)
        self.assertEqual(self._q_count(self.q_url), 1)   # NOT acked

    def test_missing_xscan_fails_closed(self):
        # No X-Scan-Time header -> cannot verify the slot -> must NOT publish/ack.
        self._send(raw_event(C13_A))
        r2 = FakeR2()
        w = self._worker(render=RenderStub(mode="no_xscan"), r2=r2)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertNotIn(S.shadow_frame_key("shadow", "20260618T210057Z"), r2.store)
        self.assertEqual(self._q_count(self.q_url), 1)   # NOT acked -> retry/DLQ

    # -- 422 off-disk -> acked as no-data (not a failure, not a frame) -------
    def test_render_skip_acked_as_no_data(self):
        self._send(raw_event(C13_A))
        w = self._worker(render=RenderStub(mode="skip"))
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.no_data, 1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertEqual(self._q_count(self.q_url), 0)   # acked (covered)

    # -- DLQ after maxReceiveCount (real moto redrive) -----------------------
    def test_dlq_after_max_receive(self):
        self._send(raw_event(C13_A))
        w = self._worker(render=RenderStub(mode="render_error"))
        # maxReceiveCount=2 -> after 2 failed receives the message redrives to DLQ.
        for _ in range(5):
            w.consume_once(wait_s=0)
        self.assertEqual(self._q_count(self.q_url), 0)   # gone from main
        self.assertEqual(self._q_count(self.dlq_url), 1) # landed in DLQ
        self.assertEqual(w.stats.rendered, 0)

    # -- backfill reconcile ---------------------------------------------------
    def test_backfill_renders_dropped_slot(self):
        # SQS never delivered slot B; backfill (newest-first) renders it.
        slot_a = S.parse_goes_key(C13_A)   # older
        slot_b = S.parse_goes_key(C13_B)   # newer
        render = RenderStub()
        w = self._worker(render=render,
                         ground_truth=lambda lookback: [slot_b, slot_a])
        w.published[slot_a.stamp] = "x"    # A already published
        w.gate.seed_complete(slot_a.stamp)
        n = w.backfill_once()
        self.assertEqual(n, 1)
        self.assertEqual(w.stats.backfilled, 1)
        self.assertEqual([c[1] for c in render.calls],
                         ["2026-06-18T21:01:57+00:00"])  # only B rendered
        self.assertEqual(w.watermark, slot_b.scan_start) # watermark advanced

    def test_backfill_duplicate_no_op(self):
        slot_a = S.parse_goes_key(C13_A)
        render = RenderStub()
        w = self._worker(render=render,
                         ground_truth=lambda lookback: [slot_a])
        w.published[slot_a.stamp] = "x"
        w.gate.seed_complete(slot_a.stamp)
        n = w.backfill_once()
        self.assertEqual(n, 0)
        self.assertEqual(render.calls, [])               # no re-render

    def test_backfill_list_failure_preserves_state(self):
        def boom(lookback):
            raise OSError("S3 list hiccup")
        w = self._worker(ground_truth=boom)
        w.watermark = dt.datetime(2026, 6, 18, 21, 0, 57, tzinfo=UTC)
        n = w.backfill_once()
        self.assertEqual(n, 0)
        self.assertEqual(w.watermark,                    # state preserved
                         dt.datetime(2026, 6, 18, 21, 0, 57, tzinfo=UTC))
        self.assertEqual(w.health["backfill"].consecutive_failures, 1)

    # -- cold start: seed ledger + watermark from R2 reality (§3.6) -----------
    def test_cold_start_seeds_from_r2(self):
        r2 = FakeR2()
        stamps = ["20260618T210057Z", "20260618T210157Z", "20260618T210257Z"]
        for s in stamps:
            r2.store[S.shadow_frame_key("shadow", s)] = b"frame"
        r2.store[S.latest_times_key("shadow")] = b"{}"   # must be ignored
        w = self._worker(r2=r2)
        w.cold_start()
        self.assertEqual(set(w.published), set(stamps))
        self.assertEqual(w.watermark,
                         dt.datetime(2026, 6, 18, 21, 2, 57, tzinfo=UTC))
        for s in stamps:
            self.assertTrue(w.gate.is_complete(s))

    def test_cold_start_empty_bucket(self):
        w = self._worker(r2=FakeR2())
        w.cold_start()
        self.assertEqual(w.published, {})
        self.assertIsNone(w.watermark)

    def test_cold_start_then_event_no_rerender(self):
        # A slot already in R2 at cold start must not re-render when its (late)
        # SQS event arrives.
        r2 = FakeR2()
        r2.store[S.shadow_frame_key("shadow", "20260618T210057Z")] = b"frame"
        render = RenderStub()
        w = self._worker(render=render, r2=r2)
        w.cold_start()
        self._send(raw_event(C13_A))
        w.consume_once(wait_s=1)
        self.assertEqual(render.calls, [])               # idempotent post-cold-start
        self.assertEqual(w.stats.duplicates, 1)

    # -- prod cutover (S1_PROD_WRITE): dual-write shadow + prod meso key -------
    def test_prod_write_dual_writes_both_keys(self):
        self._send(raw_event(C13_A))
        r2 = FakeR2()
        render = RenderStub()
        w = self._worker(render=render, r2=r2, prod_write=True)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 1)
        self.assertEqual(len(render.calls), 1)             # rendered ONCE
        shadow_key = S.shadow_frame_key("shadow", "20260618T210057Z")
        prod_key = S.prod_frame_key("20260618T210057Z")
        self.assertIn(shadow_key, r2.store)                # shadow frame
        self.assertIn(prod_key, r2.store)                  # prod meso frame
        self.assertEqual(prod_key, "meso/goes19-m2/ir/20260618T210057Z.webp")
        self.assertEqual(r2.store[shadow_key], r2.store[prod_key])  # identical bytes
        self.assertIn(S.latest_times_key("shadow"), r2.store)       # SSOT still shadow
        self.assertEqual(self._q_count(self.q_url), 0)     # acked (both PUTs ok)

    def test_prod_put_failure_does_not_ack(self):
        # Shadow PUT ok, PROD PUT fails -> raise -> message NOT acked (redeliver).
        self._send(raw_event(C13_A))
        r2 = FailingProdPutR2()
        w = self._worker(r2=r2, prod_write=True)
        w.consume_once(wait_s=1)
        self.assertEqual(w.stats.rendered, 0)
        self.assertGreaterEqual(w.stats.put_fail, 1)
        self.assertNotIn(S.prod_frame_key("20260618T210057Z"), r2.store)
        self.assertNotIn("20260618T210057Z", w.published)  # not marked published
        self.assertEqual(self._q_count(self.q_url), 1)     # NOT acked -> retry/DLQ

    def test_prod_idempotent_when_both_present(self):
        r2 = FakeR2()
        r2.store[S.shadow_frame_key("shadow", "20260618T210057Z")] = b"x"
        r2.store[S.prod_frame_key("20260618T210057Z")] = b"x"
        self._send(raw_event(C13_A))
        render = RenderStub()
        w = self._worker(render=render, r2=r2, prod_write=True)
        w.consume_once(wait_s=1)
        self.assertEqual(render.calls, [])                 # both present -> no render
        self.assertEqual(w.stats.duplicates, 1)
        self.assertEqual(self._q_count(self.q_url), 0)     # acked

    def test_prod_backfills_missing_prod_when_shadow_present(self):
        # Cutover just enabled: shadow exists (from before), prod missing -> render
        # and write ONLY the missing prod key (shadow not re-PUT).
        r2 = FakeR2()
        r2.store[S.shadow_frame_key("shadow", "20260618T210057Z")] = b"old-shadow"
        self._send(raw_event(C13_A))
        render = RenderStub()
        w = self._worker(render=render, r2=r2, prod_write=True)
        w.consume_once(wait_s=1)
        self.assertEqual(len(render.calls), 1)             # re-rendered for prod
        self.assertIn(S.prod_frame_key("20260618T210057Z"), r2.store)
        self.assertEqual(r2.store[S.shadow_frame_key("shadow", "20260618T210057Z")],
                         b"old-shadow")                    # shadow untouched
        self.assertEqual(self._q_count(self.q_url), 0)

    # -- cold-start target-completeness (the cutover/restart prod-gap critical) -
    def test_cold_start_prod_missing_leaves_unseeded_then_writes_prod(self):
        # After cutover/restart with prod_write=True, a shadow-present/prod-absent
        # slot must NOT be seeded into the ledger (else the ledger short-circuit
        # suppresses its prod PUT forever). It stays re-processable: the next SQS
        # event writes the missing prod frame.
        r2 = FakeR2()
        stamp = "20260618T210057Z"
        r2.store[S.shadow_frame_key("shadow", stamp)] = b"shadow"   # shadow only
        render = RenderStub()
        w = self._worker(render=render, r2=r2, prod_write=True)
        w.cold_start()
        self.assertNotIn(stamp, w.published)            # NOT seeded (prod missing)
        self.assertEqual(w.watermark,                   # watermark still advanced
                         dt.datetime(2026, 6, 18, 21, 0, 57, tzinfo=UTC))
        self._send(raw_event(C13_A))
        w.consume_once(wait_s=1)
        self.assertIn(S.prod_frame_key(stamp), r2.store)   # prod now written
        self.assertEqual(w.stats.rendered, 1)

    def test_cold_start_prod_present_seeds_normally(self):
        r2 = FakeR2()
        stamp = "20260618T210057Z"
        r2.store[S.shadow_frame_key("shadow", stamp)] = b"shadow"
        r2.store[S.prod_frame_key(stamp)] = b"prod"     # BOTH targets present
        w = self._worker(r2=r2, prod_write=True)
        w.cold_start()
        self.assertIn(stamp, w.published)               # seeded (both present)
        self.assertTrue(w.gate.is_complete(stamp))

    # -- stale-slot guard: ancient backlog acked, not rendered ----------------
    def test_stale_slot_acked_not_rendered(self):
        self._send(raw_event(C13_A))                     # scan 2026-06-18 21:00
        render = RenderStub()
        # Clock far in the future -> the slot is older than RETAIN_H.
        future = dt.datetime(2026, 7, 1, tzinfo=UTC)
        w = self._worker(render=render, clock=lambda: future)
        w.consume_once(wait_s=1)
        self.assertEqual(render.calls, [])               # not rendered
        self.assertEqual(self._q_count(self.q_url), 0)   # acked


if __name__ == "__main__":
    unittest.main()
