"""Run the EXISTING browser ARCHER/ADT code headlessly and publish its fixes.

WHY THIS EXISTS
    satellite/explorer/objfix.js is a line-faithful ARCHER/ADT port whose every
    constant traces to primary source (OBJFIX-METHODS.md). It runs in the
    browser, and its output used to live only in ``window.ObjFix.tracks`` plus a
    manual download button - trapped in whoever's tab happened to be open.

    Server-side products (the center-fix plot, CycloLab, the guidance layer)
    need those fixes. The obvious route - reimplement ARCHER in Python - is the
    wrong one: it is ~1600 lines of carefully-sourced numerics, and a second
    implementation is a second thing to keep correct. Two implementations of a
    method that has documented departures D1-D7 WILL drift, and the drift would
    be invisible (both produce plausible lat/lons).

    So we run the SAME CODE. Headless Chromium loads the real explorer page,
    drives the panel through its existing programmatic seam
    (``window.ObjFixPanel`` - open/loadStorms/select/analyze/running), and reads
    back the same ``trackJSON()`` payload the download button produces. Zero
    algorithmic drift by construction: there is only one implementation, and
    this file contains none of the method.

    Crucially this also keeps the ORCHESTRATION, not just the algorithm. The
    per-frame first guess must stay the OFFICIAL-TRACK anchor (the floater box
    center); chaining ARCHER's own fixes un-anchors the penalty term and the
    track drifts (CLAUDE.md). That anchoring lives in the panel's runAnalysis,
    so driving the panel preserves it - calling archerFix() directly would not.

OUTPUT
    r2://<bucket>/cyclolab/objfix/<storm_id>.json   - one storm's fix track
    r2://<bucket>/cyclolab/objfix/index.json        - what was published, when

    The payload is trackJSON() VERBATIM plus a small envelope; we do not
    reshape the science fields. Every record carries the panel's own honesty
    disclosure string.

USAGE
    python objfix_headless.py                 # every storm the feed lists
    python objfix_headless.py --storm 12W     # one storm (name/id substring)
    python objfix_headless.py --no-publish    # dry run, print to stdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from typing import Any, Optional

log = logging.getLogger("objfix-headless")

SITE = os.environ.get("OBJFIX_SITE", "https://triple-a-tropics.com")
EXPLORER_PATH = os.environ.get("OBJFIX_EXPLORER_PATH", "/satellite/explorer/")
R2_PREFIX = os.environ.get("OBJFIX_R2_PREFIX", "cyclolab/objfix")
# A loop workup is tens of frames, each a BT-image fetch + an ARCHER search.
# Generous because the alternative to waiting is publishing a partial track.
RUN_TIMEOUT_S = int(os.environ.get("OBJFIX_RUN_TIMEOUT_S", "900"))
PAGE_TIMEOUT_S = int(os.environ.get("OBJFIX_PAGE_TIMEOUT_S", "120"))


# ---------------------------------------------------------------------------
# The browser side. Everything here is DRIVING, never computing.
# ---------------------------------------------------------------------------
COLLECTOR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "objfix_headless.cjs")


def _collect(storm_filter: Optional[str] = None,
             loop: bool = True) -> list[dict[str, Any]]:
    """Boot the explorer headlessly and return one trackJSON per storm.

    Driving happens in node (objfix_headless.cjs) because the browser
    automation and the page are both JS - the same python-wrapper/node-harness
    split the cyclolab shell tests already use. Python owns argument handling
    and publishing; node owns the browser.
    """
    import subprocess
    cmd = ["node", COLLECTOR, "--site", SITE, "--path", EXPLORER_PATH]
    if storm_filter:
        cmd += ["--storm", storm_filter]
    if not loop:
        cmd.append("--single")
    log.info("collector: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=RUN_TIMEOUT_S + PAGE_TIMEOUT_S + 300)
    for line in (proc.stderr or "").splitlines():
        log.info("%s", line)
    if proc.returncode != 0:
        raise RuntimeError(
            f"headless collector failed (rc={proc.returncode})")
    body = (proc.stdout or "").strip()
    if not body:
        return []
    return json.loads(body)


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------
class _R2Json:
    """Minimal R2 JSON writer.

    Deliberately NOT intensity_poller.R2Sink: importing that drags pandas +
    ace_core + the whole feed stack into a lane whose only job is to drive a
    browser and PUT a few KB of JSON. This lane runs on a Playwright image
    where that stack does not belong. Same headers as the poller's writer so
    the CDN behaviour is identical.
    """

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig
        # R2-only, hard-required (s2_prune pattern). endpoint=None would sign
        # real-AWS calls; an AWS_* cred fallback (explicit OR via boto3's
        # default chain) would ship the real tat-sat-ingest key to Cloudflare.
        missing = [n for n in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID",
                               "R2_SECRET_ACCESS_KEY")
                   if not os.environ.get(n)]
        if missing:
            sys.exit(f"ERROR: {', '.join(missing)} not set. "
                     "Run on the box (or export the R2 env).")
        self.bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
        self.s3 = boto3.client(
            "s3", endpoint_url=os.environ.get("R2_ENDPOINT"),
            aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))

    def write(self, key: str, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body,
                           ContentType="application/json",
                           CacheControl="public, max-age=60")


def _sink():
    return _R2Json()


def publish(tracks: list[dict[str, Any]], sink=None) -> list[str]:
    sink = sink or _sink()
    keys = []
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    index = {"generated_utc": now.isoformat().replace("+00:00", "Z"),
             "method": "ARCHER-style IR center fix — the explorer's objfix.js "
                       "run headlessly; identical code, no reimplementation",
             "disclosure": "AUTOMATED OBJECTIVE SATELLITE ESTIMATE — "
                           "experimental, not official. See NHC/JTWC.",
             "storms": []}
    for tr in tracks:
        st = tr.get("_storm") or {}
        sid = st.get("id") or st.get("name") or "unknown"
        key = f"{R2_PREFIX}/{sid}.json"
        body = dict(tr)
        body.pop("_storm", None)
        body["published_utc"] = index["generated_utc"]
        body["storm_feed"] = st
        sink.write(key, body)
        keys.append(key)
        pts = body.get("points") or []
        index["storms"].append({
            "id": sid, "name": st.get("name"), "basin": st.get("basin"),
            "slug": st.get("slug"), "key": key,
            "frames": len(pts),
            "fixes": sum(1 for p in pts if p.get("fix")),
            "newest_utc": pts[-1]["t"] if pts else None,
        })
    sink.write(f"{R2_PREFIX}/index.json", index)
    keys.append(f"{R2_PREFIX}/index.json")
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--storm", help="only storms matching this substring")
    ap.add_argument("--single", action="store_true",
                    help="single newest frame instead of the loop workup")
    ap.add_argument("--no-publish", action="store_true",
                    help="print the tracks instead of writing to R2")
    ap.add_argument("--out", help="also write the tracks to this local path")
    a = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")

    tracks = _collect(storm_filter=a.storm, loop=not a.single)
    if not tracks:
        # Genuine off-season / no-storm is a clean no-op, not a failure: the
        # lane must not page anyone because the Pacific is quiet.
        log.info("no storms produced fixes — nothing to publish")
        if not a.no_publish:
            # Still refresh the index so consumers can tell "quiet basin"
            # (an index with zero storms, freshly stamped) from "the lane is
            # dead" (a stale index). Silence must not look like data.
            publish([])
        return 0
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(tracks, fh, indent=1)
        log.info("wrote %s", a.out)
    if a.no_publish:
        json.dump(tracks, sys.stdout, indent=1)
        return 0
    keys = publish(tracks)
    log.info("published %d key(s): %s", len(keys), ", ".join(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
