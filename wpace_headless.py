"""Capture the CycloLab wind/pressure + ACE chart and publish it as a PNG.

WHY THIS EXISTS
    The storm diagnostic plate (centerfix_plot.render_composite) wants that
    chart as its bottom-left panel. The chart itself is a browser renderer
    inside cyclolab_shell.py, and its ACE method is not trivial: forecast
    intensity is interpolated onto the 6-hourly synoptic grid before summing
    (issued taus are 12/24/36/48/72/96/120 h, so summing them directly would
    skip the points between and weight a 24 h gap like a 12 h one), only
    >= 34 kt points count, and the projection RESUMES from the latest observed
    fix rather than the advisory t0 because the b-deck runs ahead of the
    advisory.

    Porting that into matplotlib would be a second implementation of a method
    with real subtleties, and both copies would keep rendering plausible curves
    while drifting apart. So we photograph the one that exists — exactly the
    objfix_headless.py argument, and the same python-wrapper / node-harness
    split.

OUTPUT
    r2://<bucket>/cyclolab/wpace/<storm_id>.png

    Storms are taken from the objfix index, because that is precisely the set
    the plate is rendered for. A storm whose chart does not render is skipped,
    never published blank: the plate has an honest placeholder for that, and a
    blank chart would read as "no ACE".

USAGE
    python wpace_headless.py                  # every storm in the objfix index
    python wpace_headless.py --storm 12W      # one storm (id substring)
    python wpace_headless.py --no-publish --out-dir /tmp/wpace
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

log = logging.getLogger("wpace-headless")

CDN = os.environ.get("TAT_CDN", "https://cdn.triple-a-tropics.com")
SITE = os.environ.get("WPACE_SITE",
                      os.environ.get("OBJFIX_SITE",
                                     "https://triple-a-tropics.com"))
OBJFIX_R2_PREFIX = os.environ.get("OBJFIX_R2_PREFIX", "cyclolab/objfix")
R2_PREFIX = os.environ.get("WPACE_R2_PREFIX", "cyclolab/wpace")
RUN_TIMEOUT_S = int(os.environ.get("WPACE_RUN_TIMEOUT_S", "600"))

CAPTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "wpace_headless.cjs")


def storm_ids(storm_filter: Optional[str] = None) -> list[str]:
    """The storms the plate will be rendered for, from the objfix index."""
    import urllib.request
    url = f"{CDN}/{OBJFIX_R2_PREFIX}/index.json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "tat-wpace/1.0 (+https://triple-a-tropics.com)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            idx = json.loads(r.read().decode("utf-8"))
    except Exception as e:      # noqa: BLE001 - no index is nothing to capture
        log.warning("objfix index unavailable (%s) — nothing to capture", e)
        return []
    ids = []
    for s in (idx.get("storms") or []):
        sid = s.get("id")
        if not sid:
            continue
        if storm_filter and storm_filter.lower() not in json.dumps(s).lower():
            continue
        ids.append(sid)
    return ids


def capture(ids: list[str], out_dir: str) -> list[dict]:
    """Drive the browser. Python owns arguments and publishing; node owns the
    page — the same split objfix_headless.py uses."""
    if not ids:
        return []
    cmd = ["node", CAPTURE, "--site", SITE, "--out", out_dir,
           "--storms", ",".join(ids)]
    log.info("capture: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=RUN_TIMEOUT_S)
    for line in (proc.stderr or "").splitlines():
        log.info("%s", line)
    if proc.returncode != 0:
        raise RuntimeError(f"wpace capture failed (rc={proc.returncode})")
    body = (proc.stdout or "").strip()
    return json.loads(body) if body else []


class _R2Png:
    """Minimal R2 PNG writer.

    Deliberately NOT intensity_poller.R2Sink: importing that drags pandas +
    ace_core + the whole feed stack into a lane whose only job is to drive a
    browser and PUT a few PNGs. Mirrors objfix_headless._R2Json.
    """

    def __init__(self) -> None:
        import boto3
        from botocore.config import Config as BotoConfig
        self.bucket = os.environ.get("R2_BUCKET", "triple-a-tropics-media")
        self.s3 = boto3.client(
            "s3", endpoint_url=os.environ.get("R2_ENDPOINT"),
            aws_access_key_id=(os.environ.get("R2_ACCESS_KEY_ID")
                               or os.environ.get("AWS_ACCESS_KEY_ID")),
            aws_secret_access_key=(os.environ.get("R2_SECRET_ACCESS_KEY")
                                   or os.environ.get("AWS_SECRET_ACCESS_KEY")),
            config=BotoConfig(retries={"max_attempts": 3, "mode": "standard"}))

    def write_png(self, key: str, data: bytes) -> None:
        # max-age matches the chart's own refresh cadence: the plate reads this
        # object every render, and a long TTL would paste a stale chart beside
        # fresh imagery under one VALID stamp.
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                           ContentType="image/png",
                           CacheControl="public, max-age=120")


def publish(shots: list[dict], sink=None) -> list[str]:
    sink = sink or _R2Png()
    keys = []
    for s in shots:
        with open(s["file"], "rb") as fh:
            data = fh.read()
        key = f"{R2_PREFIX}/{s['id']}.png"
        sink.write_png(key, data)
        keys.append(key)
        log.info("%s -> %s (%d bytes, %sx%s)", s["id"], key, len(data),
                 s.get("w"), s.get("h"))
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--storm", help="only storms matching this substring")
    ap.add_argument("--no-publish", action="store_true",
                    help="capture only; do not write to R2")
    ap.add_argument("--out-dir", help="keep the PNGs here instead of a tempdir")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ids = storm_ids(a.storm)
    if not ids:
        log.info("no storms to capture")
        return 0
    log.info("capturing %d storm(s): %s", len(ids), ", ".join(ids))
    tmp = None
    out_dir = a.out_dir
    if not out_dir:
        tmp = tempfile.TemporaryDirectory(prefix="wpace-")
        out_dir = tmp.name
    else:
        os.makedirs(out_dir, exist_ok=True)
    try:
        shots = capture(ids, out_dir)
        missed = [i for i in ids if i not in {s["id"] for s in shots}]
        if missed:
            # Never silent: a storm that did not produce a chart is a fact the
            # plate will show as a placeholder, so say which ones here.
            log.info("no chart for %s (the plate shows its placeholder)",
                     ", ".join(missed))
        if a.no_publish:
            log.info("captured %d chart(s) into %s (not published)",
                     len(shots), out_dir)
            return 0
        publish(shots)
        log.info("published %d chart(s)", len(shots))
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
