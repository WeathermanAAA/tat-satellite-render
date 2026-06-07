"""CycloLab per-storm page writer - key contract (Stage 0/2).

THE EDGE CONTRACT (locked against the deployed cyclolab-router Worker,
version 8b77a818, Stage-0 gate-verified 2026-06-06): the Worker serves
``triple-a-tropics.com/cyclolab/{sid}/`` from bucket
``triple-a-tropics-media`` at key ``cyclolab/{sid}/index.html`` with the
stored Content-Type. The writer side therefore MUST produce exactly:

    key          {live_prefix}/{sid}/index.html
    content-type text/html; charset=utf-8   (R2Sink.write_html)
    bucket       R2_BUCKET env, default triple-a-tropics-media

Shadow-first discipline composes cleanly: shadow pages live under
``shadow/cyclolab/...`` which the Worker CANNOT serve (it only reads
``cyclolab/...``) - shadow content is unreachable from the edge by
construction, and promotion is just writing under the live prefix.

tests/test_cyclolab_pages.py pins the key shapes; the main repo's
tests/test_cyclolab_router.py pins the Worker's resolve() to the SAME
strings - the two suites together are the cross-repo contract.
"""
from __future__ import annotations

from cyclolab_intensity import basin_entry
from storm_ids import StormIds, parse_sid

LIVE_PREFIX = "cyclolab"          # what the deployed Worker reads


def page_key(sid: str, prefix: str = LIVE_PREFIX) -> str:
    """R2 key for a storm's shell page. Validates the sid (designated
    storms only - parse_sid raises on invests/malformed)."""
    ids: StormIds = parse_sid(sid)
    return f"{prefix}/{ids.sid}/index.html"


def adv_key(sid: str, prefix: str = LIVE_PREFIX) -> str:
    """R2 key for a storm's cached advisory JSON (the §8.3 contract the
    shell hydrates from; same shape the cyclolab-adv Source writes)."""
    ids: StormIds = parse_sid(sid)
    return f"{prefix}/adv/{ids.sid}.json"


def page_url_path(sid: str) -> str:
    """The public path the Worker serves for a storm (deep-link target +
    og:url value)."""
    ids: StormIds = parse_sid(sid)
    return f"/cyclolab/{ids.sid}/"


# ---------------------------------------------------------------------------
# The per-storm page LIFECYCLE writer (CYCLOLAB_DESIGN.md §3.1/§3.4).
# Mirrors the GlobalGeojsonComposer pattern: one shared instance, fed by
# every basin source AFTER its feeds are safely written; best-effort by
# contract (a page-write blip never fails or stales a basin's feeds).
# ---------------------------------------------------------------------------
import logging as _logging
import os as _os

_log = _logging.getLogger("cyclolab-pages")

PAGES_ENABLED = (_os.environ.get("CYCLOLAB_PAGES") or "1").lower() \
    not in ("0", "false", "no")
# Dissipation debounce: a storm must be absent/inactive this many
# consecutive polls before its page freezes (the system's standard
# transient guard - one flaky poll must not bury a live storm).
ENDED_DEBOUNCE_POLLS = 2


class CycloLabPageWriter:
    """Owns the /cyclolab/{sid}/ page lifecycle:

    BIRTH   first time a designated storm appears active in a basin feed
            -> render + write the live shell.
    REFRESH category change or a new fix -> rewrite (cheap: the page is
            ~26 KB; the baked snapshot + OG intensity stay current).
    ENDED   storm leaves the active set for ENDED_DEBOUNCE_POLLS polls
            -> write the frozen archive page ONCE (last snapshot baked,
            polling disabled); the key stays live so links never 404.

    Shadow-first: prefix defaults to the advisories Source's
    CYCLOLAB_PREFIX (one promote knob for pages + advisories together).
    """

    def __init__(self, sink, *, prefix: str | None = None,
                 feed_base: str = "https://cdn.triple-a-tropics.com/feeds",
                 cdn_base: str = "https://cdn.triple-a-tropics.com"):
        from cyclolab_advisories import CYCLOLAB_PREFIX
        self.sink = sink
        self.prefix = (prefix if prefix is not None else CYCLOLAB_PREFIX).rstrip("/")
        self.feed_base = feed_base.rstrip("/")
        self.cdn_base = cdn_base.rstrip("/")
        # sid -> {"cat": str, "fix": str|None, "missing": int,
        #         "ended": bool, "last": dict (snapshot)}
        self._state: dict[str, dict] = {}
        # final-gate-2 #1: the storm-centered SST hero layers ride the
        # same per-fix cadence (best-effort; its own kill switch).
        try:
            from cyclolab_sst import SstHeroWriter
            self._sst = SstHeroWriter(sink, prefix=self.prefix)
        except Exception as e:  # noqa: BLE001 - never block pages
            _log.warning("sst hero writer unavailable: %s", e)
            self._sst = None

    def _adv_url(self, sid: str) -> str:
        # Live prefix -> same-origin relative path through the Worker;
        # shadow -> absolute cdn URL (the Worker never serves shadow).
        if self.prefix == LIVE_PREFIX:
            return f"/cyclolab/adv/{sid}.json"
        return f"{self.cdn_base}/{adv_key(sid, prefix=self.prefix)}"

    def _sst_base(self, sid: str) -> str:
        # Same live/shadow convention as _adv_url (the Worker serves any
        # file-ish key under cyclolab/; shadow keys go absolute-CDN).
        if self.prefix == LIVE_PREFIX:
            return f"/cyclolab/{sid}/sst"
        return f"{self.cdn_base}/{self.prefix}/{sid}/sst"

    def _og_url(self, sid: str) -> str | None:
        """The intensity OG card URL - only when the storm's basin has a
        published-error registry entry (the honesty guard: no statistics,
        no card, no og:image tag)."""
        try:
            if basin_entry(parse_sid(sid).basin) is None:
                return None
        except Exception:  # noqa: BLE001 - unparseable sid -> no card
            return None
        return f"{self.cdn_base}/{self.prefix}/og/{sid}.png"

    def update(self, basin: str, tracks_feed: dict, now=None) -> None:
        """Feed one basin's freshly-written tracks feed through the
        lifecycle. NEVER raises (best-effort contract)."""
        try:
            self._update(basin, tracks_feed)
        except Exception as e:  # noqa: BLE001
            _log.warning("page lifecycle update failed (%s): %s", basin, e)

    def _update(self, basin: str, tracks_feed: dict) -> None:
        from cyclolab_shell import render_page
        feed_url = f"{self.feed_base}/{basin}_tracks_data.json"
        seen_active: set[str] = set()
        for storm in tracks_feed.get("storms", []) or []:
            sid = storm.get("sid") or ""
            try:
                parse_sid(sid)          # designated storms only (V1)
            except Exception:           # noqa: BLE001 - invests/malformed
                continue
            if not storm.get("is_active"):
                continue
            seen_active.add(sid)
            st = self._state.get(sid)
            cat = storm.get("current_category") or "TD"
            fix = storm.get("latest_fix_valid_utc")
            # SST hero layers ride EVERY poll, not just page rewrites:
            # maybe_render self-gates cheaply (state + TTL'd probe) and
            # that lets a partial family (SSTA lagging SST) heal between
            # fixes (adversarial-review find). Best-effort by contract.
            if self._sst is not None:
                self._sst.maybe_render(sid, storm, basin)
            if st is None or st["ended"] or st["cat"] != cat or st["fix"] != fix:
                html = render_page(storm, feed_url=feed_url,
                                   adv_url=self._adv_url(sid),
                                   og_image_url=self._og_url(sid),
                                   sst_base=self._sst_base(sid))
                self.sink.write_html(page_key(sid, prefix=self.prefix), html)
                _log.info("cyclolab page %s: %s (%s, fix %s)",
                          "BIRTH" if st is None else "refresh", sid, cat, fix)
            self._state[sid] = {"cat": cat, "fix": fix, "missing": 0,
                                "ended": False, "last": storm}
        # dissipation sweep: previously-known storms of THIS basin's feed
        # that are no longer active (absent or is_active False).
        for sid, st in list(self._state.items()):
            if sid in seen_active or st["ended"]:
                continue
            # only sweep sids whose basin matches this feed (the writer is
            # shared across basin sources).
            if parse_sid(sid).basin.lower() != (tracks_feed.get("basin") or "").lower():
                continue
            st["missing"] += 1
            if st["missing"] >= ENDED_DEBOUNCE_POLLS:
                from cyclolab_shell import render_page as _rp
                html = _rp(st["last"], feed_url=feed_url,
                           adv_url=self._adv_url(sid), ended=True,
                           sst_base=self._sst_base(sid))
                self.sink.write_html(page_key(sid, prefix=self.prefix), html)
                st["ended"] = True
                _log.info("cyclolab page ENDED (frozen): %s", sid)
