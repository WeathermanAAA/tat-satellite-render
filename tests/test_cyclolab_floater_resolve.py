"""CycloLab Satellite-tab floater resolution across agencies.

Regression for: the Satellite tab on a JTWC/WPAC storm page (e.g. Mekkhala /
JTWC_WP072026) was permanently empty. The tab resolves its live floater by
matching the baked atcf_long against the floater index's per-storm id, but the
index keys NHC storms by the bare atcf_long ("ep012026") while it keys JTWC/WP
named storms by the agency-prefixed sid ("JTWC_WP072026"). The bare-equality
match therefore never matched a WPAC storm -> no manifest URL -> empty tab.

This test exercises the REAL emitted artifact: it renders a page, extracts the
baked FLOATER_ID / FLOATER_SLUG and the actual SAT_SOURCES[0].resolve function,
and runs that resolve in node against synthetic floater indices -- so it guards
BOTH the bake wiring and the matching logic (no re-implementation to drift).

Run: python -m pytest tests/test_cyclolab_floater_resolve.py -q
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import cyclolab_shell  # noqa: E402

FIXTURE = HERE / "fixtures" / "cyclolab" / "synth_storm.json"
FEED_URL = "https://cdn.triple-a-tropics.com/feeds/wp_tracks_data.json"
NODE = shutil.which("node")


def _render(sid: str, name: str) -> str:
    storm = json.loads(FIXTURE.read_text(encoding="utf-8"))
    storm["sid"] = sid
    storm["name"] = name
    return cyclolab_shell.render_page(storm, feed_url=FEED_URL)


def _baked(html: str, var: str) -> str:
    m = re.search(r'var %s = "([^"]*)"' % var, html)
    assert m, f"{var} not baked in page"
    return m.group(1)


def _resolve(html: str, tops: list) -> list:
    """Run the page's REAL SAT_SOURCES[0].resolve against each synthetic top
    floater index, returning the resolved URL (or None) for each."""
    fid = _baked(html, "FLOATER_ID")
    fslug = _baked(html, "FLOATER_SLUG")
    m = re.search(r"var SAT_SOURCES = (\[\{.*?\}\]);", html, re.S)
    assert m, "SAT_SOURCES literal not found"
    sat_sources = m.group(1)
    js = (
        'var CDN = "https://cdn.triple-a-tropics.com";\n'
        f'var FLOATER_ID = {json.dumps(fid)};\n'
        f'var FLOATER_SLUG = {json.dumps(fslug)};\n'
        f'var SAT_SOURCES = {sat_sources};\n'
        f'var TOPS = {json.dumps(tops)};\n'
        'console.log(JSON.stringify(TOPS.map(function (t) {\n'
        '  return SAT_SOURCES[0].resolve(t);\n'
        '})));\n'
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(js)
        path = f.name
    out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip())


CDN = "https://cdn.triple-a-tropics.com"


@unittest.skipUnless(NODE, "node not on PATH")
class FloaterResolveTests(unittest.TestCase):
    def test_jtwc_named_storm_resolves(self):
        """Mekkhala / JTWC_WP072026: the agency-prefixed index id AND the slug
        must both resolve the floater (the bug: neither did -> empty tab)."""
        html = _render("JTWC_WP072026", "MEKKHALA")
        self.assertEqual(_baked(html, "FLOATER_ID"), "wp072026")
        self.assertEqual(_baked(html, "FLOATER_SLUG"), "wp07")
        wp = CDN + "/floaters/wp07/manifest.json"
        got = _resolve(html, [
            # the live index shape: prefixed id + clean slug
            {"storms": [{"id": "JTWC_WP072026", "slug": "wp07",
                         "manifest": "floaters/wp07/manifest.json"}]},
            # prefix-strip alone (no slug present) still resolves
            {"storms": [{"id": "JTWC_WP072026",
                         "manifest": "floaters/wp07/manifest.json"}]},
            # slug-only match (id drift) still resolves
            {"storms": [{"id": "something_else", "slug": "wp07",
                         "manifest": "floaters/wp07/manifest.json"}]},
            # a different storm must NOT match
            {"storms": [{"id": "al012026", "slug": "al01",
                         "manifest": "floaters/al01/manifest.json"}]},
        ])
        self.assertEqual(got, [wp, wp, wp, None])

    def test_nhc_named_storm_still_resolves(self):
        """No regression for NHC (the index keys these by the bare atcf_long)."""
        html = _render("NHC_EP082026", "SYNTH")
        self.assertEqual(_baked(html, "FLOATER_ID"), "ep082026")
        self.assertEqual(_baked(html, "FLOATER_SLUG"), "ep08")
        ep = CDN + "/floaters/ep08/manifest.json"
        got = _resolve(html, [
            {"storms": [{"id": "ep082026", "slug": "ep08",
                         "manifest": "floaters/ep08/manifest.json"}]},
        ])
        self.assertEqual(got, [ep])

    def test_invest_resolves(self):
        """WP invest (index id is the unprefixed WP912026)."""
        html = _render("JTWC_WP912026", "INVEST")
        self.assertEqual(_baked(html, "FLOATER_ID"), "wp912026")
        self.assertEqual(_baked(html, "FLOATER_SLUG"), "wp91")
        wp91 = CDN + "/floaters/wp91/manifest.json"
        got = _resolve(html, [
            {"storms": [{"id": "WP912026", "slug": "wp91",
                         "manifest": "floaters/wp91/manifest.json"}]},
        ])
        self.assertEqual(got, [wp91])


if __name__ == "__main__":
    unittest.main()
