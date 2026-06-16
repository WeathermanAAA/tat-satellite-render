"""Generate the graceful-degrade review states: a FRESH INVEST (statistical-only,
empty track_aids) and SHIPS available:false. Real-shaped mock JSON, not live."""
from pathlib import Path
from cyclolab_basemap import basemap_for
import cyclolab_guidance_review as R

HERE = Path("/tmp/tsr/_review_out"); HERE.mkdir(exist_ok=True)

# fresh invest: only statistical intensity aids; NO dynamical track aids
def ramp(v0, dv, n=11):
    return [{"tau": i * 6, "lat": None, "lon": None, "vmax": max(15, v0 + dv * i), "mslp": None}
            for i in range(n)]
fresh = {
    "init_time": "2026-06-16T06:00:00Z", "init_cycle": "2026061606",
    "aids": {"DSHP": ramp(25, 2), "LGEM": ramp(25, 1), "SHIP": ramp(25, 2)},
    "present_aids": ["DSHP", "LGEM", "SHIP"], "track_aids": [],
    "intensity_aids": ["DSHP", "LGEM", "SHIP"], "consensus": [],
    "sid": "NHC_AL952026", "basin": "AL",
}
ships_unavail = {"available": False, "reason": "unavailable", "sid": "NHC_AL952026"}
bm = basemap_for(14.0, -42.0, "AL")
html = R.build_page("NHC_AL952026", "95L", fresh, ships_unavail, bm)
(HERE / "review_FRESH_INVEST.html").write_text(html)
print("wrote review_FRESH_INVEST.html (track_aids=[], ships available:false)")
