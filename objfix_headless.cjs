// Headless collector for the browser ARCHER/ADT fixes.
//
// Boots the REAL explorer page in headless Chromium and drives the objfix
// panel through its existing programmatic seam (window.ObjFixPanel), then
// prints the same trackJSON() payload the panel's download button produces —
// one JSON array on stdout. The Python wrapper (objfix_headless.py) publishes
// it to R2.
//
// This file contains NONE of the method. It clicks the buttons. The ARCHER /
// ADT numerics live only in satellite/explorer/objfix.js, which is why there
// is no second implementation to keep in sync — and why the per-frame
// first-guess anchoring (official track, never chained fixes) is preserved:
// that lives in the panel's runAnalysis, which is what we drive.
//
//   node objfix_headless.cjs [--storm <substr>] [--single] [--site <url>]
//
// stdout: JSON array of tracks. stderr: progress/diagnostics.
"use strict";
const { chromium } = require("playwright");

function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}
const HAS = (name) => process.argv.includes("--" + name);

const SITE = arg("site", process.env.OBJFIX_SITE || "https://triple-a-tropics.com");
const PATH_ = arg("path", process.env.OBJFIX_EXPLORER_PATH || "/satellite/explorer/");
const FILTER = arg("storm", null);
const LOOP = !HAS("single");
const RUN_TIMEOUT = Number(process.env.OBJFIX_RUN_TIMEOUT_S || 900) * 1000;
const PAGE_TIMEOUT = Number(process.env.OBJFIX_PAGE_TIMEOUT_S || 120) * 1000;

const log = (...a) => console.error("[objfix]", ...a);

(async () => {
  const browser = await chromium.launch({
    args: ["--no-sandbox", "--enable-unsafe-swiftshader",
           "--use-gl=angle", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e).slice(0, 300)));

  const out = [];
  try {
    const url = SITE.replace(/\/$/, "") + PATH_;
    log("loading", url);
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: PAGE_TIMEOUT });
    await page.waitForFunction(() => !!window.ObjFixPanel, { timeout: PAGE_TIMEOUT });
    await page.evaluate(() => window.ObjFixPanel.open());
    await page.evaluate(() => window.ObjFixPanel.loadStorms());
    await page.waitForFunction(
      () => window.ObjFixPanel.storms() && window.ObjFixPanel.storms().length > 0,
      { timeout: PAGE_TIMEOUT }).catch(() => {});
    const storms = await page.evaluate(() => window.ObjFixPanel.storms());
    log("explorer lists", storms.length, "storm(s)");

    for (let i = 0; i < storms.length; i++) {
      const st = storms[i];
      const label = `${st.name} (${st.id || st.slug})`;
      if (FILTER && !JSON.stringify(st).toLowerCase().includes(FILTER.toLowerCase())) continue;
      if (st.lat == null) {
        // runAnalysis refuses without a first guess; say so rather than
        // silently publishing nothing for this storm.
        log(label, "- no first-guess position in the feed, skipped");
        continue;
      }
      log("analyzing", label, LOOP ? "(loop)" : "(single frame)");
      await page.evaluate((n) => window.ObjFixPanel.select(n), i);
      await page.evaluate((l) => window.ObjFixPanel.analyze(l), LOOP);
      let truncated = false;
      try {
        await page.waitForFunction(() => window.ObjFixPanel.running() === false,
                                   { timeout: RUN_TIMEOUT });
      } catch (e) {
        // A long loop can outrun the budget. Publishing what completed is
        // fine; publishing it as if it were the WHOLE recent track is not —
        // the newest analyzed frame is then older than the newest available
        // one, and a plot would date itself off a stale fix without saying so.
        truncated = true;
        log(label, "- run did not finish in time; stopping and taking what completed");
        await page.evaluate(() => window.ObjFixPanel.stop());
        await page.waitForTimeout(2000);
      }
      const key = st.id || st.name;
      const track = await page.evaluate(
        (k) => (window.ObjFix && window.ObjFix.tracks) ? window.ObjFix.tracks[k] : null,
        key);
      if (!track || !(track.points || []).length) {
        log(label, "- no fixes produced, nothing published");
        continue;
      }
      const fixes = track.points.filter((p) => p.fix).length;
      log(label, `- ${track.points.length} frame(s), ${fixes} accepted fix(es)`);
      track._storm = st;
      track.truncated = truncated;

      // SATCON is an INTENSITY consensus, not a center fix — it produces no
      // position and never appears as a centre marker. Captured here so the
      // plot header can show it beside the ADT intensity. It legitimately
      // returns null when its own membership rule is unmet (>= 2 coincident
      // members), and that stays null: no silent fallback to bare ADT.
      track.satcon = await page.evaluate(async (storm) => {
        if (!window.SatCon || !window.SatCon.latest) return null;
        try {
          window.SatCon.setStorm(storm);
          window.SatCon.update(window.ObjFixPanel.results());
          // give the MW-overpass fetch a moment; the consensus is ADT-only
          // (i.e. null) until a second member lands
          await new Promise((r) => setTimeout(r, 6000));
          window.SatCon.update(window.ObjFixPanel.results());
          const l = window.SatCon.latest();
          return l ? { t: l.t, vmax: l.vmax, mslp: l.mslp,
                       adt: l.adt ? { vmax: l.adt.vmax, mslp: l.adt.mslp,
                                      scene: l.adt.scene } : null,
                       state: window.SatCon.state() } : null;
        } catch (e) { return { error: String(e).slice(0, 200) }; }
      }, st);
      if (track.satcon && track.satcon.vmax) {
        log(label, "- SATCON vmax", JSON.stringify(track.satcon.vmax));
      } else {
        log(label, "- SATCON: no consensus (membership rule unmet)");
      }
      out.push(track);
    }
  } finally {
    if (errors.length) log("page errors:", errors.slice(0, 5).join(" | "));
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { log("FATAL", e && e.message); process.exit(1); });
