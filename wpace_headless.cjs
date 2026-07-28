// Headless capture of the CycloLab wind/pressure + ACE chart.
//
// Loads the REAL per-storm CycloLab page and screenshots its existing
// <svg id="chart"> — the two-panel intensity diagnostic (observed + official
// forecast wind, pressure, cumulative observed ACE + projected ACE).
//
// This file contains NONE of the method. It takes a picture. The ACE rules —
// the 6-hourly regrid of forecast intensity, the ">= 34 kt counts" gate, the
// resume-from-the-latest-observed-fix projection — live only in the chart
// renderer inside cyclolab_shell.py, which is why there is no second
// implementation to keep in sync. Same reasoning as objfix_headless.cjs.
//
//   node wpace_headless.cjs --storms ID1,ID2 --out /tmp/wpace [--site url]
//
// stdout: JSON array of {id, file, w, h} for the storms that produced a chart.
// stderr: progress/diagnostics.
//
// A storm whose chart does not render (fewer than two usable fixes — the
// renderer bails and leaves the SVG empty) is REPORTED AS SKIPPED, never
// captured. A blank PNG pasted into the plate would read as "no ACE", which
// is a claim; the plate's own placeholder says the truth instead.
"use strict";
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

function arg(name, dflt) {
  const i = process.argv.indexOf("--" + name);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const SITE = arg("site", process.env.WPACE_SITE ||
                 process.env.OBJFIX_SITE || "https://triple-a-tropics.com");
const OUT = arg("out", "/tmp/wpace");
const STORMS = (arg("storms", "") || "").split(",").map(s => s.trim())
  .filter(Boolean);
const PAGE_TIMEOUT = Number(process.env.WPACE_PAGE_TIMEOUT_S || 90) * 1000;
const CHART_TIMEOUT = Number(process.env.WPACE_CHART_TIMEOUT_S || 45) * 1000;
// The plate's own background. Forcing it here rather than cropping around the
// card keeps the pasted raster from showing a seam against the panel.
const BG = process.env.WPACE_BG || "#0a1019";

const log = (...a) => console.error("[wpace]", ...a);

(async () => {
  if (!STORMS.length) { process.stdout.write("[]"); return; }
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ args: ["--no-sandbox"] });
  // Wide viewport so the chart card is laid out at full width; DSF 2 so the
  // capture out-resolves the plate cell it lands in rather than being upscaled.
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 2 });
  const out = [];
  try {
    for (const sid of STORMS) {
      const url = SITE.replace(/\/$/, "") + "/cyclolab/" + sid + "/";
      try {
        log("loading", url);
        await page.goto(url, { waitUntil: "domcontentloaded",
                               timeout: PAGE_TIMEOUT });
        // The renderer bails to an EMPTY svg when the storm has fewer than two
        // usable fixes, so "has children" is the real ready signal — waiting on
        // the element alone would capture the blank frame every time.
        await page.waitForFunction(() => {
          const s = document.getElementById("chart");
          return !!s && s.children.length > 8;
        }, { timeout: CHART_TIMEOUT });
        await page.evaluate((bg) => {
          const s = document.getElementById("chart");
          s.style.background = bg;
          s.style.display = "block";
        }, BG);
        // Let webfonts settle so labels are not captured mid-swap.
        await page.waitForTimeout(600);
        const el = await page.$("#chart");
        const box = await el.boundingBox();
        const file = path.join(OUT, sid + ".png");
        await el.screenshot({ path: file });
        out.push({ id: sid, file: file,
                   w: Math.round(box ? box.width : 0),
                   h: Math.round(box ? box.height : 0) });
        log("captured", sid, box ? `${Math.round(box.width)}x${Math.round(box.height)}` : "");
      } catch (e) {
        // PER-STORM ISOLATION: one storm without a chart must not cost the
        // others their capture.
        log("skipped", sid, "-", String(e).split("\n")[0].slice(0, 160));
      }
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { log("fatal", e); process.exit(1); });
