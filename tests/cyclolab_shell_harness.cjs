// Node harness for the CycloLab per-storm shell (cyclolab_shell.render_page).
//
// Loads the rendered page into jsdom (runScripts:'dangerously') so the page's
// REAL inline hydration script runs, then executes a plan of deterministic
// ops against the script's exposed hooks (window.__lab.openSec / setCategory /
// apply / odoSet) and prints a JSON state snapshot after every op. The python
// wrapper (tests/test_cyclolab_shell.py) drives this via subprocess and
// asserts on the snapshots.
//
//   node cyclolab_shell_harness.cjs <page.html> <plan.json>
//
// plan.json: an array of ops. Each op is one of:
//   {"op":"snapshot"}                       -> snapshot only
//   {"op":"openSec","name":"models"}        -> window.__lab.openSec(name)
//   {"op":"setCategory","cat":"C5"}         -> window.__lab.setCategory(cat)
//   {"op":"apply","storm":{...}}            -> window.__lab.apply(storm)
// Every op is followed by a snapshot in the output stream; the harness prints
// one JSON object PER LINE (the op echoed + its resulting state).
//
// jsdom shims (per the shell's real DOM usage):
//   * window.matchMedia      -> reduced-motion ALWAYS false (deterministic
//                               full-motion path; never throws)
//   * window.fetch           -> resolves the fixture feed embedded in the plan
//                               (plan.feed), so poll() finds the storm; absent
//                               feed -> a never-ok response (baked snapshot
//                               stands), so the test owns whether a poll fires
//   * requestAnimationFrame  -> immediate (jsdom has one under pretendToBeVisual,
//                               re-stubbed here so timing is deterministic)
//   * SVGElement.getTotalLength -> 2000 (jsdom lacks it; the chart draw-in
//                               reads it)
//   * setTimeout(...,60000)  -> RECORDED (the 60s poll re-arm) but NOT fired,
//                               so the harness can report whether the live page
//                               scheduled the next poll without looping forever.
"use strict";

const fs = require("fs");
const { JSDOM } = require("jsdom");

const PAGE = fs.readFileSync(process.argv[2], "utf8");
const PLAN = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const OPS = Array.isArray(PLAN) ? PLAN : (PLAN.ops || []);
const FEED = Array.isArray(PLAN) ? null : (PLAN.feed || null);

// ---- timer bookkeeping: record every scheduled timeout's delay -------------
// The page re-arms its poll with setTimeout(poll, 60000) ONLY after the first
// fetch resolves and ONLY when not ENDED. We record delays so a test can prove
// "no 60000ms timer on the ended page" / "a 60000ms timer on the live page".
const scheduledDelays = [];

(async () => {
  const dom = new JSDOM(PAGE, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "https://triple-a-tropics.com/cyclolab/NHC_EP082026/",
    beforeParse(window) {
      // matchMedia: reduced-motion false (full-motion, deterministic).
      window.matchMedia = function (q) {
        return {
          matches: false,
          media: String(q),
          addListener() {},
          removeListener() {},
          addEventListener() {},
          removeEventListener() {},
          onchange: null,
          dispatchEvent() { return false; },
        };
      };

      // requestAnimationFrame: fire on the next macrotask, deterministically.
      window.requestAnimationFrame = function (cb) {
        return window.setTimeout(function () { cb(Date.now()); }, 0);
      };
      window.cancelAnimationFrame = function (id) { window.clearTimeout(id); };

      // getTotalLength: jsdom models SVG <path> as a bare SVGElement with no
      // geometry methods; the chart draw-in calls series.getTotalLength().
      // Stub it on every SVG prototype the element might inherit from.
      for (const ctor of ["SVGElement", "SVGGraphicsElement", "SVGGeometryElement",
                          "SVGPathElement"]) {
        const C = window[ctor];
        if (C && C.prototype && !C.prototype.getTotalLength) {
          C.prototype.getTotalLength = function () { return 2000; };
        }
      }

      // fetch: resolve the plan's embedded feed (the storms array the page's
      // poll() filters by sid). No feed -> a !ok response, so the baked
      // snapshot stands and no live apply() fires from the poll.
      window.fetch = function () {
        if (FEED == null) {
          return Promise.resolve({
            ok: false,
            json() { return Promise.reject(new Error("no feed")); },
          });
        }
        return Promise.resolve({
          ok: true,
          json() { return Promise.resolve(FEED); },
        });
      };

      // Record-and-suppress the 60s poll re-arm; let everything else (the
      // 0ms rAF/microtask drains, animation restarts) run normally.
      const realSetTimeout = window.setTimeout;
      window.setTimeout = function (fn, delay) {
        scheduledDelays.push(delay);
        if (delay && delay >= 60000) {
          // the poll re-arm: record only, never fire (no infinite loop).
          return 0;
        }
        return realSetTimeout(fn, delay, ...Array.prototype.slice.call(arguments, 2));
      };
    },
  });

  const window = dom.window;
  const document = window.document;

  // Let the page's own boot (buildStats + baked apply + first poll) settle.
  await drain();

  const out = [];

  function drain() {
    // a couple of macrotasks flush the fetch().then() microtask chains and
    // the 0ms rAF/animation-restart timers the shims schedule.
    return new Promise((resolve) => setTimeout(resolve, 30));
  }

  function odo(id) {
    const el = document.getElementById("odo-" + id);
    return el ? el.getAttribute("data-odo") : null;
  }

  function odoCols(id) {
    // the per-digit column transforms (translateY) - proves the roll moved.
    const el = document.getElementById("odo-" + id);
    if (!el) return [];
    return Array.prototype.map.call(el.querySelectorAll(".col"),
      (c) => c.style.transform || "");
  }

  function endedStripVisible() {
    const el = document.querySelector(".ended-strip");
    if (!el) return false;
    let disp = "";
    try { disp = window.getComputedStyle(el).display; } catch (e) { disp = ""; }
    if (disp) return disp !== "none";
    // fallback (computed style unreliable): the attribute that drives it.
    return document.documentElement.hasAttribute("data-ended");
  }

  function activeSection() {
    const s = document.querySelector(".sec.active");
    return s ? s.id : null;
  }

  function activeNav() {
    const b = document.querySelector(".sec-btn.active");
    return b ? b.getAttribute("data-sec") : null;
  }

  function bannerClasses() {
    const b = document.getElementById("banner");
    return b ? Array.prototype.slice.call(b.classList) : [];
  }

  function text(id) {
    const el = document.getElementById(id);
    return el ? el.textContent : null;
  }

  function snapshot() {
    return {
      cat: document.documentElement.getAttribute("data-cat"),
      ended: document.documentElement.hasAttribute("data-ended"),
      chip: text("chip"),
      stormName: text("storm-name"),
      activeSection: activeSection(),
      activeNav: activeNav(),
      odo: {
        vmax: odo("vmax"), mslp: odo("mslp"), ace: odo("ace"),
        pos: odo("pos"), fix: odo("fix"), cat: odo("cat"),
      },
      odoColsVmax: odoCols("vmax"),
      odoAriaVmax: (function () {
        const el = document.getElementById("odo-vmax");
        return el ? el.getAttribute("aria-label") : null;
      })(),
      // AD R2 integrated card: the canon label on the corner glyph
      // (+ the Category hero rides odo.cat above).
      glyphCat: text("glyph-cat"),
      heroCatText: (function () {
        const el = document.getElementById("odo-cat");
        return el ? el.textContent : null;
      })(),
      bannerClasses: bannerClasses(),
      trackmapChildCount: (document.getElementById("trackmap") || { children: [] })
        .children.length,
      chartChildCount: (document.getElementById("chart") || { children: [] })
        .children.length,
      endedStripVisible: endedStripVisible(),
      scheduledDelays: scheduledDelays.slice(),
    };
  }

  for (const op of OPS) {
    if (op.op === "snapshot") {
      // no-op
    } else if (op.op === "openSec") {
      window.__lab.openSec(op.name);
    } else if (op.op === "setCategory") {
      window.__lab.setCategory(op.cat);
    } else if (op.op === "apply") {
      window.__lab.apply(op.storm);
    } else if (op.op === "removeBannerClasses") {
      // test helper: clear banner anim classes to detect re-add / churn.
      const b = document.getElementById("banner");
      if (b) { b.classList.remove("shine"); b.classList.remove("xfade"); }
    } else {
      out.push({ op: op.op, error: "unknown op" });
      continue;
    }
    await drain();
    out.push({ op: op.op, state: snapshot() });
  }

  process.stdout.write(out.map((o) => JSON.stringify(o)).join("\n") + "\n");
  process.exit(0);
})().catch((e) => {
  process.stderr.write(String(e && e.stack || e) + "\n");
  process.exit(1);
});
