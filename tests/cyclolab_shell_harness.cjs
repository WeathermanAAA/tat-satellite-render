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

      // fetch: URL-routed plan fixtures. Stage-3 mounts fetch the floater
      // manifests (plan.floaters = top index, plan.floater_storm = the
      // per-storm manifest); everything else keeps the legacy behavior -
      // resolve plan.feed (the storms array poll() filters by sid), or a
      // !ok response when no feed is embedded.
      const FLOATERS = Array.isArray(PLAN) ? null : (PLAN.floaters || null);
      const FLOATER_STORM = Array.isArray(PLAN) ? null
        : (PLAN.floater_storm || null);
      window.__fetched = [];
      function jsonResponse(body) {
        if (body == null) {
          return Promise.resolve({
            ok: false,
            json() { return Promise.reject(new Error("no fixture")); },
          });
        }
        return Promise.resolve({
          ok: true,
          json() { return Promise.resolve(body); },
        });
      }
      // Stage-3 models mount: a recorder stub stands in for the real
      // /models/hafs.js (jsdom has no network). withHafsViewer() sees
      // window.HafsViewer and constructs immediately; the snapshot
      // exposes what the mount passed.
      if (!Array.isArray(PLAN) && PLAN.hafs_stub) {
        window.__hafsCtor = null;
        window.HafsViewer = function (root, opts) {
          window.__hafsCtor = { root: root, opts: opts || {} };
        };
      }

      window.fetch = function (url) {
        url = String(url);
        window.__fetched.push(url);
        if (/floaters\/manifest\.json/.test(url)) {
          return jsonResponse(FLOATERS);
        }
        if (/floaters\/[^/]+\/manifest\.json/.test(url)) {
          return jsonResponse(FLOATER_STORM);
        }
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
    // the per-digit strip transforms (translateY) - proves the roll
    // moved. (AD R3: the transform rides the abspos .strip inside each
    // .col; the col itself stays put as the in-flow baseline anchor.)
    const el = document.getElementById("odo-" + id);
    if (!el) return [];
    return Array.prototype.map.call(el.querySelectorAll(".col .strip"),
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

  function stage3Probe() {
    // Stage-3 mounts: what the models mount constructed + the satellite
    // viewer's state + lazy-load evidence (script tag / fetched URLs).
    const win = dom.window;
    const ctor = win.__hafsCtor || null;
    let hafs = null;
    if (ctor) {
      const o = ctor.opts;
      hafs = {
        rootId: ctor.root ? ctor.root.id : null,
        stormLock: o.stormLock || null,
        manifestUrl: o.manifestUrl || null,
        assetBase: o.assetBase || null,
        elsKeys: o.els ? Object.keys(o.els).sort() : [],
        elsWired: (function () {
          // EVERY key identity-checked against its cl-hafs-* element (a
          // 2-key spot check let 22 silent mis-wirings through review).
          const idmap = { stage: "stage", img: "img", status: "status",
            empty: "empty", controls: "controls",
            cycleGroup: "cycle-group", cycles: "cycles",
            stormSel: "storm", models: "models", domains: "domains",
            products: "products", hours: "hours", play: "play",
            stepB: "step-back", stepF: "step-fwd", speed: "speed",
            fhour: "fhour", valid: "valid", meta: "meta", badge: "badge",
            pill: "pill", buffer: "buffer", player: "player",
            caption: "caption" };
          if (!o.els) return false;
          return Object.keys(idmap).every((k) =>
            o.els[k] === win.document.getElementById("cl-hafs-" + idmap[k]));
        })(),
      };
    }
    const sat = win.__lab && win.__lab.satState ? win.__lab.satState() : null;
    const bandHost = win.document.getElementById("sat-bands");
    const scriptEl = win.document.querySelector(
      'script[src*="/models/hafs.js"]');
    const coneSvg = win.document.getElementById("advcone");
    const intSvg = win.document.getElementById("intensity");
    const adv = {
      coneIcons: coneSvg
        ? coneSvg.querySelectorAll(".ac-icon").length : 0,
      coneIconDelays: coneSvg ? Array.prototype.map.call(
        coneSvg.querySelectorAll(".ac-icon"),
        (g) => (g.getAttribute("style") || "")).slice(0, 3) : [],
      coneHasRevealer: !!(coneSvg &&
        coneSvg.querySelector("circle.ac-revealer")),
      coneHasPlacard: !!(coneSvg && coneSvg.querySelector(".ac-placard")),
      coneSpinners: coneSvg
        ? coneSvg.querySelectorAll(".ac-spin").length : 0,
      coneNote: text("advcone-note"),
      coneMethodBody: text("advcone-method-body"),
      coneEmptyShown:
        ((win.document.getElementById("advcone-empty") || {}).style || {})
          .display === "block",
      intRendered: !!(intSvg && intSvg.innerHTML.length > 100),
      intMissingText: text("intensity-missing"),
      intMissingShown:
        ((win.document.getElementById("intensity-missing") || {}).style ||
          {}).display === "block",
      intMethodBody: text("intensity-method-body"),
      intRows: win.__lab && win.__lab.intensityRows
        ? win.__lab.intensityRows() : null,
      advText: text("advtext"),
    };
    return {
      adv: adv,
      hafsCtor: hafs,
      hafsScriptInjected: !!scriptEl,
      hafsScriptSrc: scriptEl ? scriptEl.src : null,
      sat: sat,
      satBands: bandHost ? Array.prototype.map.call(
        bandHost.children,
        (b) => ({ slug: b.getAttribute("data-slug"),
                  active: b.classList.contains("active") })) : [],
      satImgSrc: (win.document.getElementById("sat-img") || {}).src || "",
      satTime: text("sat-time"),
      satEmptyShown:
        ((win.document.getElementById("sat-empty") || {}).style || {})
          .display === "block",
      fetched: (win.__fetched || []).slice(),
    };
  }

  function snapshot() {
    return {
      stage3: stage3Probe(),
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
    } else if (op.op === "applyAdvisory") {
      window.__lab.applyAdvisory(op.adv);
    } else if (op.op === "clickAdvTextTab") {
      const host = document.getElementById("advtext-tabs");
      const btn = host && Array.prototype.find.call(
        host.children, (b) => b.getAttribute("data-prod") === op.prod);
      if (btn) btn.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "clickSatPlay") {
      const btn = document.getElementById("sat-play");
      if (btn) btn.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "clickSatBand") {
      // Stage-3 satellite: click a band toggle by slug.
      const host = document.getElementById("sat-bands");
      const btn = host && Array.prototype.find.call(
        host.children, (b) => b.getAttribute("data-slug") === op.slug);
      if (btn) btn.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
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
