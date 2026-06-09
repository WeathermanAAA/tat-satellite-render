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
      // final-gate-2 #1: the per-storm SST hero meta (plan.sst_meta);
      // absent -> !ok response -> the shell's honest fallback panel.
      const SST_META = Array.isArray(PLAN) ? null : (PLAN.sst_meta || null);
      // final-gate-3 #4: the advisory JSON (plan.adv) - the page's own
      // poll/single-ENDED-fetch hydrates the Advisories tab from this.
      // Absent -> a clean fetch failure (the pre-first-advisory window).
      const ADV = Array.isArray(PLAN) ? null : (PLAN.adv || null);
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
        if (/\/sst\/meta\.json/.test(url)) {
          return jsonResponse(SST_META);
        }
        if (/\/adv\/[^/]+\.json/.test(url)) {
          return jsonResponse(ADV);
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

  function odoText(id) {
    // final-gate-3 #2: the RENDERED plain-text value (no cells/strips).
    const el = document.getElementById("odo-" + id);
    return el ? el.textContent : null;
  }
  function odoSwapping(id) {
    // the only animation left: the subtle fade-in class on a CHANGE.
    const el = document.getElementById("odo-" + id);
    return !!(el && el.classList.contains("odo-swap"));
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
      coneReveal: coneSvg && coneSvg.querySelector(".ac-conegrp")
        ? coneSvg.querySelector(".ac-conegrp").getAttribute("data-reveal")
        : null,
      coneClip: coneSvg && coneSvg.querySelector(".ac-conegrp")
        ? (coneSvg.querySelector(".ac-conegrp")
             .getAttribute("clip-path") || "none")
        : "",
      // maps-pass R3 #2: the lockup is an HTML OVERLAY pinned to the panel's
      // top-left corner (not an in-SVG element), so read it from the DOM.
      coneTitle: (document.getElementById("advcone-lockup") &&
                  !document.getElementById("advcone-lockup").hidden)
        ? {
            head: (document.getElementById("advcone-lockup-name") || {})
              .textContent || "",
            eyebrow: (document.querySelector(
              "#advcone-lockup .al-eyebrow") || {}).textContent || "",
            corner: "top-left",
          } : null,
      coneWatermark: coneSvg && coneSvg.querySelector(".ac-ocean")
        ? {
            x: parseFloat(coneSvg.querySelector(".ac-ocean")
              .getAttribute("x")),
            y: parseFloat(coneSvg.querySelector(".ac-ocean")
              .getAttribute("y")),
            text: coneSvg.querySelector(".ac-ocean").textContent,
          } : null,
      coneFramed: !!(coneSvg && coneSvg.querySelector(".ac-frame")),
      coneViewBox: coneSvg ? (coneSvg.getAttribute("viewBox") || "") : "",
      coneIconsDetail: coneSvg ? Array.prototype.map.call(
        coneSvg.querySelectorAll(".ac-icon"), (g) => ({
          tau: parseInt(g.getAttribute("data-tau"), 10),
          tropical: g.getAttribute("data-tropical") === "1",
          delay: parseFloat(((g.getAttribute("style") || "")
            .match(/animation-delay:([0-9.]+)s/) || [0, "0"])[1]),
          hasCatLabel: !!g.querySelector("text.ac-cat"),
        })) : [],
      conePlacards: coneSvg ? Array.prototype.map.call(
        coneSvg.querySelectorAll('[data-role="placard"]'), (g) => ({
          i: parseInt(g.getAttribute("data-i"), 10),
          x: parseFloat(g.getAttribute("data-x")),
          y: parseFloat(g.getAttribute("data-y")),
          w: parseFloat(g.getAttribute("data-w")),
          h: parseFloat(g.getAttribute("data-h")),
          gap: parseFloat(g.getAttribute("data-gap")),
          iconr: parseFloat(g.getAttribute("data-iconr")),
          label: (g.querySelector("text") || {}).textContent || "",
        })) : [],
      // final-gate-3 #1: leader lines are GONE. The probe stays so the
      // invariant ("zero leader lines ever") is directly assertable.
      coneLeaders: coneSvg ? Array.prototype.map.call(
        coneSvg.querySelectorAll('line[data-role="leader"]'),
        (l) => parseInt(l.getAttribute("data-for"), 10)) : [],
      coneLand: coneSvg
        ? coneSvg.querySelectorAll(".ac-land").length : 0,
      coneGraticule: coneSvg
        ? coneSvg.querySelectorAll(".ac-graticule line").length : 0,
      coneHasPlacard: !!(coneSvg && coneSvg.querySelector(".ac-placard")),
      coneSpinners: coneSvg
        ? coneSvg.querySelectorAll(".ac-spin").length : 0,
      coneNote: text("advcone-note"),
      coneMethodBody: text("advcone-method-body"),
      coneHooks: !!(win.__lab && win.__lab.cone && win.__lab.cone()),
      coneSeek: win.__coneSeek || null,
      coneRevealPathLen: (() => {
        const p = coneSvg && coneSvg.querySelector(".ac-reveal-path");
        return p ? (p.getAttribute("d") || "").length : 0;
      })(),
      // raw path data for python-side geometry assertions (final-gate-2
      // #3: the corridor must CONTAIN the cone; point-in-polygon over
      // the outline vertices proves it without pixels).
      coneRevealD: (() => {
        const p = coneSvg && coneSvg.querySelector(".ac-reveal-path");
        return p ? (p.getAttribute("d") || "") : "";
      })(),
      coneOutlineD: (() => {
        const g = coneSvg && coneSvg.querySelector(".ac-conegrp");
        return g && g.children[1]
          ? (g.children[1].getAttribute("d") || "") : "";
      })(),
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

  function overviewProbe() {
    // FG-R3 #7/#8/#11: the two Overview plots + the two-column layout.
    const tp = document.getElementById("trackplot");
    const sw = document.getElementById("swathplot");
    const wipe = document.querySelector("#sec-overview .wipe");
    let gridCols = "";
    try {
      gridCols = wipe ? (window.getComputedStyle(wipe)
        .getPropertyValue("grid-template-columns") || "") : "";
    } catch (e) { gridCols = ""; }
    function shapeCounts(svgEl, cls) {
      if (!svgEl) return { circle: 0, square: 0, triangle: 0, total: 0 };
      const circ = svgEl.querySelectorAll("circle." + cls).length;
      const sq = svgEl.querySelectorAll("rect." + cls).length;
      const tri = svgEl.querySelectorAll("polygon." + cls).length;
      return { circle: circ, square: sq, triangle: tri,
               total: circ + sq + tri };
    }
    const tpShapes = shapeCounts(tp, "tp-dot");
    return {
      // ---- track-history plot (#7) ----
      trackRendered: !!(tp && tp.innerHTML.length > 100),
      trackChildCount: tp ? tp.children.length : 0,
      trackDotCount: tpShapes.total,
      trackShapes: tpShapes,
      trackColorbar: !!(tp && tp.querySelector(".tp-cbar-frame")),
      trackColorbarTicks: tp
        ? tp.querySelectorAll(".tp-cbar-tick").length : 0,
      trackWindField: tp
        ? tp.querySelectorAll('path[fill-rule="evenodd"]').length : 0,
      trackTitle: (function () {            // maps-pass R5: HTML overlay
        var b = document.getElementById("trackplot-lockup");
        var h = document.getElementById("trackplot-lockup-head");
        return (b && h && !b.hidden) ? h.textContent.trim() : null;
      })(),
      trackWindKey: (function () {
        var b = document.getElementById("trackplot-windkey");
        return (b && !b.hidden) ? b.textContent.trim() : null;
      })(),
      trackLegend: (function () {           // maps-pass R6: HTML overlay
        var b = document.getElementById("trackplot-stats");
        return !!(b && !b.hidden);
      })(),
      trackNote: text("trackplot-note"),
      trackLand: tp ? tp.querySelectorAll(".ac-land").length : 0,
      trackGraticule: tp ? tp.querySelectorAll(".ac-graticule line").length
        : 0,
      // ---- wind-history swath plot (#8) ----
      swathDisplay: sw ? (sw.style.display || "") : "",
      swathRendered: !!(sw && sw.style.display !== "none" &&
        sw.innerHTML.length > 100),
      swath34: sw ? sw.querySelectorAll(".sw-34").length : 0,
      swath64: sw ? sw.querySelectorAll(".sw-64").length : 0,
      swathTitle: (function () {            // maps-pass R5: HTML overlay
        var b = document.getElementById("swathplot-lockup");
        var h = document.getElementById("swathplot-lockup-head");
        return (b && h && !b.hidden) ? h.textContent.trim() : null;
      })(),
      swathEmptyShown:
        ((document.getElementById("swath-empty") || {}).style || {})
          .display === "block",
      swathEmptyText: text("swath-empty"),
      swathDerivedShown: (() => {
        const d = document.getElementById("swath-derived");
        return !!(d && !d.hidden);
      })(),
      swathMethodShown: (() => {
        const m = document.getElementById("swath-method");
        return !!(m && !m.hidden);
      })(),
      swathMethodBody: text("swath-method-body"),
      swathNote: text("swathplot-note"),
      // ---- two-column layout (#11) ----
      ovGridCols: gridCols,
      ovCols: document.querySelectorAll("#sec-overview .ov-col").length,
      ovCardOrder: (() => {
        const ids = ["card-hero", "card-track", "card-wp", "card-swath"];
        return ids.map((id) => {
          const el = document.getElementById(id);
          let o = "";
          try { o = el ? window.getComputedStyle(el).order : ""; }
          catch (e) { o = ""; }
          return { id: id, order: o, present: !!el };
        });
      })(),
    };
  }

  function snapshot() {
    return {
      stage3: stage3Probe(),
      overview: overviewProbe(),
      cat: document.documentElement.getAttribute("data-cat"),
      ended: document.documentElement.hasAttribute("data-ended"),
      chip: text("chip"),
      chipShown: (() => {
        const c = document.getElementById("chip");
        return !!c && c.style.display !== "none";
      })(),
      stormName: text("storm-name"),
      activeSection: activeSection(),
      activeNav: activeNav(),
      odo: {
        vmax: odo("vmax"), mslp: odo("mslp"), ace: odo("ace"),
        pos: odo("pos"), fix: odo("fix"), cat: odo("cat"),
      },
      // final-gate-3 #2: the rendered plain text + the fade-on-change flag
      odoText: {
        vmax: odoText("vmax"), mslp: odoText("mslp"), cat: odoText("cat"),
      },
      odoSwapVmax: odoSwapping("vmax"),
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
      // final-gate-3 #3: wind-unit display surfaces
      windUnits: (window.__lab && window.__lab.windUnits)
        ? window.__lab.windUnits() : null,
      vmaxText: text("odo-vmax"),
      vmaxUnit: text("vmax-unit"),
      moveText: text("odo-move"),
      chartLabel: (function () {
        const c = document.getElementById("chart");
        if (!c) return "";
        const t = Array.prototype.find.call(
          c.querySelectorAll("text"),
          (x) => /wind/.test(x.textContent || ""));
        return t ? t.textContent : "";
      })(),
      settingsUnits: (function () {
        const h = document.getElementById("settings-units");
        if (!h) return null;
        return Array.prototype.map.call(h.children, function (b) {
          return { unit: b.getAttribute("data-unit"),
                   checked: b.getAttribute("aria-checked") === "true" }; });
      })(),
      settingsOpen: (function () {
        const p = document.getElementById("settings-pop");
        return !!(p && !p.hidden);
      })(),
      bannerClasses: bannerClasses(),
      hero: (() => {
        const img = document.getElementById("sst-hero-img");
        const g = document.getElementById("sst-hero-glyph");
        const note = document.getElementById("sst-hero-note");
        if (!img) return null;
        const host = document.getElementById("sst-hero-layers");
        const layers = host
          ? Array.prototype.map.call(host.children,
              (b) => b.getAttribute("data-slug")) : [];
        const active = host
          ? Array.prototype.filter.call(host.children,
              (b) => b.classList.contains("active"))
              .map((b) => b.getAttribute("data-slug")) : [];
        return {
          imgUrl: img.getAttribute("data-url") || "",
          imgShown: img.style.display !== "none" &&
            !!img.getAttribute("data-url"),
          layers: layers,
          activeLayer: active.length === 1 ? active[0] : null,
          head: (document.getElementById("sst-hero-head") || {})
            .textContent || "",
          glyphHtml: g ? g.innerHTML.length : 0,
          glyphLabel: g && g.querySelector("text.ac-cat")
            ? g.querySelector("text.ac-cat").textContent : "",
          caption: note ? note.textContent : "",
          sub: (document.getElementById("sst-hero-sub") || {})
            .textContent || "",
        };
      })(),
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
    } else if (op.op === "setWindUnits") {
      // final-gate-3 #3: change the display wind unit.
      if (window.__lab && window.__lab.setWindUnits)
        window.__lab.setWindUnits(op.unit);
    } else if (op.op === "openSettings") {
      var sb = document.getElementById("settings-btn");
      if (sb) sb.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "clickSettingsUnit") {
      var sh = document.getElementById("settings-units");
      var ub = sh && Array.prototype.find.call(sh.children,
        function (b) { return b.getAttribute("data-unit") === op.unit; });
      if (ub) ub.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "clickSatSpeed") {
      // final-gate-3 #5: click a speed preset by multiplier.
      const host = document.getElementById("sat-speed");
      const btn = host && Array.prototype.find.call(
        host.children,
        (b) => parseFloat(b.getAttribute("data-speed")) === op.speed);
      if (btn) btn.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "coneSeek") {
      // final-gate-2 #3: place the growth front deterministically.
      const h = window.__lab.cone && window.__lab.cone();
      window.__coneSeek = h ? h.seek(op.f) : null;
    } else if (op.op === "coneSettle") {
      const h2 = window.__lab.cone && window.__lab.cone();
      window.__coneSettle = h2 ? h2.settle() : null;
    } else if (op.op === "clickHeroLayer") {
      // final-gate-2 #1: pick an SST hero base layer by slug.
      const host = document.getElementById("sst-hero-layers");
      const btn = host && Array.prototype.find.call(
        host.children, (b) => b.getAttribute("data-slug") === op.slug);
      if (btn) btn.dispatchEvent(
        new dom.window.Event("click", { bubbles: true }));
    } else if (op.op === "renderTrackPlot") {
      if (window.__lab && window.__lab.renderTrackPlot)
        window.__lab.renderTrackPlot(op.storm || null);
    } else if (op.op === "renderSwathPlot") {
      if (window.__lab && window.__lab.renderSwathPlot)
        window.__lab.renderSwathPlot(op.storm || null);
    } else if (op.op === "removeBannerClasses") {
      // test helper: clear banner anim classes to detect re-add / churn.
      const b = document.getElementById("banner");
      if (b) { b.classList.remove("shine"); b.classList.remove("xfade"); }
    } else if (op.op === "callLab") {
      // generic: call window.__lab[fn](...args) and return its result
      // alongside the snapshot (FG-R3 #1/#4 unit + palette tests).
      const f = window.__lab && window.__lab[op.fn];
      const r = (typeof f === "function") ? f.apply(null, op.args || []) : null;
      await drain();
      out.push({ op: op.op, fn: op.fn, result: r, state: snapshot() });
      continue;
    } else {
      out.push({ op: op.op, error: "unknown op" });
      continue;
    }
    await drain();
    out.push({ op: op.op, state: snapshot() });
  }

  // exit ONLY after stdout flushes - process.exit() truncates large
  // writes (the cone path snapshots exceed the pipe buffer).
  process.stdout.write(
    out.map((o) => JSON.stringify(o)).join("\n") + "\n",
    () => process.exit(0));
})().catch((e) => {
  process.stderr.write(String(e && e.stack || e) + "\n");
  process.exit(1);
});
