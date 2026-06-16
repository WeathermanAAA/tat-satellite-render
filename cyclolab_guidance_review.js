/* CycloLab guidance renderers (STAGE B, review-only) - hand-rolled SVG in the
   house style. Reuses the cone's fitProjection + graticule math, ace_core SSHS
   (baked as window.__SSHS__), the WIND_TIER C anchors, and the baked basemap. */
(function () {
  "use strict";
  var G = window.__GUIDANCE__ || {}, SH = window.__SHIPS__ || {},
      BM = window.__BASEMAP__ || { land: [], coast: [], borders: [], window: [0, 0, 0, 0] },
      SSHS = window.__SSHS__ || {};
  var AID = G.aids || {};
  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function svgEl(id, vw, vh, inner) {
    var s = $(id); if (!s) return;
    s.setAttribute("viewBox", "0 0 " + vw + " " + vh);
    s.innerHTML = inner;
  }
  // ---- palettes (the 3-way options board) ----------------------------------
  function lerp(a, b, t) { return a + (b - a) * t; }
  function rgb(c) { return "rgb(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + ")"; }
  function rampColor(stops, kt) {       // stops: [[kt,[r,g,b]], ...] ascending
    if (kt == null || isNaN(kt)) kt = 0;
    if (kt <= stops[0][0]) return stops[0][1].slice();
    for (var i = 1; i < stops.length; i++) {
      if (kt <= stops[i][0]) {
        var t = (kt - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
        var a = stops[i - 1][1], b = stops[i][1];
        return [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
      }
    }
    return stops[stops.length - 1][1].slice();
  }
  var RAINBOW = [[20, [91, 143, 249]], [40, [54, 197, 214]], [64, [70, 197, 106]],
    [90, [255, 225, 77]], [113, [255, 154, 47]], [137, [255, 77, 59]], [165, [227, 58, 212]]];
  var WINDTIER = [[0, [38, 104, 200]], [34, [38, 104, 200]], [50, [44, 168, 240]],
    [64, [255, 190, 52]], [100, [255, 150, 40]], [140, [255, 96, 32]]];
  var SSHS_BANDS = [[0, 34, "TD"], [34, 64, "TS"], [64, 83, "C1"], [83, 96, "C2"],
    [96, 113, "C3"], [113, 137, "C4"], [137, 999, "C5"]];
  var SSHS_THRESH = [34, 64, 83, 96, 113, 137];
  function sshsCat(kt) { var v = (kt == null || isNaN(kt)) ? 0 : +kt;
    for (var i = 0; i < SSHS_BANDS.length; i++) if (v < SSHS_BANDS[i][1]) return SSHS_BANDS[i][2];
    return "C5"; }
  function hexToRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
  function trackColor(pal, kt) {
    if (pal === "A") return rgb(rampColor(RAINBOW, kt));
    if (pal === "B") return SSHS[sshsCat(kt)] || "#888";
    return rgb(rampColor(WINDTIER, kt));   // C (default/locked house)
  }
  // continuous colorbar swatch (rainbow / windtier) or discrete (sshs)
  function colorbarSVG(pal, x, y, w, h) {
    var out = [], i, defsId = "cb" + pal;
    if (pal === "B") {
      var n = SSHS_BANDS.length, bw = w / n;
      for (i = 0; i < n; i++) out.push('<rect x="' + (x + i * bw).toFixed(1) + '" y="' + y +
        '" width="' + (bw + 0.6).toFixed(1) + '" height="' + h + '" fill="' + SSHS[SSHS_BANDS[i][2]] + '"/>');
    } else {
      var stops = (pal === "A") ? RAINBOW : WINDTIER, lo = stops[0][0], hi = 165;
      out.push('<defs><linearGradient id="' + defsId + '" x1="0" x2="1">');
      for (i = 0; i < stops.length; i++) {
        var off = ((stops[i][0] - lo) / (hi - lo) * 100).toFixed(1);
        out.push('<stop offset="' + Math.max(0, Math.min(100, off)) + '%" stop-color="' + rgb(stops[i][1]) + '"/>');
      }
      out.push('</linearGradient></defs>');
      out.push('<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" fill="url(#' + defsId + ')"/>');
    }
    out.push('<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>');
    [34, 64, 96, 137].forEach(function (kt) {
      var tx = x + (Math.min(kt, 165) / 165) * w;
      out.push('<text x="' + tx.toFixed(1) + '" y="' + (y + h + 11) + '" text-anchor="middle" fill="#8ea2bd" font-size="9" font-weight="600">' + kt + '</text>');
    });
    out.push('<text x="' + (x + w + 6) + '" y="' + (y + h - 1) + '" fill="#8ea2bd" font-size="9" font-weight="600">kt</text>');
    return out.join("");
  }
  // ---- shared projection (mirrors cone fitProjection) ----------------------
  function fitProjection(extent, viewW, hMin, hMax, margin) {
    var frameLon = (BM.window[2] + BM.window[3]) / 2.0;
    function normLon(lon) { while (lon - frameLon > 180) lon -= 360; while (lon - frameLon < -180) lon += 360; return lon; }
    var lats = extent.map(function (p) { return p.lat; }),
        lons = extent.map(function (p) { return normLon(p.lon); });
    var latMid = (Math.min.apply(null, lats) + Math.max.apply(null, lats)) / 2;
    var K = Math.max(0.2, Math.cos(latMid * Math.PI / 180));
    function pxu(lon) { return lon * 60 * K; } function pyu(lat) { return -lat * 60; }
    var xs = lons.map(pxu), ys = lats.map(pyu);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var spanX = Math.max(1e-6, x1 - x0), spanY = Math.max(1e-6, y1 - y0);
    var sW = (viewW - 2 * margin) / spanX, sH = (hMax - 2 * margin) / spanY, S = Math.min(sW, sH);
    var H = Math.max(hMin, Math.min(hMax, Math.round(spanY * S + 2 * margin)));
    var offX = (viewW - spanX * S) / 2, offY = (H - spanY * S) / 2;
    return { W: viewW, H: H, normLon: normLon,
      X: function (lon) { return (pxu(normLon(lon)) - x0) * S + offX; },
      Y: function (lat) { return (pyu(lat) - y0) * S + offY; },
      lonAt: function (x) { return ((x - offX) / S + x0) / (60 * K); },
      latAt: function (y) { return -((y - offY) / S + y0) / 60; } };
  }
  function graticule(pr) {
    var W = pr.W, H = pr.H, cas = [], lin = [], lab = [];
    var lonL = pr.lonAt(0), lonR = pr.lonAt(W), latT = pr.latAt(0), latB = pr.latAt(H), gl, ga, gx, gy;
    for (gl = Math.ceil(lonL / 5) * 5; gl <= lonR; gl += 5) {
      gx = pr.X(gl); if (gx < 1 || gx > W - 1) continue;
      cas.push('<line class="grat-cas" x1="' + gx.toFixed(1) + '" y1="0" x2="' + gx.toFixed(1) + '" y2="' + H + '"/>');
      lin.push('<line class="grat-lin" x1="' + gx.toFixed(1) + '" y1="0" x2="' + gx.toFixed(1) + '" y2="' + H + '"/>');
      var gn = ((gl % 360) + 360) % 360, glab = gn > 180 ? (360 - gn) + "°W" : (gn === 0 || gn === 180 ? gn + "°" : gn + "°E");
      if (gx > 26 && gx < W - 26) lab.push('<text class="grat-lab" x="' + gx.toFixed(1) + '" y="' + (H - 7) + '" text-anchor="middle">' + glab + '</text>');
    }
    for (ga = Math.ceil(latB / 5) * 5; ga <= latT; ga += 5) {
      gy = pr.Y(ga); if (gy < 1 || gy > H - 1) continue;
      cas.push('<line class="grat-cas" x1="0" y1="' + gy.toFixed(1) + '" x2="' + W + '" y2="' + gy.toFixed(1) + '"/>');
      lin.push('<line class="grat-lin" x1="0" y1="' + gy.toFixed(1) + '" x2="' + W + '" y2="' + gy.toFixed(1) + '"/>');
      if (gy > 16 && gy < H - 16) lab.push('<text class="grat-lab" x="7" y="' + (gy - 4).toFixed(1) + '">' + Math.abs(ga) + '°' + (ga >= 0 ? "N" : "S") + '</text>');
    }
    return '<g>' + cas.join("") + lin.join("") + lab.join("") + '</g>';
  }
  function basemapSVG(pr) {
    function path(rings, close) {
      return rings.map(function (r) {
        var d = "M" + r.map(function (p) { return pr.X(p[0]).toFixed(1) + "," + pr.Y(p[1]).toFixed(1); }).join("L");
        return d + (close ? "Z" : "");
      }).join("");
    }
    var land = '<path d="' + path(BM.land, true) + '" fill="#2a3344" fill-opacity="0.92"/>';
    var coast = '<path d="' + path(BM.coast, false) + '" fill="none" stroke="#e8eef5" stroke-width="1.3" stroke-opacity="0.8" stroke-linejoin="round" stroke-linecap="round"/>';
    var bord = '<path d="' + path(BM.borders, false) + '" fill="none" stroke="#e8eef5" stroke-width="0.6" stroke-opacity="0.3"/>';
    return '<rect x="0" y="0" width="' + pr.W + '" height="' + pr.H + '" fill="#0d2136"/>' + land + bord + coast;
  }
  function peakV(pts) { var m = null; pts.forEach(function (p) { if (p.vmax != null && (m == null || p.vmax > m)) m = p.vmax; }); return m; }

  // ============================ TRACKS ====================================
  var TRACK_AIDS = G.track_aids || [], CONS = new Set(G.consensus || []);
  var TAU_LABELS = [0, 24, 48, 72, 96, 120];
  function renderTracks(svgId, pal, compact) {
    var ext = [];
    TRACK_AIDS.forEach(function (t) { (AID[t] || []).forEach(function (p) { if (p.lat != null) ext.push(p); }); });
    if (!ext.length) { svgEl(svgId, 1000, 360, '<rect width="1000" height="360" fill="#0d2136"/><text x="500" y="180" fill="#8ea2bd" font-size="15" text-anchor="middle" font-weight="700">No track aids (statistical-only / fresh invest)</text>'); return; }
    // pad the extent ~0.6 deg so strands don't kiss the edge
    var lats = ext.map(function (p) { return p.lat; }), lons = ext.map(function (p) { return p.lon; });
    var pad = [{ lat: Math.min.apply(null, lats) - 0.8, lon: Math.min.apply(null, lons) - 0.8 },
               { lat: Math.max.apply(null, lats) + 0.8, lon: Math.max.apply(null, lons) + 0.8 }];
    var W = 1000, pr = fitProjection(ext.concat(pad), W, compact ? 240 : 360, compact ? 300 : 560, 16);
    var H = pr.H, body = [basemapSVG(pr), graticule(pr)];
    // each track aid colored by its peak wind; consensus drawn heavier + cased
    var ordered = TRACK_AIDS.slice().sort(function (a, b) { return (CONS.has(a) ? 1 : 0) - (CONS.has(b) ? 1 : 0); });
    ordered.forEach(function (t) {
      var pts = (AID[t] || []).filter(function (p) { return p.lat != null; });
      if (pts.length < 2) return;
      var col = trackColor(pal, peakV(pts)), cons = CONS.has(t);
      var d = "M" + pts.map(function (p) { return pr.X(p.lon).toFixed(1) + "," + pr.Y(p.lat).toFixed(1); }).join("L");
      if (cons) body.push('<path d="' + d + '" fill="none" stroke="#0a1320" stroke-width="6" stroke-linejoin="round" stroke-linecap="round" stroke-opacity="0.85"/>');
      body.push('<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="' + (cons ? 3.4 : 1.7) + '" stroke-opacity="' + (cons ? 1 : 0.82) + '" stroke-linejoin="round" stroke-linecap="round"/>');
      // dots at each fix
      pts.forEach(function (p) { body.push('<circle cx="' + pr.X(p.lon).toFixed(1) + '" cy="' + pr.Y(p.lat).toFixed(1) + '" r="' + (cons ? 2.4 : 1.5) + '" fill="' + col + '"/>'); });
    });
    // forecast-hour labels along the TVCN consensus (else first track aid)
    if (!compact) {
      var spine = AID.TVCN || AID.HCCA || AID[TRACK_AIDS[0]] || [];
      spine.filter(function (p) { return p.lat != null && TAU_LABELS.indexOf(p.tau) >= 0; }).forEach(function (p) {
        var x = pr.X(p.lon), y = pr.Y(p.lat);
        body.push('<g><rect x="' + (x + 5).toFixed(1) + '" y="' + (y - 8).toFixed(1) + '" width="' + (p.tau >= 100 ? 22 : 16) + '" height="13" rx="3" fill="rgba(7,16,28,0.86)" stroke="rgba(120,140,170,0.4)" stroke-width="0.7"/>' +
          '<text x="' + (x + 7).toFixed(1) + '" y="' + (y + 1.5).toFixed(1) + '" fill="#e8eef5" font-size="9.5" font-weight="700">' + p.tau + '</text></g>');
      });
    }
    // current position (tau 0)
    var c0 = (AID.TVCN || AID[TRACK_AIDS[0]] || []).filter(function (p) { return p.tau === 0 && p.lat != null; })[0];
    if (c0) body.push('<circle cx="' + pr.X(c0.lon).toFixed(1) + '" cy="' + pr.Y(c0.lat).toFixed(1) + '" r="4.5" fill="#fff" stroke="#0a1320" stroke-width="1.5"/>');
    // palette colorbar inset - a top-left key card (off the graticule edge labels).
    // The compact board minis omit it; their shared intensity-ramp demo carries the scale.
    if (!compact) {
      body.push('<g transform="translate(12,12)">' +
        '<rect x="-5" y="-4" width="190" height="42" rx="7" fill="rgba(7,16,28,0.82)" stroke="rgba(120,140,170,0.35)" stroke-width="1"/>' +
        '<text x="0" y="9" fill="#8ea2bd" font-size="10" font-weight="700">PEAK WIND</text>' +
        colorbarSVG(pal, 0, 15, 150, 8) + '</g>');
    }
    svgEl(svgId, W, H, body.join(""));
  }
  function tracksLegend() {
    var el = $("tracks-legend"); if (!el) return;
    var out = [];
    (G.track_aids || []).forEach(function (t) {
      var cons = CONS.has(t);
      out.push('<span class="lg' + (cons ? ' cons' : '') + '"><span class="sw" style="background:' + trackColor("C", peakV(AID[t] || [])) + ';height:' + (cons ? 4 : 3) + 'px"></span>' + (cons ? '<b>' + esc(t) + '</b> (consensus)' : esc(t)) + '</span>');
    });
    el.innerHTML = out.join("");
  }

  // intensity-ramp demo (board): the SAME palette on illustrative tracks at fixed
  // peak winds, so the scale's behavior is visible even when the live storm is weak.
  function renderScaleDemo(svgId, pal) {
    var W = 340, H = 92, kts = [30, 50, 70, 90, 120, 150], n = kts.length;
    var body = ['<rect x="0" y="0" width="' + W + '" height="' + H + '" fill="#0d2136"/>'];
    var mL = 8, mR = 8, gap = (W - mL - mR) / n, y0 = 18, y1 = H - 22;
    kts.forEach(function (kt, i) {
      var cx = mL + gap * (i + 0.5), col = trackColor(pal, kt);
      // a short curved strand
      var d = "M" + (cx - gap * 0.34).toFixed(1) + "," + y1 + " Q" + cx.toFixed(1) + "," + (y0 + 6) + " " + (cx + gap * 0.34).toFixed(1) + "," + (y0 + 14);
      body.push('<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="3.2" stroke-linecap="round"/>');
      body.push('<circle cx="' + (cx + gap * 0.34).toFixed(1) + '" cy="' + (y0 + 14) + '" r="2.6" fill="' + col + '"/>');
      body.push('<text x="' + cx.toFixed(1) + '" y="' + (H - 7) + '" text-anchor="middle" fill="#8ea2bd" font-size="10" font-weight="700">' + kt + '</text>');
    });
    body.push('<text x="' + (W - mR) + '" y="12" text-anchor="end" fill="#566b80" font-size="9" font-weight="600">peak wind (kt)</text>');
    svgEl(svgId, W, H, body.join(""));
  }

  // ========================== INTENSITY ===================================
  function renderIntensity() {
    var aids = G.intensity_aids || [];
    var W = 1000, H = 380, mL = 46, mR = 16, mT = 14, mB = 30;
    var pw = W - mL - mR, ph = H - mT - mB;
    var taus = [], vmaxes = [];
    aids.forEach(function (t) { (AID[t] || []).forEach(function (p) { if (p.vmax != null) { taus.push(p.tau); vmaxes.push(p.vmax); } }); });
    if (!taus.length) { svgEl("intensity", W, H, '<rect width="' + W + '" height="' + H + '" fill="#0d2136"/><text x="' + (W / 2) + '" y="' + (H / 2) + '" fill="#8ea2bd" text-anchor="middle" font-size="15" font-weight="700">No intensity aids</text>'); return; }
    var tmax = Math.max(120, Math.max.apply(null, taus)), vmax = Math.max(80, Math.ceil((Math.max.apply(null, vmaxes) + 10) / 20) * 20);
    function X(t) { return mL + (t / tmax) * pw; } function Y(v) { return mT + ph - (v / vmax) * ph; }
    var body = ['<rect x="0" y="0" width="' + W + '" height="' + H + '" fill="#0d2136"/>'];
    // SSHS category bands (reuse cone thresholds), faint
    SSHS_BANDS.forEach(function (b) {
      if (b[0] >= vmax) return;
      var y1 = Y(Math.min(b[1], vmax)), y0 = Y(b[0]);
      body.push('<rect x="' + mL + '" y="' + y1.toFixed(1) + '" width="' + pw + '" height="' + (y0 - y1).toFixed(1) + '" fill="' + SSHS[b[2]] + '" fill-opacity="0.12"/>');
      if (b[0] > 0) body.push('<line x1="' + mL + '" y1="' + y0.toFixed(1) + '" x2="' + (mL + pw) + '" y2="' + y0.toFixed(1) + '" stroke="' + SSHS[b[2]] + '" stroke-opacity="0.3" stroke-width="1"/>');
      if (b[0] > 0) body.push('<text x="' + (mL + pw - 4) + '" y="' + (Y(b[0]) - 3).toFixed(1) + '" text-anchor="end" fill="' + SSHS[b[2]] + '" font-size="9.5" font-weight="700" opacity="0.8">' + b[2] + '</text>');
    });
    // axes
    for (var v = 0; v <= vmax; v += 20) body.push('<text x="' + (mL - 7) + '" y="' + (Y(v) + 3).toFixed(1) + '" text-anchor="end" fill="#8ea2bd" font-size="10" font-weight="600">' + v + '</text>');
    body.push('<text x="14" y="' + (mT + 4) + '" fill="#8ea2bd" font-size="10" font-weight="700">kt</text>');
    for (var t = 0; t <= tmax; t += 24) { body.push('<line x1="' + X(t).toFixed(1) + '" y1="' + mT + '" x2="' + X(t).toFixed(1) + '" y2="' + (mT + ph) + '" stroke="rgba(150,170,200,0.12)" stroke-width="1"/>'); body.push('<text x="' + X(t).toFixed(1) + '" y="' + (H - 10) + '" text-anchor="middle" fill="#8ea2bd" font-size="10" font-weight="600">' + t + '</text>'); }
    body.push('<text x="' + (mL + pw / 2) + '" y="' + (H - 0.5) + '" text-anchor="middle" fill="#8ea2bd" font-size="9.5" font-weight="600">forecast hour</text>');
    // aid styling tiers
    var HIRES = { HFAI: "#46c56a", HFBI: "#2bd4c0", HWFI: "#ffe14d", HMNI: "#ff9a2f" };
    function styleOf(t) {
      if (t === "IVCN") return { c: "#ffffff", w: 3.2, op: 1, dash: "", cons: true };
      if (HIRES[t]) return { c: HIRES[t], w: 2.2, op: 0.95, dash: "" };
      if (t === "DSHP" || t === "LGEM" || t === "SHIP") return { c: "#8ea2bd", w: 1.4, op: 0.85, dash: "3,3" };
      return { c: "#5d6b80", w: 1.2, op: 0.7, dash: "" };   // coarse globals (AVNI)
    }
    var ordered = aids.slice().sort(function (a, b) { return (a === "IVCN" ? 1 : 0) - (b === "IVCN" ? 1 : 0); });
    ordered.forEach(function (t) {
      var pts = (AID[t] || []).filter(function (p) { return p.vmax != null; }); if (pts.length < 2) return;
      var st = styleOf(t), d = "M" + pts.map(function (p) { return X(p.tau).toFixed(1) + "," + Y(p.vmax).toFixed(1); }).join("L");
      if (st.cons) body.push('<path d="' + d + '" fill="none" stroke="#0a1320" stroke-width="5.4" stroke-linejoin="round" stroke-opacity="0.8"/>');
      body.push('<path d="' + d + '" fill="none" stroke="' + st.c + '" stroke-width="' + st.w + '" stroke-opacity="' + st.op + '" stroke-dasharray="' + st.dash + '" stroke-linejoin="round" stroke-linecap="round"/>');
      pts.forEach(function (p) { body.push('<circle cx="' + X(p.tau).toFixed(1) + '" cy="' + Y(p.vmax).toFixed(1) + '" r="' + (st.cons ? 2.2 : 1.4) + '" fill="' + st.c + '"/>'); });
    });
    body.push('<rect x="' + mL + '" y="' + mT + '" width="' + pw + '" height="' + ph + '" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1"/>');
    svgEl("intensity", W, H, body.join(""));
    // legend
    var leg = $("intensity-legend"); if (leg) {
      leg.innerHTML = ordered.slice().reverse().map(function (t) {
        var st = styleOf(t);
        var tier = st.cons ? " (consensus)" : (HIRES[t] ? " (hi-res)" : (st.dash ? " (statistical)" : " (global)"));
        return '<span class="lg' + (st.cons ? ' cons' : '') + '"><span class="sw" style="background:' + st.c + ';height:' + Math.max(3, st.w) + 'px"></span>' + (st.cons ? '<b>' + esc(t) + '</b>' : esc(t)) + tier + '</span>';
      }).join("");
    }
  }

  // ============================= SHIPS ====================================
  function spark(vals, taus, w, h, color) {
    var ok = vals.map(function (v, i) { return { v: v, t: taus[i] }; }).filter(function (o) { return o.v != null; });
    if (ok.length < 2) return '<svg viewBox="0 0 ' + w + ' ' + h + '"><text x="' + (w / 2) + '" y="' + (h / 2) + '" text-anchor="middle" fill="#566" font-size="9">no data</text></svg>';
    var vs = ok.map(function (o) { return o.v; }), lo = Math.min.apply(null, vs), hi = Math.max.apply(null, vs);
    if (hi === lo) { hi += 1; lo -= 1; }
    var tmax = Math.max.apply(null, taus), mb = 12, mt = 4;
    function X(t) { return 2 + (t / tmax) * (w - 4); } function Y(v) { return mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo)); }
    var d = "M" + ok.map(function (o) { return X(o.t).toFixed(1) + "," + Y(o.v).toFixed(1); }).join("L");
    var out = ['<svg viewBox="0 0 ' + w + ' ' + h + '">'];
    out.push('<line x1="2" y1="' + (h - mb).toFixed(1) + '" x2="' + (w - 2) + '" y2="' + (h - mb).toFixed(1) + '" stroke="rgba(150,170,200,0.18)" stroke-width="1"/>');
    out.push('<path d="' + d + '" fill="none" stroke="' + color + '" stroke-width="1.8" stroke-linejoin="round"/>');
    out.push('<circle cx="' + X(ok[0].t).toFixed(1) + '" cy="' + Y(ok[0].v).toFixed(1) + '" r="2" fill="' + color + '"/>');
    out.push('<text x="2" y="' + (h - 2) + '" fill="#566b80" font-size="8">' + lo.toFixed(0) + '</text>');
    out.push('<text x="' + (w - 2) + '" y="' + (h - 2) + '" text-anchor="end" fill="#566b80" font-size="8">' + hi.toFixed(0) + '</text>');
    return out.join("") + '</svg>';
  }
  function renderShips() {
    var root = $("ships-root"); if (!root) return;
    if (!SH || SH.available === false) {
      root.innerHTML = '<div class="unavail">SHIPS unavailable for this system' +
        (SH && SH.reason ? ' (' + esc(SH.reason) + ')' : '') + '</div>'; return;
    }
    var taus = SH.taus || [], env = SH.env_series || {};
    var head = [];
    var idl = (SH.header || {}).id_line || SH.sid || "";
    head.push('<span class="chip"><b>' + esc(idl) + '</b></span>');
    if (SH.ahi) head.push('<span class="chip">Annularity (AHI) <b>' + esc(SH.ahi.value) + '</b>' + (SH.ahi.verdict ? ' &middot; ' + esc(SH.ahi.verdict.split(",")[0]) : '') + '</span>');
    if (SH.prelim_ri_prob != null) head.push('<span class="chip ri">Prelim RI prob <b>' + esc(SH.prelim_ri_prob) + '%</b></span>');
    var stype = (SH.storm_type || [])[0]; if (stype) head.push('<span class="chip">Storm type <b>' + esc(stype) + '</b></span>');
    // env small-multiples - the operationally key fields
    var WANT = [["SHEAR (KT)", "#ffd24d"], ["SST (C)", "#ff7a59"], ["700-500 MB RH", "#46c56a"],
      ["POT. INT. (KT)", "#2b9cff"], ["HEAT CONTENT", "#ff9a2f"], ["200 MB DIV", "#7aa0ff"],
      ["STM SPEED (KT)", "#8ea2bd"], ["V (KT) NO LAND", "#e8eef5"], ["TH_E DEV (C)", "#c08bff"]];
    var cells = WANT.filter(function (p) { return env[p[0]]; }).map(function (p) {
      var v = env[p[0]], cur = v.find(function (x) { return x != null; });
      return '<div class="sm-cell"><div class="smt">' + esc(p[0]) + '</div>' +
        '<div class="smv">now ' + (cur == null ? "n/a" : cur) + '</div>' + spark(v, taus, 200, 64, p[1]) + '</div>';
    }).join("");
    // RI matrix table
    var rm = SH.ri_matrix || { cols: [], rows: {} }, tbl = "";
    if (rm.cols && rm.cols.length) {
      var thead = '<tr><th>RI (kt/h)</th>' + rm.cols.map(function (c) { return '<th>' + esc(c) + '</th>'; }).join("") + '</tr>';
      var trows = Object.keys(rm.rows).map(function (rn) {
        return '<tr><td class="rowname">' + esc(rn) + '</td>' + rm.cols.map(function (c) {
          var val = rm.rows[rn][c]; return '<td>' + (val == null ? "&middot;" : esc(val) + '%') + '</td>';
        }).join("") + '</tr>';
      }).join("");
      tbl = '<table class="ri-table"><caption>RI probability matrix (% in next, vs threshold/hours)</caption>' + thead + trows + '</table>';
    }
    root.innerHTML = '<div class="ships-head">' + head.join("") + '</div>' +
      '<div class="sm-grid">' + cells + '</div>' + tbl;
  }

  // ---- page meta + drive --------------------------------------------------
  function meta() {
    var el = $("ph-meta"); if (!el) return;
    var it = G.init_time || "", src = "NHC ATCF aid_public + SHIPS";
    el.innerHTML = 'Init <b>' + esc((it || "").replace("T", " ").replace("Z", "Z")) + '</b> &middot; ' +
      '<b>' + (G.track_aids || []).length + '</b> track aids &middot; <b>' + (G.intensity_aids || []).length + '</b> intensity aids &middot; ' + esc(src);
  }
  function drawAll() {
    meta();
    renderTracks("tracks", "C", false);
    tracksLegend();
    renderTracks("boardA", "A", true); renderScaleDemo("demoA", "A");
    renderTracks("boardB", "B", true); renderScaleDemo("demoB", "B");
    renderTracks("boardC", "C", true); renderScaleDemo("demoC", "C");
    renderIntensity();
    renderShips();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", drawAll);
  else drawAll();
  window.__drawAll = drawAll;
})();
