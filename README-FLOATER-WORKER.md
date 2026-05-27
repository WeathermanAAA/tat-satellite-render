# Floater poller — deploy into `tat-satellite-render`

A persistent worker that keeps R2 continuously fresh with storm-following
satellite floaters, alongside the existing `/render` web service. Drop these
files into the `tat-satellite-render` repo on a new branch and deploy as a
**second Railway service** in the same project.

## Files
- `floater_poller.py` — the worker (this is the whole thing).
- `Procfile` — add the `worker:` line to your existing Procfile **(or)**
- `railway.toml` — config-as-code alternative for the second service.
- `requirements-delta.txt` — merge `boto3` + `requests` into requirements.txt.

## Create the second Railway service
1. New branch in `tat-satellite-render`; add `floater_poller.py`, merge the
   requirements delta, add the `worker:` Procfile line (or `railway.toml`).
2. Railway → same project → **New Service → from repo** (same repo/branch).
   Start command: `python floater_poller.py`. No port, no healthcheck.
3. Set env vars (below). Deploy. Watch logs for `active storms: N`.

## Environment
Shared with the render service:
- `R2_ENDPOINT`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (or `AWS_*`)
- `R2_BUCKET` (default `triple-a-tropics-media`)

Worker-specific:
- `RENDER_BASE_URL` — **use private networking**, e.g.
  `http://<render-service-name>.railway.internal:<PORT>`. This bypasses the
  public 10 req/min/IP limiter so every band polls at its 60 s target.
- `RATE_MIN_SPACING_S` — min gap between `/render` calls. Leave at `1.0` for
  private networking. **If you point `RENDER_BASE_URL` at the public https URL,
  set `7`** (≈8.5/min) — the cold-band cadence then auto-stretches:
  `C_cold = max(60, U_cold × spacing)` so hot IR bands stay fresh.

Optional tunables (sane defaults in code): `BBOX_DEG=8`, `CADENCE_TARGET_S=60`,
`TRACKS_BASE=https://triple-a-tropics.com`, `TRACKS_REFRESH_S=600`,
`NIGHT_ZENITH_DEG=85`, `RECENT_WINDOW_H=6`, `HISTORY_WINDOW_H=24`,
`THIN_SPACING_S=300`, `EXTRAPOLATE_MAX_H=6`, `LOG_LEVEL=INFO`.

## What it writes to R2
- Frames: `floaters/{slug}/{band}/{YYYYMMDDTHHMMZ}.png` — `Cache-Control:
  public, max-age=31536000, immutable` (timestamped keys never change).
- Per-storm manifest: `floaters/{slug}/manifest.json` — `max-age=30`.
- Top-level manifest: `floaters/manifest.json` — `max-age=30` (empty
  `storms` array when nothing is active → the frontend widget self-hides).

Bands: `ir` (clean IR), `irbd` (Dvorak BD, **hot**), `wv_up`, `wv_low`,
`truecolor` (daytime-only), `swir`. Hot = `ir`, `irbd` (always 60 s).

## Resilience
- `/render` errors → retry + backoff, then skip the unit; 8 consecutive fails
  open a 60 s circuit breaker. `422` (coverage/night) is treated as a skip.
- R2 upload failure → log + skip; the manifest only references keys that
  uploaded successfully, so it never points at a missing frame.
- No active storms → idle (refresh storm list every ~10 min, else sleep).
- Restart → resyncs `last_hash` per (storm, band) from the R2 manifests;
  content-hash dedup makes re-uploads idempotent (no dupes, no lost history).

## New-frame detection
sha256 of the returned PNG is the source of truth (skip if unchanged). A
`X-Cache: HIT` or a source-scan-time header (`X-Source-Time`/`X-Scan-Time`/
`X-Timestamp`), **if** `/render` exposes one, is used opportunistically:
the scan-time header (when present) also sets the frame's timestamp for a more
accurate loop time-axis; otherwise the upload time is used.
