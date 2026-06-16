# CycloLab guidance DATA layer (`cyclolab_guidance_poller.py`)

DATA LAYER ONLY — fetch + parse the NHC ATCF a-deck (model **tracks + intensity**)
and **SHIPS** text for every active NHC storm AND invest (AL/EP/CP), and write clean
per-entity JSON to R2 for the CycloLab pages to hydrate later. No renderers, no UI.

Always-on Railway worker (NOT a GitHub cron — immune to the scheduled-event
throttling that stalls the enscenters ingest). Per-entity isolation, an always-emitted
heartbeat, idempotent never-miss re-writes (a-decks update each 00/06/12/18Z; early
aids land first, full guidance trickles over hours), graceful degrade (fresh invest =
statistical-only; missing SHIPS = `{"available": false}`). Reads the active-entity list
from the public global feed (read-only); does NOT touch ACE/track/climo or floater code.

## Sources
- a-deck (tracks + intensity, one source): `aid_public/a{al|ep|cp}{NN}{YYYY}.dat.gz`
- SHIPS: `stext/{YYMMDDCC}{AL|EP|CP}{NN}{YY}_ships.txt`

## Output (R2)
`{GUIDANCE_R2_PREFIX}/{sid}/guidance.json` and `.../{sid}/ships.json`, sid =
`NHC_{BASIN}{NN}{YYYY}` (matches the global feed / CycloLab key convention). Each JSON
carries `init_time` + `init_cycle` + `generated_at` as the cache-bust token; written
with `Cache-Control: public, max-age=120` so re-writes propagate.

  guidance: { sid, init_time, init_cycle, aids:{TECH:[{tau,lat,lon,vmax,mslp}]},
              present_aids, track_aids, intensity_aids, consensus }
  ships:    { available, header, taus, env_series:{param:[by-tau]}, storm_type,
              prelim_ri_prob, ri_predictor_table, ri_threshold_probs,
              ri_matrix:{cols,rows}, ahi:{value,verdict} }   |   { available:false }

## Deploy (Railway)
Dedicated service in this repo using **`railway.guidance.json`**
(`. /opt/venv/bin/activate && python cyclolab_guidance_poller.py`) with the existing
R2 env (`R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`/`AWS_*`,
`R2_SECRET_ACCESS_KEY`). Optional: `GUIDANCE_R2_PREFIX` (default `cyclolab`; set
`shadow/cyclolab` for a staging dry-run), `GUIDANCE_POLL_INTERVAL_S` (default 900).

### Slim build (own deps, not the satellite stack)
The poller only fetches text and writes JSON, so it builds from its OWN minimal deps
(`requests` + `boto3` only — see `requirements-guidance.txt`) via
`nixpacks-guidance.toml`, mirroring the HAFS service's `requirements-hafs.txt` /
`nixpacks-hafs.toml` split. This drops the build from 12+ min (the default
`requirements.txt` pulls Pillow / pyspectral — the Rayleigh-LUT download alone is
~9 min — ace-core, and the whole satellite-render stack) to a few seconds.

**Railway step:** `railway.guidance.json` now carries
`build.nixpacksConfigPath = "nixpacks-guidance.toml"`. The guidance service already
reads `railway.guidance.json` (that is where its `cyclolab_guidance_poller.py` start
command lives), so the slim build is picked up **automatically on the next deploy —
no manual change needed**. Only verify, if in doubt, that the service's
*Settings → Config-as-code path* is `railway.guidance.json` (it must be already,
since the service runs the guidance poller and not the default `uvicorn` web start) —
that is the same one-line per-service setting the HAFS service uses for
`railway.hafs.json`.

Parsers (`cyclolab_guidance.py`) are pure + unit-tested:
`python -m unittest tests.test_cyclolab_guidance` (offline, with captured EP932026
fixtures as format-drift guards).
