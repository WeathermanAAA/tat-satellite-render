# tat-satellite-render

Custom GOES-19 ABI render service for [triple-a-tropics.com/satellite/](https://triple-a-tropics.com/satellite/).

Worldview-Snapshots-style: user picks a bbox + time + channel + enhancement, gets back a clean cropped PNG with title strip and basemap. Backend pulls the source NetCDF straight from the public NOAA AWS bucket, crops in geostationary projection space before materializing pixels (RAM-cheap), and renders with matplotlib + cartopy.

## Endpoints

### `POST /render`

```json
{
  "bbox": [-67, -10, 10, 24],
  "time": "2017-09-05T17:45:00Z",
  "channel": 13,
  "enhancement": "tat_neon"
}
```

Returns `image/png`. Headers:
- `X-Cache: HIT | MISS`
- `X-Render-Ms: <int>`
- `X-Product: CMIPF | CMIPC | CMIPM1 | CMIPM2`

Constraints:
- `bbox` size capped at 30°×30°, must overlap GOES-19 disk.
- `channel`: 2 (visible), 8 (upper WV), or 13 (clean IR).
- `enhancement`: `tat_neon`, `dvorak_bd`, or `grayscale`.
- `time`: ISO 8601 string or `"latest"`.

### `GET /health`

```json
{
  "status": "ok",
  "goes_bucket": "noaa-goes19",
  "buckets_for_latest": ["noaa-goes19"],
  "override_active": false,
  "goes_bucket_reachable": true,
  "cache_entries": 12,
  "cache_bytes": 4823104
}
```

## Local dev

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # tweak ALLOWED_ORIGINS for your dev URL
uvicorn app:app --reload --port 8000
```

Smoke test:

```bash
curl -s http://localhost:8000/health | jq

curl -s -X POST http://localhost:8000/render \
  -H 'Content-Type: application/json' \
  -d '{"bbox":[-85,15,-65,30],"time":"latest","channel":13,"enhancement":"tat_neon"}' \
  -o /tmp/test.png
file /tmp/test.png  # should report PNG image
```

## Deploy to Railway

```bash
railway login
railway init             # creates project, links this directory
railway up               # builds via nixpacks, deploys
railway domain           # mints a public *.up.railway.app URL
```

Set env vars in the Railway dashboard (Variables tab):
- `ALLOWED_ORIGINS=https://triple-a-tropics.com,https://www.triple-a-tropics.com`
- (optional) `MAX_CONCURRENT_RENDERS`, `RATE_LIMIT`, `LATEST_CACHE_TTL`

Healthcheck path is wired in `railway.json` → `/health`.

## Architecture

| Module | Job |
| --- | --- |
| `app.py` | FastAPI, CORS, slowapi rate limit, asyncio.Semaphore for concurrency, JSON logging |
| `goes.py` | s3fs listing of `noaa-goes{16,19}`, time-based bucket picker, product selection (Meso → CONUS → Full Disk by bbox area), geos-projection bbox crop |
| `render.py` | matplotlib + cartopy PlateCarree pipeline, dark theme, dashed gridlines, title/footer |
| `colormaps.py` | `tat_neon`, `dvorak_bd`, `grayscale` |
| `cache.py` | byte-budgeted LRU keyed by hash(bbox, snapped_time, channel, enhancement) |

### Selection logic

**Bucket** (per requested time):

| When | Buckets tried |
| --- | --- |
| `"latest"` or t >= 2025-04-04 | `noaa-goes19` |
| 2018-08-01 ≤ t < 2025-04-04 | `noaa-goes19` → `noaa-goes16` (fallback) |
| t < 2018-08-01 | `noaa-goes16` |

`GOES_BUCKET` env var, when set, overrides the picker and forces a single bucket for every render. The render's title strip and footer reflect the actual bucket used (`GOES-19` vs `GOES-16`).

**Product** (per bbox area, same logic in every bucket):

| bbox area | Tries (in order) |
| --- | --- |
| < 30 sqdeg | Mesoscale M1, Mesoscale M2 (after coverage check), CONUS, Full Disk |
| 30–200 sqdeg | CONUS (if overlaps), Full Disk |
| > 200 sqdeg | Full Disk |

Mesoscale coverage check reads only the NetCDF global attrs (`geospatial_lat_min/max`, `geospatial_lon_min/max`) — no full file download. If the requested bbox sits inside the sector with a 0.5° buffer, we use it.

### RAM safety

- s3fs.open + xarray with `chunks={x:2048, y:2048}` — file is read lazily.
- `latlon_to_xy` projects bbox corners to ABI scan-angle space.
- `ds.isel(x=slice, y=slice)` slices BEFORE `.load()` materializes CMI.
- `gc.collect()` after each render.
- `asyncio.Semaphore(2)` caps simultaneous renders.

## Free-tier limits

- 200 cache entries / 100 MB total memory cache.
- 10 renders/min/IP (slowapi, X-Forwarded-For aware).
- 2 concurrent renders.
- "Latest" results cached for 5 minutes.

## Monitoring

```bash
railway logs           # JSON-formatted log lines
railway logs --tail    # follow
```
