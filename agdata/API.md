# agdata API

Unified, object-oriented, asynchronous access to agricultural and Earth-observation datasets.
Every accessor is a coroutine that queries the dataset's own web API and returns a `pandas.DataFrame`.

```bash
pip install -r agdata/requirements.txt
```

```python
import asyncio
import aiohttp
import agdata as ag


async def main():
    async with aiohttp.ClientSession() as http:         # Shared session.
        weather, crops, soil = await asyncio.gather(
            ag.NasaPower(http).point(42.0, -93.6, '2024-06-01', '2024-06-30'),
            ag.CropScape(http).crops(42.0, -93.6, range(2015, 2024)),
            ag.SSURGO(http).properties(42.0, -93.6))

    async with ag.NLCD() as nlcd:                        # Own session.
        cover = await nlcd.land_cover(42.0, -93.6)
        field = await nlcd.land_cover_area((-93.62, 42.0, -93.6, 42.01))

asyncio.run(main())                                      # Jupyter: await main()
```

## Sessions

- `Dataset(session=None)`: pass an `aiohttp.ClientSession` to share it. You keep ownership, and the library never modifies it.
- Without a session, the instance opens its own. Close it with `await ds.close()` or `async with`.

## Conventions

- Coordinates are WGS84 decimal degrees: `lat, lon`; `bbox = (west, south, east, north)`.
- Every point accessor `name(lat, lon, ...)` has an area twin `name_area(bbox, ...)` (`point` → `area`) taking mostly the same
  other arguments. Area results have `lat, lon` columns (pixel or grid-cell centers), except aggregates (`OpenET.area`).
- Dates accept anything `pandas.Timestamp` parses (`'2024-06-01'`, `datetime`, ...).
- Units and metadata, when the API provides them, are in `df.attrs`.
- Invalid arguments raise `ValueError`. API failures raise `ag.DatasetError`. Missing credentials raise `PermissionError`.
- Methods marked **bulk** raise `NotImplementedError`: the endpoint needs rate-limited bulk downloads.

## Authorization

Each class authorizes once, on first use. The result is cached and attached to each request, not to the session.
After an HTTP 401, the class authorizes once more and repeats the request (expired tokens).
Credentials come from environment variables or `Class.login(...)`, which also applies to subclasses.

| Classes | Credentials (`login` args / env vars) | Sign up |
|---|---|---|
| All NASA Earthdata classes | `Earthdata.login(user, password)` / `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` | urs.earthdata.nasa.gov |
| `SRTM` | `OPENTOPOGRAPHY_API_KEY` | portal.opentopography.org |
| `QuickStats` | `NASS_API_KEY` | quickstats.nass.usda.gov/api |
| `MarketNews` | `AMS_API_KEY` | mymarketnews.ams.usda.gov |
| `OpenET` | `OPENET_API_KEY` | etdata.org |
| `NWIS` (optional, raises limits) | `USGS_API_KEY` | api.waterdata.usgs.gov |
| Others | none | |

`granules()` (CMR search) needs no credentials.

## Safety

- HTTPS only. Credentials are sent only to the dataset's own host, and are stripped on redirects to other hosts.
- Error messages omit query strings, which may hold API keys.
- Responses are capped at `MAX_BYTES` (200 MB), rasters at `MAX_PIXELS` (50 M). Each host gets at most `CONCURRENCY` (4) simultaneous requests.
- GET requests are retried `RETRIES` (3) times, with backoff, on timeouts and HTTP 429/502/503/504.
- Coordinates, identifiers and URLs are validated. SQL values are quoted.
- Download and redirect URLs taken from responses must stay on the expected hosts.
- No pickle, `eval` or `exec`. Data is parsed only as JSON, CSV, plain text, GeoTIFF (`tifffile`) or netCDF-3 (`scipy`).

All of these limits are class attributes, so subclasses can override them (e.g. `TIMEOUT`).

## Datasets

### NASA Earthdata (`agdata.earthdata`)

All have `granules(bbox, start, end, short_name=None, limit=2000)`, which lists data files (title, time range, size, URL).

| Class | Accessors | Default product / layers |
|---|---|---|
| `SMAP` | `point(lat, lon, start, end, layers=None, product=None)`, `area(bbox, ...)`, `freeze_thaw(lat, lon, start, end)`, `freeze_thaw_area(bbox, start, end)` | SPL3SMP_E.006 AM/PM soil moisture; SPL3FTP.004 |
| `Landsat` | `point(...)`, `area(...)` | HLSL30_VI.020: NDVI, EVI, NDMI |
| `MODIS` | `point(...)`, `area(...)` | MOD13Q1.061: NDVI, EVI (250 m, 16-day) |
| `VIIRS` | `point(...)`, `area(...)` | VNP13A1.002: NDVI, EVI (500 m, 16-day) |
| `ECOSTRESS` | `point(...)`, `area(...)`, `evapotranspiration(lat, lon, start, end)`, `evapotranspiration_area(bbox, start, end)` | ECO_L2T_LSTE.002 LST; ECO_L3T_JET.002 ETdaily |
| `GPM` (`IMERG`) | `timeseries(lat, lon, start, end, variable=None)`, `timeseries_area(bbox, ...)` | GPM_3IMERGHH_07 precipitation (0.1°) |
| `NLDAS` | `timeseries(...)`, `timeseries_area(...)` | NLDAS_FORA0125_H_2_0 Tair (0.125°) |
| `GLDAS` | `timeseries(...)`, `timeseries_area(...)` | GLDAS_NOAH025_3H_2_1 SoilMoi0_10cm_inst (0.25°) |
| `GEDI` | `footprints(bbox, start, end)` **bulk** | GEDI L4A V3 biomass |

`point` and `area` run an AppEEARS task, polled for up to `MAX_WAIT` (1 h); it can take minutes.
`area` returns date, lat, lon and one column per layer, for every pixel (fill values dropped).
`timeseries` uses the Giovanni Time Series API, which serves points only: `timeseries_area` requests each
`RESOLUTION`-degree grid cell of the bbox, at most `MAX_POINTS` (256).

### Climate (`agdata.climate`)

| Class | Methods |
|---|---|
| `NasaPower` | `point(lat, lon, start, end, parameters=None, resolution='daily')` (hourly, daily or monthly), `climatology(lat, lon, parameters=None)`, `area(bbox, ...)` (daily or monthly), `climatology_area(bbox, ...)` |
| `NMME` | `point(lat, lon, start, end, variables=None, model=None)`, `area(bbox, ...)`: seasonal forecasts issued from `start` to `end` |
| `PRISM` | `daily(variable, start, end)`, `monthly(variable, start, end)`, `normals(variable)`: all **bulk** |
| `CoCoRaHS` | `daily(state, start, end, county=None)` |

`NasaPower` area methods use the regional API: grid nodes every 0.5° lat × 0.625° lon, bbox up to 10° per side,
one request per parameter.

`NMME` reads NOAA CPC's ensemble-mean monthly anomaly forecasts (1° grid, 9 target months from the issue month,
`lead` 0-8), by default of NASA GMAO's GEOS-S2S (`MODEL = 'NASA_GEOS5v2'`, from 2018-04) for
`VARIABLES = ('prate', 'tmp2m')` (mm/s, K). Each issue month downloads one 2.2 MB file per variable.
Months without a forecast are skipped and listed in `df.attrs['missing']`.

### Land, terrain, imagery (`agdata.land`, `agdata.arcgis`)

| Class | Methods |
|---|---|
| `SRTM` | `elevation(bbox, dem='SRTMGL1')`: lat, lon, elevation, slope |
| `NLCD` | `land_cover(lat, lon, years=NLCD.YEARS)`: code, class per year; `land_cover_area(bbox, years=2021)`: every 30 m pixel |
| `GAEZ` | `suitability(lat, lon, crop, rcp='Historical')`, `identify(lat, lon, where=None, layer='res05')`, `suitability_area(bbox, crop, rcp, samples=1024)`, `identify_area(bbox, where=None, layer=None, samples=1024)` |
| `LANDFIRE` | `identify(lat, lon, layer='Landfire_LF2024/LF2024_EVT_CONUS')`, `identify_area(bbox, ...)` |
| `NAIP` | `pixels(lat, lon)`: RGB, NIR, NDVI per acquisition; `pixels_area(bbox, samples=1024)` |

ArcGIS `identify_area` samples a regular grid (`samples` ≤ 1024) on each catalog raster intersecting the bbox and matching
`where`, at most `MAX_RASTERS` (250): narrow `where` (e.g. `sub_theme_name='Suitability Index'` for GAEZ) when needed.

### Water (`agdata.water`)

| Class | Methods |
|---|---|
| `NWIS` | `sites(bbox, site_type=None, **filters)`, `daily(site, start, end, parameter='00060')`, `daily_area(bbox, start, end, parameter='00060')`: all sites, `items(collection, **params)` |
| `OpenET` | `point(lat, lon, start, end, interval='monthly', model='Ensemble', variable='ET')`, `area(bbox, ..., reducer='mean')`: one series aggregated over the bbox |

### USDA (`agdata.usda`)

| Class | Methods |
|---|---|
| `QuickStats` | `query(**filters)`, `values(param, **filters)` |
| `MarketNews` | `reports()`, `report(slug_id, start=None, end=None, **filters)` |
| `CropScape` (`CDL`) | `crops(lat, lon, years)`, `crops_area(bbox, years)`: CDL code of every 30 m pixel, `stats(year, fips=None, bbox=None)` |
| `SSURGO`, `STATSGO2`, `WebSoilSurvey` | `properties(lat, lon)`, `ratings(lat, lon, rule=None)`, `properties_area(bbox)`, `ratings_area(bbox, rule=None)`: all map units intersecting the bbox, `query(sql)` |
| `AgDataCommons` | `search(query)`, `files(dataset_id)`, `read(download_url, **read_csv_kwargs)` |

## Extending

Subclass the closest base and set class attributes. For a new auth scheme, override the
`_authorize` coroutine to return the request arguments (`headers`, `params` or `auth`) that carry the credentials.

```python
import pandas as pd
import agdata as ag


class SMAPRootZone(ag.earthdata.AppEEARS):
    """SMAP L4 root-zone soil moisture."""
    SHORT_NAME = 'SPL4SMGP'
    PRODUCT = 'SPL4SMGP.008'
    LAYERS = ('Geophysical_Data_sm_rootzone',)


class MyApi(ag.Dataset):
    BASE_URL = 'https://example.org/api/'
    CREDENTIALS = ('MY_API_KEY',)

    @classmethod
    async def _authorize(cls, http, credentials):
        return {'headers': {'Authorization': credentials[0]}}

    async def records(self, **params) -> pd.DataFrame:
        return pd.DataFrame(await self._json('GET', 'records', params=params))
```

Base classes: `Dataset` (any API), `Earthdata` (CMR granules), `AppEEARS` (point extraction),
`Giovanni` (time series), `arcgis.ImageService` (ArcGIS ImageServer pixels),
`usda.SoilDataAccess` (soil SQL).
Request helpers: `_json`, `_text` and `_request` (bytes), each taking `(method, path, **aiohttp_kwargs)`, and
`_raster` (GeoTIFF to lat, lon, value rows; also `bbox=` and `nodata=`).
Geometry helpers in `agdata.base`: `check_bbox`, `ring(bbox)` (closed lon/lat outline), `grid(bbox, step)` (cell centers).
