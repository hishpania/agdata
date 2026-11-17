"""Core abstraction shared by all datasets."""

import asyncio
import functools
import io
import json
import logging
import math
import os
import re
import weakref
from typing import Any, ClassVar, Iterable
from urllib import parse

import aiohttp
import numpy as np
import pandas as pd
import pyproj
import tifffile

# (west, south, east, north) in decimal degrees.
Bbox = tuple[float, float, float, float]
# Keyword arguments carrying credentials: 'headers', 'params' and/or 'auth'.
Auth = dict[str, Any]

# WGS84 to CONUS Albers (EPSG:5070), the grid of NLCD and CDL.
TO_ALBERS = pyproj.Transformer.from_crs(4326, 5070, always_xy=True)

_TRANSIENT = frozenset({429, 502, 503, 504})
_AUTH: dict[type, Auth] = {}  # Authorization cache, one entry per class.
# Per event loop: authorization locks by class, request slots by host.
_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[type, asyncio.Lock]
] = weakref.WeakKeyDictionary()
_SLOTS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str | None, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()

# tifffile warns on valid float GDAL_NODATA strings ('0.0') of integer
# rasters; `_raster` reads that tag itself.
logging.getLogger('tifffile').addFilter(
    lambda record: 'GDAL_NODATA' not in record.getMessage())


class DatasetError(RuntimeError):
    """A dataset API failed or answered unexpectedly."""


class Dataset:
    """A remote dataset accessed asynchronously through its web API.

    Pass an `aiohttp.ClientSession` to share connections; otherwise the
    instance opens its own, closed by `close()` or `async with`.

    Subclasses set `BASE_URL` and, when the API needs authorization, list
    the environment variables holding credentials in `CREDENTIALS` and
    override `_authorize`. Authorization runs once per class, on first use.
    Credentials are only sent over HTTPS to the `BASE_URL` host.
    """

    BASE_URL: ClassVar[str] = ''
    CREDENTIALS: ClassVar[tuple[str, ...]] = ()
    CREDENTIALS_OPTIONAL: ClassVar[bool] = False
    TIMEOUT: ClassVar[float] = 120
    MAX_BYTES: ClassVar[int] = 200 * 2**20  # Largest accepted response.
    MAX_PIXELS: ClassVar[int] = 50_000_000  # Largest accepted raster.
    CONCURRENCY: ClassVar[int] = 4  # Simultaneous requests per host.
    RETRIES: ClassVar[int] = 3  # GET retries on transient HTTP errors.

    _credentials: ClassVar[tuple[str, ...] | None] = None

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._http = session
        self._owns_http = session is None

    async def __aenter__(self) -> 'Dataset':
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Closes the HTTP session if this instance created it."""
        if self._owns_http and self._http is not None:
            await self._http.close()
            self._http = None

    @classmethod
    def login(cls, *credentials: str) -> None:
        """Sets credentials for this class and its subclasses.

        Overrides the `CREDENTIALS` environment variables.
        """
        cls._credentials = credentials
        for klass in [cls, *_subclasses(cls)]:
            _AUTH.pop(klass, None)

    @classmethod
    async def _authorize(cls, http: aiohttp.ClientSession,
                         credentials: tuple[str, ...]) -> Auth:
        """Returns request arguments carrying `credentials`."""
        del http, credentials
        return {}

    def _session(self) -> aiohttp.ClientSession:
        if self._http is None:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _auth(self) -> Auth:
        """Returns the class authorization, obtaining it once."""
        cls = type(self)
        locks = _LOCKS.setdefault(asyncio.get_running_loop(), {})
        async with locks.setdefault(cls, asyncio.Lock()):
            if cls not in _AUTH:
                credentials = self._credentials or tuple(
                    os.environ.get(name) for name in cls.CREDENTIALS)
                if credentials and all(credentials):
                    _AUTH[cls] = await self._authorize(self._session(),
                                                       credentials)
                elif cls.CREDENTIALS and not cls.CREDENTIALS_OPTIONAL:
                    raise PermissionError(
                        f'{cls.__name__} needs credentials: call '
                        f'{cls.__name__}.login() or set {cls.CREDENTIALS}.')
                else:
                    _AUTH[cls] = {}
        return _AUTH[cls]

    async def _request(self, method: str, path: str = '', *,
                       public: bool = False, **kwargs: Any) -> bytes:
        """Sends a request and returns the response body.

        Args:
            method: HTTP method.
            path: Path under `BASE_URL`, or an absolute URL.
            public: Sends no credentials; allows any HTTPS host.
            **kwargs: `aiohttp` request arguments.
        """
        url = check_url(path if '://' in path else self.BASE_URL + path)
        if not public and _host(url) != _host(self.BASE_URL):
            raise DatasetError(f'Credentials refused for host {_host(url)}.')
        status, body = await self._send(method, url, public, kwargs)
        if status == 401 and not public and _AUTH.get(type(self)):
            _AUTH.pop(type(self), None)  # Expired token: authorize again.
            status, body = await self._send(method, url, public, kwargs)
        for attempt in range(self.RETRIES if method == 'GET' else 0):
            if status not in _TRANSIENT:
                break
            await asyncio.sleep(2 ** attempt)
            status, body = await self._send(method, url, public, kwargs)
        if status >= 400:
            raise DatasetError(
                f'{type(self).__name__}: HTTP {status} from {_strip(url)}: '
                f'{body[:300].decode(errors="replace")}')
        return body

    async def _send(self, method, url, public, kwargs) -> tuple[int, bytes]:
        auth = {} if public else await self._auth()
        kwargs = {'timeout': aiohttp.ClientTimeout(total=self.TIMEOUT),
                  **_merge(kwargs, auth)}
        slots = _SLOTS.setdefault(asyncio.get_running_loop(), {})
        async with slots.setdefault(_host(url),
                                    asyncio.Semaphore(self.CONCURRENCY)):
            try:
                async with self._session().request(method, url,
                                                   **kwargs) as resp:
                    body = bytearray()
                    async for chunk in resp.content.iter_chunked(2**16):
                        body += chunk
                        if len(body) > self.MAX_BYTES:
                            raise DatasetError(
                                f'Response over {self.MAX_BYTES} B.')
                    return resp.status, bytes(body)
            except TimeoutError:
                if method != 'GET':
                    raise
                return 504, b'Timed out.'  # Retried as transient.

    async def _text(self, method: str, path: str = '', **kwargs: Any) -> str:
        return (await self._request(method, path, **kwargs)).decode(
            errors='replace')

    async def _json(self, method: str, path: str = '', **kwargs: Any) -> Any:
        body = await self._request(method, path, **kwargs)
        try:
            return json.loads(body)
        except ValueError as error:
            raise DatasetError(f'{type(self).__name__}: not JSON: '
                               f'{body[:200].decode(errors="replace")}'
                               ) from error

    async def _raster(self, method: str, path: str = '', *,
                      bbox: Bbox | None = None, nodata: Iterable[float] = (),
                      **kwargs: Any) -> pd.DataFrame:
        """Downloads a single-band, north-up GeoTIFF.

        Args:
            method: HTTP method.
            path: As in `_request`.
            bbox: Keeps only pixels centered in this (west, south, east,
                north) box.
            nodata: Values marking missing pixels, besides the file's own.
            **kwargs: `_request` arguments.

        Returns:
            One row per valid pixel: lat, lon (pixel center) and value.
        """
        body = await self._request(method, path, **kwargs)
        try:
            with tifffile.TiffFile(io.BytesIO(body)) as tif:
                page = tif.pages.first
                if len(page.shape) != 2:
                    raise DatasetError(f'Not a single-band raster: '
                                       f'{page.shape}.')
                if math.prod(page.shape) > self.MAX_PIXELS:
                    raise DatasetError(f'Raster over {self.MAX_PIXELS} px.')
                geo, pixels = page.geotiff_tags or {}, page.asarray()
                gdal = page.tags.get('GDAL_NODATA')
        except (tifffile.TiffFileError, ValueError, KeyError) as error:
            raise DatasetError(f'{type(self).__name__}: unreadable GeoTIFF: '
                               f'{error}') from error
        missing = [*nodata, *([float(gdal.value)] if gdal else [])]
        valid = ~np.isin(pixels, missing)
        if pixels.dtype.kind == 'f':
            valid &= ~np.isnan(pixels)
        rows, cols = np.nonzero(valid)
        x0, dx, y0, dy = _georeference(geo)
        lon, lat = _to_wgs84(_epsg(geo)).transform(x0 + cols * dx,
                                                   y0 + rows * dy)
        frame = pd.DataFrame({'lat': lat, 'lon': lon, 'value': pixels[valid]})
        if bbox is not None:
            west, south, east, north = check_bbox(bbox)
            frame = frame[frame.lat.between(south, north)
                          & frame.lon.between(west, east)]
        return frame.reset_index(drop=True)


def check_point(lat: Any, lon: Any) -> tuple[float, float]:
    """Returns valid WGS84 coordinates as floats."""
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f'Invalid coordinates: {lat}, {lon}.')
    return lat, lon


def check_bbox(bbox: Any) -> Bbox:
    """Returns a valid (west, south, east, north) bbox of floats."""
    west, south, east, north = map(float, bbox)
    check_point(south, west)
    check_point(north, east)
    if not (west < east and south < north):
        raise ValueError(f'Invalid bbox: {bbox}.')
    return west, south, east, north


def check_name(value: Any, pattern: str = r'[\w.-]+') -> str:
    """Returns `value` as text if it fully matches `pattern`."""
    value = str(value)
    if not re.fullmatch(pattern, value) or '..' in value:
        raise ValueError(f'Invalid identifier: {value!r}.')
    return value


def check_url(url: str, hosts: set[str] | None = None) -> str:
    """Returns `url` if it is HTTPS (and, if given, on one of `hosts`)."""
    parts = parse.urlsplit(url)
    if parts.scheme != 'https' or (hosts and parts.hostname not in hosts):
        raise DatasetError(f'Untrusted URL: {_strip(url)}')
    return url


def ring(bbox: Any) -> list[list[float]]:
    """Returns the closed (lon, lat) outline of a bbox, counterclockwise."""
    west, south, east, north = check_bbox(bbox)
    return [[west, south], [east, south], [east, north], [west, north],
            [west, south]]


def grid(bbox: Any, step: float) -> list[tuple[float, float]]:
    """Returns (lat, lon) centers of `step`-degree cells covering a bbox.

    Cells are aligned to multiples of `step`, as in most global grids.
    """
    west, south, east, north = check_bbox(bbox)

    def centers(low: float, high: float) -> np.ndarray:
        first = math.floor(low / step + 1e-9)
        last = max(math.ceil(high / step - 1e-9), first + 1)
        return np.round((np.arange(first, last) + 0.5) * step, 6)

    return [(float(lat), float(lon)) for lat in centers(south, north)
            for lon in centers(west, east)]


def sql_text(value: Any) -> str:
    """Quotes a value as an SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def date(value: Any, fmt: str = '%Y-%m-%d') -> str:
    """Formats any date-like value (str, datetime, Timestamp) with `fmt`."""
    return pd.Timestamp(value).strftime(fmt)


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Converts the columns holding only numbers (as text) to numbers."""
    for column in frame:
        try:
            frame[column] = pd.to_numeric(frame[column])
        except (ValueError, TypeError):
            pass
    return frame


def bulk(*_: Any, **__: Any) -> pd.DataFrame:
    """Placeholder for rate-limited endpoints that need bulk downloads."""
    raise NotImplementedError('Bulk, rate-limited download: not implemented.')


def _georeference(geo: dict) -> tuple[float, float, float, float]:
    """Returns x0, dx, y0, dy: first pixel center and pixel steps."""
    if 'ModelTransformation' in geo:
        matrix = np.asarray(geo['ModelTransformation'], float).reshape(4, 4)
        if matrix[0, 1] or matrix[1, 0]:
            raise DatasetError('Rotated rasters are not supported.')
        dx, x0, dy, y0 = matrix[0, 0], matrix[0, 3], matrix[1, 1], matrix[1, 3]
    elif 'ModelPixelScale' in geo and 'ModelTiepoint' in geo:
        (dx, dy), (i, j, _, x, y) = (geo['ModelPixelScale'][:2],
                                     geo['ModelTiepoint'][:5])
        dy = -dy
        x0, y0 = x - i * dx, y - j * dy
    else:
        raise DatasetError('GeoTIFF without georeference.')
    # Unless the raster is PixelIsPoint (2), (x0, y0) is a pixel corner.
    half = 0 if int(geo.get('GTRasterTypeGeoKey', 1)) == 2 else 0.5
    return x0 + half * dx, dx, y0 + half * dy, dy


def _epsg(geo: dict) -> int:
    code = geo.get('ProjectedCSTypeGeoKey') or geo.get('GeographicTypeGeoKey')
    if code is None or not 1024 <= int(code) < 32767:
        raise DatasetError(f'GeoTIFF without EPSG code: {code}.')
    return int(code)


@functools.cache
def _to_wgs84(epsg: int) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)


def _merge(kwargs: dict[str, Any], auth: Auth) -> dict[str, Any]:
    merged = {**kwargs, **auth}
    for key in ('headers', 'params'):
        if kwargs.get(key) and key in auth:
            merged[key] = {**kwargs[key], **auth[key]}
    return merged


def _host(url: str) -> str | None:
    return parse.urlsplit(url).hostname


def _strip(url: str) -> str:
    """Drops the query string, which may carry API keys."""
    return parse.urlsplit(url)._replace(query='').geturl()


def _subclasses(cls: type) -> list[type]:
    return [s for c in cls.__subclasses__() for s in [c, *_subclasses(c)]]
