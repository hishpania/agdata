"""Climate datasets: NASA POWER, NMME forecasts, PRISM and CoCoRaHS."""

import asyncio
import io
import re
from typing import ClassVar, Sequence

import numpy as np
import pandas as pd
from scipy.io import netcdf

from agdata import base


class NasaPower(base.Dataset):
    """NASA POWER agroclimatology: temperature, humidity, radiation, rain."""

    BASE_URL = 'https://power.larc.nasa.gov/api/temporal/'
    PARAMETERS: ClassVar[tuple[str, ...]] = (
        'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOTCORR',
        'ALLSKY_SFC_SW_DWN', 'WS2M', 'EVPTRNS')
    _FORMATS: ClassVar[dict[str, tuple[str, str]]] = {  # (request, index)
        'hourly': ('%Y%m%d', '%Y%m%d%H'),
        'daily': ('%Y%m%d', '%Y%m%d'),
        'monthly': ('%Y', ''),
    }
    # Regional requests: one parameter, 2 to 10 degrees per side; grid
    # nodes every 0.5 deg latitude and 0.625 deg longitude.
    _REGION: ClassVar[tuple[float, float]] = (2, 10)
    _CELL: ClassVar[tuple[float, float]] = (0.5, 0.625)

    async def point(self, lat: float, lon: float, start, end,
                    parameters: Sequence[str] | None = None,
                    resolution: str = 'daily') -> pd.DataFrame:
        """Returns a time series of POWER parameters at a point.

        Args:
            lat: Latitude.
            lon: Longitude.
            start: Start date.
            end: End date.
            parameters: POWER parameter codes; defaults to `PARAMETERS`.
            resolution: 'hourly', 'daily' or 'monthly' (month 13 = annual).

        Returns:
            One column per parameter, indexed by time; units in `.attrs`.
        """
        if resolution not in self._FORMATS:
            raise ValueError(f'Resolution not in {list(self._FORMATS)}.')
        request_fmt, index_fmt = self._FORMATS[resolution]
        return await self._fetch(f'{resolution}/point', lat, lon, parameters, {
            'start': base.date(start, request_fmt),
            'end': base.date(end, request_fmt),
        }, index_fmt)

    async def climatology(self, lat: float, lon: float,
                          parameters: Sequence[str] | None = None
                          ) -> pd.DataFrame:
        """Returns long-term monthly means (JAN..DEC, ANN) at a point."""
        return await self._fetch('climatology/point', lat, lon, parameters,
                                 {})

    async def area(self, bbox: base.Bbox, start, end,
                   parameters: Sequence[str] | None = None,
                   resolution: str = 'daily') -> pd.DataFrame:
        """Returns POWER parameters at each grid node of a bbox.

        Args:
            bbox: (west, south, east, north), up to 10 degrees per side.
            start: Start date.
            end: End date.
            parameters: POWER parameter codes; defaults to `PARAMETERS`.
            resolution: 'daily' or 'monthly' (month 13 = annual).

        Returns:
            Columns time, lat, lon and one per parameter; units in `.attrs`.
        """
        if resolution not in ('daily', 'monthly'):
            raise ValueError("Resolution not in ['daily', 'monthly'].")
        request_fmt, index_fmt = self._FORMATS[resolution]
        return await self._regional(f'{resolution}/regional', bbox,
                                    parameters, {
            'start': base.date(start, request_fmt),
            'end': base.date(end, request_fmt),
        }, index_fmt)

    async def climatology_area(self, bbox: base.Bbox,
                               parameters: Sequence[str] | None = None
                               ) -> pd.DataFrame:
        """Returns long-term monthly means at each grid node of a bbox."""
        return await self._regional('climatology/regional', bbox, parameters,
                                    {})

    async def _regional(self, path, bbox, parameters, params, index_fmt=''):
        west, south, east, north = base.check_bbox(bbox)
        smallest, largest = self._REGION
        if max(east - west, north - south) > largest:
            raise ValueError(f'POWER regions span at most {largest} degrees.')
        # Small boxes are widened for the API, then cut back.
        lat_min, lat_max = _widen(south, north, smallest, 90)
        lon_min, lon_max = _widen(west, east, smallest, 180)
        names = [base.check_name(p) for p in parameters or self.PARAMETERS]

        async def fetch(name: str) -> tuple[pd.Series, str]:
            data = await self._json('GET', path, params={
                **params, 'latitude-min': lat_min, 'latitude-max': lat_max,
                'longitude-min': lon_min, 'longitude-max': lon_max,
                'community': 'AG', 'parameters': name, 'format': 'JSON'})
            series = pd.concat({
                tuple(f['geometry']['coordinates'][1::-1]):
                    pd.Series(f['properties']['parameter'][name], dtype=float)
                for f in data['features']}, names=['lat', 'lon', 'time'])
            series = series.replace(data['header']['fill_value'], float('nan'))
            return series.rename(name), data['parameters'][name]['units']

        results = await asyncio.gather(*map(fetch, names))
        frame = pd.concat([series for series, _ in results],
                          axis=1).reset_index()
        half_lat, half_lon = self._CELL[0] / 2, self._CELL[1] / 2
        frame = frame[
            frame.lat.between(south - half_lat, north + half_lat)
            & frame.lon.between(west - half_lon, east + half_lon)]
        if index_fmt:
            frame['time'] = pd.to_datetime(frame['time'], format=index_fmt)
        frame = frame[['time', 'lat', 'lon', *names]].reset_index(drop=True)
        frame.attrs = {name: units for name, (_, units) in zip(names, results)}
        return frame

    async def _fetch(self, path, lat, lon, parameters, params, index_fmt=''):
        lat, lon = base.check_point(lat, lon)
        names = [base.check_name(p) for p in parameters or self.PARAMETERS]
        data = await self._json('GET', path, params={
            **params, 'latitude': lat, 'longitude': lon, 'community': 'AG',
            'parameters': ','.join(names), 'format': 'JSON'})
        frame = pd.DataFrame(data['properties']['parameter'])
        frame = frame.replace(data['header']['fill_value'], float('nan'))
        if index_fmt:
            frame.index = pd.to_datetime(frame.index, format=index_fmt)
        frame.attrs = {k: v['units'] for k, v in data['parameters'].items()}
        return frame


class PRISM(base.Dataset):
    """PRISM US climate grids (Oregon State University).

    The web service serves zipped grids with per-file daily download limits,
    so access is left unimplemented for now.
    """

    BASE_URL = 'https://services.nacse.org/prism/data/get/us/'

    async def daily(self, variable: str, start, end) -> pd.DataFrame:
        """Daily grids of ppt, tmin, tmax, tmean, tdmean, vpdmin, vpdmax."""
        return base.bulk(variable, start, end)

    async def monthly(self, variable: str, start, end) -> pd.DataFrame:
        """Monthly grids."""
        return base.bulk(variable, start, end)

    async def normals(self, variable: str) -> pd.DataFrame:
        """30-year (1991-2020) normals."""
        return base.bulk(variable)


class CoCoRaHS(base.Dataset):
    """CoCoRaHS crowd-sourced daily precipitation and snow reports."""

    BASE_URL = 'https://data.cocorahs.org/export/exportreports.aspx'

    async def daily(self, state: str, start, end,
                    county: str | None = None) -> pd.DataFrame:
        """Returns daily reports of a state (and county), e.g. 'IA'.

        Precipitation 'T' means trace; 'NA' is missing.
        """
        params = {
            'ReportType': 'Daily', 'dtf': 1, 'Format': 'CSV',
            'State': base.check_name(state, r'[A-Za-z]{2}'),
            'ReportDateType': 'reportdate',
            'StartDate': base.date(start, '%m/%d/%Y'),
            'EndDate': base.date(end, '%m/%d/%Y'),
        }
        if county:
            params['County'] = base.check_name(county, r'[A-Za-z]{1,3}')
        return pd.read_csv(io.StringIO(await self._text('GET', params=params)),
                           skipinitialspace=True,
                           parse_dates=['ObservationDate'])


class NMME(base.Dataset):
    """NMME seasonal forecasts (NOAA CPC): ensemble-mean monthly anomalies.

    Defaults to NASA GMAO's GEOS-S2S model. A forecast is issued about the
    8th of each month for 9 target months, starting with the issue month,
    on a 1-degree global grid. Anomalies are relative to the model's own
    hindcast climatology.
    """

    BASE_URL = 'https://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/'
    MODEL: ClassVar[str] = 'NASA_GEOS5v2'
    VARIABLES: ClassVar[tuple[str, ...]] = ('prate', 'tmp2m')
    _CELL: ClassVar[float] = 1.0

    async def point(self, lat: float, lon: float, start, end,
                    variables: Sequence[str] | None = None,
                    model: str | None = None) -> pd.DataFrame:
        """Returns forecasts issued from `start` to `end` at a point.

        Args:
            lat: Latitude.
            lon: Longitude.
            start: First issue month.
            end: Last issue month.
            variables: CPC variables ('prate', 'tmp2m', 'tmpsfc');
                defaults to `VARIABLES`.
            model: CPC model, e.g. 'CFSv2' or 'NMME' (multi-model mean);
                defaults to `MODEL`.

        Returns:
            One row per issue and target month at the nearest grid node:
            issued, target, lead (months after issue) and one anomaly column
            per variable; units in `.attrs`. Months without a forecast of
            `model` are skipped and listed in `.attrs['missing']`.
        """
        lat, lon = base.check_point(lat, lon)
        node_lat, node_lon = round(lat), _wrap(round(lon))
        frame = await self._forecasts(
            start, end, variables, model,
            lambda lats, lons: (lats == node_lat, lons == node_lon))
        return frame.drop(columns=['lat', 'lon'])

    async def area(self, bbox: base.Bbox, start, end,
                   variables: Sequence[str] | None = None,
                   model: str | None = None) -> pd.DataFrame:
        """Returns forecasts at each grid node of a bbox.

        Args:
            bbox: (west, south, east, north).
            start: First issue month.
            end: Last issue month.
            variables: As in `point`.
            model: As in `point`.

        Returns:
            As `point`, plus lat and lon columns: grid nodes whose 1-degree
            cell intersects the bbox.
        """
        west, south, east, north = base.check_bbox(bbox)
        half = self._CELL / 2

        def select(lats, lons):
            return ((lats >= south - half) & (lats <= north + half),
                    (lons >= west - half) & (lons <= east + half))

        return await self._forecasts(start, end, variables, model, select)

    async def _forecasts(self, start, end, variables, model, select):
        model = base.check_name(model or self.MODEL)
        names = [base.check_name(v) for v in variables or self.VARIABLES]
        months = pd.period_range(pd.Timestamp(start), pd.Timestamp(end),
                                 freq='M')
        folders = {}  # Issue month -> folder named <YYYYMM><DDHH>.
        for found in re.finditer(r'href="((\d{4})(\d{2})\d{4})/"',
                                 await self._text('GET')):
            folders[pd.Period(f'{found[2]}-{found[3]}', 'M')] = found[1]

        async def issue(month: pd.Period) -> pd.DataFrame | None:
            if month not in folders:
                return None
            listing = await self._text('GET', f'{folders[month]}/')
            files = [f'{model}.{name}.{month:%Y%m}.ENSMEAN.anom.nc'
                     for name in names]
            if not all(f'href="{f}"' in listing for f in files):
                return None
            grids = await asyncio.gather(*(
                self._grid(f'{folders[month]}/{f}', select) for f in files))
            if any(g.target.min() != month.to_timestamp() for g in grids):
                raise base.DatasetError(f'NMME: {month} forecast does not '
                                        f'start in its issue month.')
            frame = pd.concat([g.set_index(['target', 'lat', 'lon']).value
                               .rename(name)
                               for g, name in zip(grids, names)], axis=1)
            frame = frame.reset_index().assign(issued=month.to_timestamp())
            frame.attrs = {n: g.attrs['value'] for g, n in zip(grids, names)}
            return frame

        results = dict(zip(months, await asyncio.gather(*map(issue, months))))
        found = [r for r in results.values() if r is not None]
        if not found:
            raise base.DatasetError(f'NMME: no {model} forecasts issued '
                                    f'from {months[0]} to {months[-1]}.')
        frame = pd.concat(found, ignore_index=True)
        frame['lead'] = ((frame.target.dt.year - frame.issued.dt.year) * 12
                         + frame.target.dt.month - frame.issued.dt.month)
        frame = frame[['issued', 'target', 'lead', 'lat', 'lon', *names]]
        # The newest units apply: early files label prate mm/s as 'mm/d'.
        frame.attrs = {**found[-1].attrs, 'missing': [
            str(m) for m, r in results.items() if r is None]}
        return frame

    async def _grid(self, path: str, select) -> pd.DataFrame:
        """Reads one netCDF-3 file: target, lat, lon, value rows."""
        body = await self._request('GET', path)
        try:
            with netcdf.netcdf_file(io.BytesIO(body), mmap=False) as nc:
                variables = nc.variables
                units = variables['target'].units.decode()
                targets = variables['target'].data.copy()
                lats = variables['lat'].data.astype(float)
                lons = _wrap(variables['lon'].data.astype(float))
                # Early (2018) files name the variable 'anom'.
                data = variables['fcst' if 'fcst' in variables else 'anom']
                values = data.data.astype(float)
                fill = getattr(data, '_FillValue', None)
                value_units = data.units.decode()
        except (KeyError, AttributeError, TypeError, ValueError,
                OSError) as error:
            raise base.DatasetError(f'NMME: unreadable netCDF {path}: '
                                    f'{error}') from error
        # Early files omit the epoch; `_forecasts` checks the dates.
        if units != 'months' and not units.startswith('months since 1960-01'):
            raise base.DatasetError(f'NMME: unexpected time units {units}.')
        if values.shape != (len(targets), len(lats), len(lons)):
            raise base.DatasetError(f'NMME: unexpected shape {values.shape}.')
        rows, cols = select(lats, lons)
        values = values[:, rows][:, :, cols]
        if fill is not None:
            values[values == fill] = np.nan
        months = np.rint(targets).astype(int)
        grid_lat, grid_lon = np.meshgrid(lats[rows], lons[cols], indexing='ij')
        frame = pd.DataFrame({
            'target': np.repeat([pd.Timestamp(1960 + m // 12, m % 12 + 1, 1)
                                 for m in months], grid_lat.size),
            'lat': np.tile(grid_lat.ravel(), len(months)),
            'lon': np.tile(grid_lon.ravel(), len(months)),
            'value': values.reshape(len(months), -1).ravel(),
        }).dropna()
        frame.attrs = {'value': value_units}
        return frame


def _wrap(lon):
    """Maps longitudes to [-180, 180)."""
    return (lon + 180) % 360 - 180


def _widen(low: float, high: float, size: float,
           limit: float) -> tuple[float, float]:
    """Widens [low, high] to at least `size`, within [-limit, limit]."""
    pad = max(0.0, size - (high - low)) / 2
    low, high = low - pad, high + pad
    shift = max(0.0, -limit - low) - max(0.0, high - limit)
    return low + shift, high + shift
