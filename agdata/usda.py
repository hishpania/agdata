"""USDA datasets: NASS QuickStats, AMS Market News, CropScape (CDL),
NRCS soils (SSURGO, STATSGO2, Web Soil Survey) and Ag Data Commons."""

import asyncio
import io
import re
from typing import ClassVar, Iterable

import aiohttp
import pandas as pd

from agdata import base

_FIGSHARE_HOSTS = {'api.figshare.com', 'figshare.com',
                   'ndownloader.figshare.com'}


class QuickStats(base.Dataset):
    """USDA NASS QuickStats agricultural statistics."""

    BASE_URL = 'https://quickstats.nass.usda.gov/api/'
    CREDENTIALS = ('NASS_API_KEY',)

    @classmethod
    async def _authorize(cls, http, credentials):
        return {'params': {'key': credentials[0]}}

    async def query(self, **filters) -> pd.DataFrame:
        """Returns records matching QuickStats filters (max 50,000 rows).

        Example: query(commodity_desc='CORN', statisticcat_desc='YIELD',
        agg_level_desc='COUNTY', state_alpha='IA', year__GE=2015).
        """
        data = await self._json('GET', 'api_GET/',
                                 params={**filters, 'format': 'JSON'})
        return pd.DataFrame(data.get('data', []))

    async def values(self, param: str, **filters) -> pd.DataFrame:
        """Returns allowed values of a parameter, e.g. 'commodity_desc'."""
        param = base.check_name(param)
        data = await self._json('GET', 'get_param_values/',
                                params={**filters, 'param': param})
        return pd.DataFrame({param: data.get(param, [])})


class MarketNews(base.Dataset):
    """USDA AMS Market News (MARS) commodity price reports."""

    BASE_URL = 'https://marsapi.ams.usda.gov/services/v1.2/'
    CREDENTIALS = ('AMS_API_KEY',)

    @classmethod
    async def _authorize(cls, http, credentials):
        return {'auth': aiohttp.BasicAuth(credentials[0], '')}

    async def reports(self) -> pd.DataFrame:
        """Lists available reports; `slug_id` identifies each."""
        return pd.DataFrame(await self._json('GET', 'reports'))

    async def report(self, slug_id: str | int, start=None, end=None,
                     **filters) -> pd.DataFrame:
        """Returns a report's records, optionally within report dates.

        Args:
            slug_id: Report id from `reports()`.
            start: First report date.
            end: Last report date.
            **filters: Field filters, e.g. commodity='Corn'.
        """
        if start:
            filters['report_begin_date'] = (
                f'{base.date(start, "%m/%d/%Y")}:'
                f'{base.date(end or pd.Timestamp.now(), "%m/%d/%Y")}')
        for key, value in filters.items():
            base.check_name(key)
            if re.search(r'[;=]', str(value)):
                raise ValueError(f'Invalid filter value: {value!r}.')
        query = ';'.join(f'{k}={v}' for k, v in filters.items())
        data = await self._json('GET', f'reports/{base.check_name(slug_id)}',
                                params={'q': query})
        return pd.DataFrame(data.get('results', [])
                            if isinstance(data, dict) else data)


class CropScape(base.Dataset):
    """USDA NASS Cropland Data Layer (30 m annual crop types) via CropScape."""

    BASE_URL = 'https://nassgeodata.gmu.edu/axis2/services/CDLService/'
    TIMEOUT = 30  # Usually answers in seconds but stalls at times.
    FILE_TIMEOUT: ClassVar[float] = 300  # Clipping rasters takes longer.

    async def crops(self, lat: float, lon: float,
                    years: int | Iterable[int]) -> pd.DataFrame:
        """Returns the crop at a point for each year (rotation history)."""
        lat, lon = base.check_point(lat, lon)
        x, y = base.TO_ALBERS.transform(lon, lat)
        years = [int(years)] if isinstance(years, int) else map(int, years)

        async def crop(year: int) -> dict:
            text = await self._text('GET', 'GetCDLValue',
                                    params={'year': year, 'x': x, 'y': y})
            fields = dict(re.findall(r'(\w+): "?([^",}]*)"?', text))
            if 'value' not in fields:
                raise base.DatasetError(f'No CDL value: {text[:200]}')
            return {'year': year, 'value': int(fields['value']),
                    'category': fields.get('category')}

        return pd.DataFrame(await asyncio.gather(*map(crop, years)))

    async def crops_area(self, bbox: base.Bbox,
                         years: int | Iterable[int]) -> pd.DataFrame:
        """Returns the CDL crop code of every 30 m pixel in a bbox.

        Returns:
            Columns year, lat, lon, value (CDL code, as in `crops`).
        """
        albers = ','.join(map(str, base.TO_ALBERS.transform_bounds(
            *base.check_bbox(bbox))))
        years = [int(years)] if isinstance(years, int) else map(int, years)
        timeout = aiohttp.ClientTimeout(total=self.FILE_TIMEOUT)

        async def crop(year: int) -> pd.DataFrame:
            url = await self._return_url('GetCDLFile', timeout=timeout,
                                         params={'year': year, 'bbox': albers})
            frame = await self._raster('GET', url, bbox=bbox, nodata=[0],
                                       timeout=timeout)
            return frame.assign(year=year)

        frame = pd.concat(await asyncio.gather(*map(crop, years)),
                          ignore_index=True)
        return frame[['year', 'lat', 'lon', 'value']]

    async def stats(self, year: int, fips: str | None = None,
                    bbox: base.Bbox | None = None) -> pd.DataFrame:
        """Returns crop acreage for a state/county FIPS code or a bbox."""
        params = {'year': int(year), 'format': 'csv'}
        if fips:
            params['fips'] = base.check_name(fips, r'\d{2}|\d{5}')
        else:
            west, south, east, north = base.check_bbox(bbox)
            west, south = base.TO_ALBERS.transform(west, south)
            east, north = base.TO_ALBERS.transform(east, north)
            params['bbox'] = f'{west},{south},{east},{north}'
        url = await self._return_url('GetCDLStat', params=params)
        return pd.read_csv(io.StringIO(await self._text('GET', url)),
                           skipinitialspace=True)

    async def _return_url(self, operation: str, **kwargs) -> str:
        """Runs a CropScape operation that answers with a file URL."""
        xml = await self._text('GET', operation, **kwargs)
        found = re.search(r'<returnURL>(.*?)</returnURL>', xml)
        if not found:
            raise base.DatasetError(f'No CDL {operation} file: {xml[:200]}')
        # Fetched as non-public requests: only the CropScape host is accepted.
        return found[1]


class SoilDataAccess(base.Dataset):
    """NRCS Soil Data Access: SQL over the US soil survey databases."""

    BASE_URL = 'https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest'
    MUKEYS: ClassVar[str] = ''  # SQL selecting map unit keys in {wkt}.

    async def query(self, sql: str) -> pd.DataFrame:
        """Runs a read-only T-SQL query against Soil Data Access."""
        data = await self._json(
            'POST', json={'query': sql, 'format': 'JSON+COLUMNNAME'})
        table = data.get('Table', [])
        return base.numeric(
            pd.DataFrame(table[1:], columns=table[0] if table else None))

    async def properties(self, lat: float, lon: float) -> pd.DataFrame:
        """Returns components and horizon properties at a point.

        Includes texture, organic matter, available water capacity, Ksat,
        erodibility, pH, drainage class, hydrologic group and slope.
        """
        return await self._properties(_point(lat, lon))

    async def properties_area(self, bbox: base.Bbox) -> pd.DataFrame:
        """Returns `properties` of every map unit intersecting a bbox."""
        return await self._properties(_polygon(bbox))

    async def ratings(self, lat: float, lon: float,
                      rule: str | None = None) -> pd.DataFrame:
        """Returns soil interpretations (suitabilities, limitations).

        Args:
            lat: Latitude.
            lon: Longitude.
            rule: Substring of the rule name to keep, e.g. 'Erosion'.
        """
        return await self._ratings(_point(lat, lon), rule)

    async def ratings_area(self, bbox: base.Bbox,
                           rule: str | None = None) -> pd.DataFrame:
        """Returns `ratings` of every map unit intersecting a bbox."""
        return await self._ratings(_polygon(bbox), rule)

    async def _properties(self, wkt: str) -> pd.DataFrame:
        return await self.query(f"""
            SELECT mu.mukey, mu.muname, c.cokey, c.compname, c.comppct_r,
                   c.slope_r, c.drainagecl, c.hydgrp, c.tfact, c.taxorder,
                   h.hzname, h.hzdept_r, h.hzdepb_r, h.sandtotal_r,
                   h.silttotal_r, h.claytotal_r, h.om_r, h.awc_r, h.ksat_r,
                   h.kwfact, h.ph1to1h2o_r, h.dbthirdbar_r
            FROM mapunit mu
            JOIN component c ON c.mukey = mu.mukey
            LEFT JOIN chorizon h ON h.cokey = c.cokey
            WHERE mu.mukey IN ({self.MUKEYS.format(wkt=wkt)})
            ORDER BY mu.mukey, c.comppct_r DESC, h.hzdept_r""")

    async def _ratings(self, wkt: str, rule: str | None) -> pd.DataFrame:
        where = (f'AND i.mrulename LIKE {base.sql_text(f"%{rule}%")}'
                 if rule else '')
        return await self.query(f"""
            SELECT c.mukey, c.compname, c.comppct_r, i.mrulename,
                   i.interphrc, i.interphr
            FROM component c
            JOIN cointerp i ON i.cokey = c.cokey AND i.ruledepth = 0
            WHERE c.mukey IN ({self.MUKEYS.format(wkt=wkt)}) {where}
            ORDER BY c.mukey, c.comppct_r DESC, i.mrulename""")


class SSURGO(SoilDataAccess):
    """SSURGO field-scale soil survey (1:12,000 - 1:63,360)."""

    MUKEYS = ("SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84"
              "('{wkt}')")


class STATSGO2(SoilDataAccess):
    """STATSGO2 regional-scale general soil map (1:250,000)."""

    MUKEYS = ("SELECT mukey FROM gsmmupolygon WHERE mupolygongeo.STIntersects("
              "geometry::STGeomFromText('{wkt}', 4326)) = 1")


class WebSoilSurvey(SSURGO):
    """Web Soil Survey: SSURGO data and interpretations (see `ratings`)."""


class AgDataCommons(base.Dataset):
    """USDA Ag Data Commons, hosted on Figshare."""

    BASE_URL = 'https://api.figshare.com/v2/'

    async def search(self, query: str) -> pd.DataFrame:
        """Returns Ag Data Commons datasets matching a full-text query."""
        hits = await self._json('POST', 'articles/search', json={
            'search_for': str(query), 'item_type': 3, 'page_size': 1000})
        return pd.DataFrame([
            {'id': a['id'], 'title': a['title'], 'doi': a['doi'],
             'published': a['published_date'], 'url': a['url_public_html']}
            for a in hits if 'agdatacommons' in a.get('url_public_html', '')])

    async def files(self, dataset_id: int) -> pd.DataFrame:
        """Lists a dataset's files: id, name, size, download_url."""
        article = await self._json('GET', f'articles/{int(dataset_id)}')
        return pd.DataFrame(article.get('files', []),
                            columns=['id', 'name', 'size', 'download_url'])

    async def read(self, download_url: str, **kwargs) -> pd.DataFrame:
        """Reads a CSV file; kwargs are passed to `pandas.read_csv`."""
        url = base.check_url(download_url, _FIGSHARE_HOSTS)
        body = await self._request('GET', url, public=True)
        return pd.read_csv(io.BytesIO(body), **kwargs)


def _point(lat: float, lon: float) -> str:
    """Returns WKT of a point (validated, so safe inside SQL)."""
    lat, lon = base.check_point(lat, lon)
    return f'point({lon} {lat})'


def _polygon(bbox: base.Bbox) -> str:
    """Returns WKT of a bbox (validated, so safe inside SQL)."""
    points = ', '.join(f'{lon} {lat}' for lon, lat in base.ring(bbox))
    return f'polygon(({points}))'
