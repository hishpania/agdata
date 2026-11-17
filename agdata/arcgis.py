"""ArcGIS ImageServer datasets: FAO GAEZ, USFS LANDFIRE and NAIP."""

import asyncio
import json
from typing import ClassVar

import pandas as pd

from agdata import base


class ImageService(base.Dataset):
    """Pixel values of an ArcGIS ImageServer at a point.

    `BASE_URL` is the services root; `LAYER` the default image service path
    under it.
    """

    LAYER: ClassVar[str] = ''
    BANDS: ClassVar[tuple[str, ...]] = ('value',)
    MAX_RASTERS: ClassVar[int] = 250  # Most rasters per area request.

    async def identify(self, lat: float, lon: float, where: str | None = None,
                       layer: str | None = None) -> pd.DataFrame:
        """Returns pixel values at a point.

        Args:
            lat: Latitude.
            lon: Longitude.
            where: SQL filter on catalog attributes of mosaic datasets,
                e.g. "crop='Maize'". Quote values with `base.sql_text`.
            layer: Image service path; defaults to `LAYER`.

        Returns:
            One row per raster (with catalog attributes for mosaics) and
            one column per band.
        """
        lat, lon = base.check_point(lat, lon)
        layer = base.check_name(layer or self.LAYER, r'[\w./-]+')
        params = {
            'geometry': json.dumps({'x': lon, 'y': lat,
                                    'spatialReference': {'wkid': 4326}}),
            'geometryType': 'esriGeometryPoint',
            'returnGeometry': 'false',
            'returnCatalogItems': 'true',
            'maxItemCount': 1000,
            'f': 'json',
        }
        if where:
            params['mosaicRule'] = json.dumps({'where': where})
        data = await self._json('GET', f'{layer}/ImageServer/identify',
                                params=params)
        if 'error' in data:
            raise base.DatasetError(data['error'])
        items = (data.get('catalogItems') or {}).get('features') or []
        if items:
            values = data['properties']['Values']
            rows = pd.DataFrame([item['attributes'] for item in items])
        else:
            values, rows = [data['value']], pd.DataFrame(index=[0])
        return rows.join(self._bands(values))

    async def identify_area(self, bbox: base.Bbox, where: str | None = None,
                            layer: str | None = None,
                            samples: int = 1024) -> pd.DataFrame:
        """Returns pixel values sampled on a regular grid over a bbox.

        Args:
            bbox: (west, south, east, north).
            where: SQL filter on catalog attributes of mosaics, as in
                `identify`.
            layer: Image service path; defaults to `LAYER`.
            samples: Samples per raster, at most 1024 (server limit).

        Returns:
            One row per raster and sample: lat, lon, catalog attributes and
            one column per band. Ids of rasters the server failed to sample
            are in `.attrs['skipped']`.
        """
        layer = base.check_name(layer or self.LAYER, r'[\w./-]+')
        path = f'{layer}/ImageServer/'
        west, south, east, north = base.check_bbox(bbox)
        area = {'geometry': json.dumps({
                    'xmin': west, 'ymin': south, 'xmax': east, 'ymax': north,
                    'spatialReference': {'wkid': 4326}}),
                'geometryType': 'esriGeometryEnvelope', 'f': 'json'}
        # Primary (Category 1) catalog rasters; plain image services have
        # no catalog and are sampled as one raster (None).
        primary = f'({where}) AND Category = 1' if where else 'Category = 1'
        data = await self._json('GET', path + 'query', params={
            **area, 'inSR': 4326, 'spatialRel': 'esriSpatialRelIntersects',
            'returnIdsOnly': 'true', 'where': primary})
        ids = [None] if 'error' in data else data.get('objectIds') or []
        if len(ids) > self.MAX_RASTERS:
            raise ValueError(f'{len(ids)} rasters: over {self.MAX_RASTERS}; '
                             'narrow `where` or `bbox`.')

        async def sample(raster: int | None) -> list[dict] | None:
            params = {**area, 'sampleCount': max(1, min(int(samples), 1024)),
                      'returnFirstValueOnly': 'true', 'outFields': '*'}
            if raster is not None:
                params['mosaicRule'] = json.dumps({
                    'mosaicMethod': 'esriMosaicLockRaster',
                    'lockRasterIds': [int(raster)]})
            data = await self._json('GET', path + 'getSamples', params=params)
            if 'error' in data:
                return None
            return [{'lat': s['location']['y'], 'lon': s['location']['x'],
                     **s.get('attributes', {}), 'value': s.get('value')}
                    for s in data.get('samples', [])]

        results = await asyncio.gather(*map(sample, ids))
        rows = pd.DataFrame([row for rows in results if rows for row in rows])
        if not rows.empty:
            rows = rows.drop(columns='value').join(self._bands(rows.value))
        rows.attrs['skipped'] = [i for i, r in zip(ids, results) if r is None]
        return rows

    def _bands(self, values) -> pd.DataFrame:
        """Splits pixel values ('r g b nir') into numeric band columns."""
        bands = pd.Series(values, dtype=str).str.split(r'[,\s]+', expand=True)
        bands.columns = list(self.BANDS)[:bands.shape[1]]
        return bands.apply(pd.to_numeric, errors='coerce')


class GAEZ(ImageService):
    """FAO GAEZ v4 crop suitability, yields and agro-climatic resources.

    Layers (themes): res01 land & water resources, res02 agro-climatic
    resources, res05 suitability & attainable yield (default), res06 actual
    yields & production, res07 yield & production gaps.
    """

    BASE_URL = 'https://gaez-services.fao.org/server/rest/services/'
    LAYER = 'res05'

    async def suitability(self, lat: float, lon: float, crop: str,
                          rcp: str = 'Historical') -> pd.DataFrame:
        """Returns suitability and attainable yield layers for a crop.

        Args:
            lat: Latitude.
            lon: Longitude.
            crop: GAEZ crop name, e.g. 'Maize', 'Soybean', 'Wheat'.
            rcp: Climate scenario: 'Historical', 'RCP2.6' ... 'RCP8.5'.
        """
        return await self.identify(lat, lon, where=_crop(crop, rcp))

    async def suitability_area(self, bbox: base.Bbox, crop: str,
                               rcp: str = 'Historical', samples: int = 1024
                               ) -> pd.DataFrame:
        """Returns suitability and attainable yield layers over a bbox.

        One raster per layer: many hundreds for some crops. See
        `identify_area`.
        """
        return await self.identify_area(bbox, where=_crop(crop, rcp),
                                        samples=samples)


class LANDFIRE(ImageService):
    """USFS LANDFIRE vegetation and fuel layers (30 m, US).

    Layers follow 'Landfire_LF<ver>/LF<ver>_<PRODUCT>_<REGION>', e.g.
    EVT (vegetation type), EVC (cover), EVH (height), FBFM40 (fuel model).
    """

    BASE_URL = 'https://lfps.usgs.gov/arcgis/rest/services/'
    LAYER = 'Landfire_LF2024/LF2024_EVT_CONUS'


class NAIP(ImageService):
    """USDA NAIP 1 m aerial imagery: RGB+NIR pixels for every acquisition."""

    BASE_URL = 'https://imagery.nationalmap.gov/arcgis/rest/services/'
    LAYER = 'USGSNAIPImagery'
    BANDS = ('red', 'green', 'blue', 'nir')

    async def pixels(self, lat: float, lon: float) -> pd.DataFrame:
        """Returns band values and NDVI per acquisition, newest first."""
        return _ndvi(await self.identify(lat, lon))

    async def pixels_area(self, bbox: base.Bbox,
                          samples: int = 1024) -> pd.DataFrame:
        """Returns band values and NDVI sampled over a bbox, per raster."""
        return _ndvi(await self.identify_area(bbox, samples=samples))


def _crop(crop: str, rcp: str) -> str:
    return f'crop={base.sql_text(crop)} AND rcp={base.sql_text(rcp)}'


def _ndvi(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds NDVI and sorts by acquisition date, newest first."""
    if 'nir' in frame:
        frame['ndvi'] = (frame.nir - frame.red) / (frame.nir + frame.red)
    if 'acquisition_date' in frame:
        frame['acquisition_date'] = pd.to_datetime(
            frame['acquisition_date'], unit='ms')
        frame = frame.sort_values('acquisition_date', ascending=False,
                                  kind='stable')
    return frame
