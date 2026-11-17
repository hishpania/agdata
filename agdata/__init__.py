"""Unified access to agricultural and Earth-observation datasets.

Every accessor is a coroutine returning a `pandas.DataFrame`. See API.md.
"""

from agdata import arcgis
from agdata import base
from agdata import climate
from agdata import earthdata
from agdata import land
from agdata import usda
from agdata import water

Dataset = base.Dataset
DatasetError = base.DatasetError

# NASA Earthdata.
Earthdata = earthdata.Earthdata
SMAP = earthdata.SMAP
GPM = IMERG = earthdata.GPM
Landsat = earthdata.Landsat
MODIS = earthdata.MODIS
VIIRS = earthdata.VIIRS
ECOSTRESS = earthdata.ECOSTRESS
NLDAS = earthdata.NLDAS
GLDAS = earthdata.GLDAS
GEDI = earthdata.GEDI

# Climate.
NasaPower = climate.NasaPower
NMME = climate.NMME
PRISM = climate.PRISM
CoCoRaHS = climate.CoCoRaHS

# Land, terrain and imagery.
SRTM = land.SRTM
NLCD = land.NLCD
GAEZ = arcgis.GAEZ
LANDFIRE = arcgis.LANDFIRE
NAIP = arcgis.NAIP

# Water.
NWIS = water.NWIS
OpenET = water.OpenET

# USDA.
QuickStats = usda.QuickStats
MarketNews = usda.MarketNews
CropScape = CDL = usda.CropScape
SSURGO = usda.SSURGO
STATSGO2 = usda.STATSGO2
WebSoilSurvey = usda.WebSoilSurvey
AgDataCommons = usda.AgDataCommons
