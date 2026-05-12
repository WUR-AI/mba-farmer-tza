"""
Downloads iSDA data for the region and extracts the soil data for each point in
the dataset.
"""

import sys
# Get the soildata package from https://github.com/WUR-AI/soil-data
from soildata.download import download_isda_raster
from soildata.utils import get_buffer_points
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
import numpy as np
import pandas as pd
import os


# Download ISDA rasters
RASTER_DIR = f'/data/mba-tza/isda'
DATA_SAVE_DIR = f'/data/mba-tza/external-covs'
GEO_DATA_DIR = f'/data/mba-tza/geo'
bounding_box = gpd.read_file(f'{GEO_DATA_DIR}/region-bounding-box.geojson')
minx, miny, maxx, maxy = bounding_box.to_crs(4326).total_bounds

props_to_download = [
    'nitrogen_total', 'clay_content', 'sand_content', 'bulk_density',
    'cation_exchange_capacity', 'ph', 'carbon_organic'
]
rasters = {}
metadata = {}
for prop in props_to_download[:]: 
    rst_path, meta = download_isda_raster(RASTER_DIR, prop, minx, miny, maxx, maxy)
    rasters[prop] = rst_path
    metadata[prop] = meta

points = gpd.read_file(f'{GEO_DATA_DIR}/points.geojson')
points = points.set_index('fieldID')
points = points.loc[~points.index.duplicated(keep='first')]
values = {i: {} for i in points.index}
# Extract the data for a all points
for prop, rast_path in rasters.items():
    rast = rio.open(rast_path)
    points = points.to_crs(rast.crs)
    band_names = [i['name'] for i in metadata[prop]['assets']['image']['eo:bands']]
    print('Extracting data for', prop, '...')
    for idx, row in points.iterrows():
        buffer_points = list( # buffer = 50 returns all 8 neighboring pixels
            get_buffer_points(row.geometry, buffer=50, res=30)
        )
        buffer_values = np.array(list(rast.sample(buffer_points)))
        avg_values = buffer_values.mean(axis=0) # Mean in the buffer
        values[idx].update({
            f'{prop}_{band}': avg_values[i] 
            for i, band in enumerate(band_names)
        })

values = pd.DataFrame.from_dict(values, orient='index')
if not os.path.exists(DATA_SAVE_DIR):
    os.mkdir(DATA_SAVE_DIR)
values.to_pickle(f'{DATA_SAVE_DIR}/soil-data.pkl')
# new_points = points.join(values)
# print()
