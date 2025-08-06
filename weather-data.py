'''
Extracts AgERA5 data for each point in the dataset. Netcdf files are saved in a path
with the next structure:
agera5/
|-- year
|   |-- variable_name
|       |-- nc_file_name.nc
The data is saved in a multi-index pickled dataframe, where the index is the index
of the original points geodatrafame, and the date.
'''
import xarray as xr
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
from itertools import product

NC_DIR = '/data/mba-tza/agera5/ncfiles'

dates = pd.date_range('2023-06-01', '2024-12-31', freq='D')
variables = [
    'Temperature-Air-2m-Mean-24h', 'Temperature-Air-2m-Max-Day-Time',
    'Temperature-Air-2m-Min-Night-Time', 'Vapour-Pressure-Mean', 'Precipitation-Flux',
    'Solar-Radiation-Flux', 'Wind-Speed-10m-Mean'
]
points = gpd.read_file('data/geo/points.geojson')
points = points.set_index('fieldID')
points = points.loc[~points.index.duplicated(keep='first')]
points = points.to_crs(4326)

data = {(idx, day): {} for idx, day in product(points.index, dates)}
for variable in variables:
    for day in tqdm(dates):
        nc_path = f'{NC_DIR}/{day.year}/{variable}/{variable}_C3S-glob-agric_AgERA5_{day.strftime("%Y%m%d")}_final-v1.1.nc'
        ds = xr.open_dataset(nc_path)
        current_day_data = ds[variable.replace('-','_')].interp(
            lon=xr.DataArray(points.geometry.x.values, dims="point"),
            lat=xr.DataArray(points.geometry.y.values, dims="point"),
            method="linear"
        ).values[0]
        ds.close()
        for i, idx in enumerate(points.index):
            data[(idx, day)].update({variable: current_day_data[i]})
        
df = pd.DataFrame.from_dict(data, orient='index')
df.to_pickle('data/external-covs/weather-data-daily-2024-linearinterp.pkl')
print()


    