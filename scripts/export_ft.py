import os
import sys

from dataclasses import dataclass
from pathlib import Path

import ee
import geopandas as gpd

from ykwmp import GRID_PATH
from ykwmp.rsd import Sentinel2, Sentinel2TimeSeries


@dataclass
class GridCell:
    grid: gpd.GeoDataFrame
    index: int

@dataclass
class Fourier:
    product: ee.Image
    region: ee.Geometry


def load_grid() -> gpd.GeoDataFrame:
    grid_path = Path(GRID_PATH)
    
    if grid_path.exists():
        grid = gpd.read_file(grid_path)
        return grid

    print("Grid path does not exist")
    return None
    

def process_grid(grid: gpd.GeoDataFrame) -> GridCell:
    # need to select the first 0 in the grid, dataframe
    grid = grid.copy()
    cell = grid[grid["ft"] == 0].iloc[0:1]
    # convert grid cell to crs: 4326
    cell.to_crs(4326, inplace=True)
    return GridCell(grid=cell, index=cell.index[0])


def compute_fourier_transform(grid_cell: GridCell) -> Fourier:
    geo_json = grid_cell.grid.__geo_interface__
    ee_cell = ee.FeatureCollection(geo_json).geometry()
    s2_ts = Sentinel2(ee_cell, start="2018-01-01", end="2022-12-31")
    ft = Sentinel2TimeSeries(
        dataset=s2_ts
    ).process()
    
    return Fourier(product=ft.transform(), region=ee_cell)



def main() -> int:
    # need to load the grid
    grid = load_grid()
    if grid is None:
        print("Usage: GRID_PATH=/path/to/grid.json python export_ft.py")
        return 1
    
    grid_cell = process_grid(grid)

    compute = compute_fourier_transform(grid_cell)
    
    product = compute.product.select("phase_1|phase_2")
    region = compute.region
    
    grid_name = grid_cell.grid.iloc[0]["NTS_SNRC"]
    
    task = ee.batch.Export.image.toCloudStorage(
        image=product,
        description="",
        bucket="yukon-wetland-mapping",
        fileNamePrefix=f"fourier/{grid_name}/{grid_name}-ft-",
        region=region,
        scale=10,
        fileDimensions=[2048, 2048],
        maxPixels=1e13,
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
        skipEmptyTiles=True,
        formatOptions={
            "cloudOptimized": True
        }
    )
    
    task.start()
    print(f"Exporting {grid_name} to Cloud Storage")
    # # update the grid
    grid.loc[grid_cell.index, "ft"] = 1
    grid.to_file(GRID_PATH)
    
    return 0


if __name__ == "__main__":
    ee.Initialize()
    sys.exit(main())