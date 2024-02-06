import os
import sys

import ee

import terrain_tools

import geopandas as gpd
import pandas as pd

from pathlib import Path
from pprint import pprint


BATCH_SIZE = 10
URI_FILE = "../data/cdem.txt"
MANIFEST_FILE = Path("../data/shapefiles/terrain.shp")


def create_manifest(out_file: Path) -> gpd.GeoDataFrame | None: 
    # load the raw text file that contains the list of the cloud files that need to be processed
    with open(URI_FILE, "r") as f:
        uris = f.read().splitlines()
    
    uri_df = pd.DataFrame(uris, columns=["uri"])
    uri_df["NTS_SNRC"] = uri_df["uri"].apply(lambda x: x.split("/")[-1].split(".")[0].split("_")[-1].upper())
    
    grid = gpd.read_file("../data/shapefiles/grid.shp")
    
    # merge the two dataframes
    uri_df = uri_df.merge(grid, on="NTS_SNRC", how='inner')
    save_path = "../data/shapefiles/terrain.shp"
    manifest = gpd.GeoDataFrame(uri_df)
    manifest.to_file(save_path)
    return manifest


def main() -> int:
    if not MANIFEST_FILE.exists():
        manifest = create_manifest(MANIFEST_FILE)
    else:
        print(f"Manifest file already exists at {MANIFEST_FILE}")
        manifest = gpd.read_file(MANIFEST_FILE)
    
    # create a list of the unique NTS_SNRC values
    cell_to_process = manifest[manifest.ta == 0].loc[:BATCH_SIZE]
    if cell_to_process.empty:
        print("All cells have been processed")
        return 0
    # print(cell_to_process)
    
    
    for _, cell in cell_to_process.iterrows():
        """need access to the uri and NTS_SNRC columns of the cell dataframe, and the geometry"""
        print(f"Processing cell {cell['NTS_SNRC']}")
        print(f"URI: {cell['uri']}")
        dem = ee.Image.loadGeoTIFF(cell['uri'])
        
        geom = ee.Geometry(cell['geometry'].__geo_interface__)
        
        pm_struct = terrain_tools.PeronaMalik(
            dataset=dem,
            geom=geom,
            elevation_band="B0"
        )
        pm_product = terrain_tools.compute_terrain_products(pm_struct)
        
        ga_struct = terrain_tools.Gaussian(
            dataset=dem,
            geom=geom,
            elevation_band="B0"
        )
        
        ga_product = terrain_tools.compute_terrain_products(ga_struct)
    
        stack = ga_product.addBands(pm_product)
        
        cloud_task = ee.batch.Export.image.toCloudStorage(
            image=stack,
            description=f"terrain_{cell['NTS_SNRC']}",
            bucket="yukon-wetland-mapping",
            fileNamePrefix=f"terrain/{cell['NTS_SNRC']}/terrain_{cell['NTS_SNRC']}-",
            region=geom,
            scale=20,
            fileFormat="GeoTIFF",
            fileDimensions=[2048,2048]
        )
        
        cloud_task.start()
        
        # update the manifest file
        manifest.loc[manifest.NTS_SNRC == cell['NTS_SNRC'], "ta"] = 1
        manifest.to_file(MANIFEST_FILE)

    return 0

if __name__ == '__main__':
    ee.Initialize()
    os.chdir(Path(__file__).parent)
    sys.exit(main())