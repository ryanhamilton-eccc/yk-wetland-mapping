from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import ee
import geopandas as gpd
import pandas as pd


def gdf2ee(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    """Convert a GeoDataFrame to an Earth Engine FeatureCollection.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to convert.

    Returns:
        ee.FeatureCollection: Earth Engine FeatureCollection.
    """
    pass


def imageCollection2FeatureCollection(imageCollection: ee.ImageCollection) -> ee.FeatureCollection:
    """Convert an Earth Engine ImageCollection to a FeatureCollection.

    Args:
        imageCollection (ee.ImageCollection): Earth Engine ImageCollection to convert.

    Returns:
        ee.FeatureCollection: Earth Engine FeatureCollection.
    """
    pass

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
@dataclass
class Ecoregion:
    features: gpd.GeoDataFrame
    region: gpd.GeoDataFrame
    

def process_eco_region(train_path, val_path, region_path, opt_map: dict[str, dict[str, int]] = None) -> Ecoregion:
    """Process the eco-region data.

    Args:
        train_path (str): Path to the training data.
        validation (str): Path to the validation data.
        region (str): Name of the region to process.

    Returns:
        EcoRegion: Processed eco-region data.
    """
    if opt_map is None:
        opt_map = {'class_name': {'wetland': 1, 'non-wetland': 2, 'water': 3}}
    
    CRS = 'EPSG:4326'
    
    train = gpd.read_file(train_path)
    train['type'] = 1
    
    validation = gpd.read_file(val_path)
    validation['type'] = 2
    
    region = gpd.read_file(region_path)
    region.to_crs(CRS, inplace=True)
    
    train = pd.concat([train, validation])
    train.to_crs(CRS, inplace=True)
    
    
    train['class_name'] = train['class_name'].map(opt_map['class_name'])
    
    return Ecoregion(train, region)


def prep_eco_region_for_ee(eco_region: Ecoregion, output_dir: str) -> None:
    """Prepare the eco-region data for Earth Engine. Zips the training and region data and saves them to the output directory.

    Args:
        eco_region (Ecoregion): Ecoregion to prepare.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    
    # save each group
    for k,v in eco_region.__dict__.items():
        try:
            
            scratch = Path(output_dir) / "scratch"
            scratch.mkdir(exist_ok=True)
            
            archive_name = f"{k}"
            v.to_file(scratch / f"{archive_name}.shp")
            # compress each dataset
            with ZipFile(scratch / f"{archive_name}.zip", "w") as zip:
                zip.write(scratch / f"{archive_name}.shp")
                zip.write(scratch / f"{archive_name}.dbf")
                zip.write(scratch / f"{archive_name}.prj")
                zip.write(scratch / f"{archive_name}.shx")
                zip.write(scratch / f"{archive_name}.cpg")

            # move the zip file to the processed directory
            (output_dir / "zipped").mkdir(exist_ok=True)
            (scratch / f"{archive_name}.zip").rename(
                output_dir / "zipped" / f"{archive_name}.zip"
            )

            # clean up the scratch directory
            # TODO: clean up needs to always happen
            (scratch / f"{archive_name}.shp").unlink()
            (scratch / f"{archive_name}.dbf").unlink()
            (scratch / f"{archive_name}.prj").unlink()
            (scratch / f"{archive_name}.shx").unlink()
            (scratch / f"{archive_name}.cpg").unlink()
            # output processed/shapefile AND processed/zipped

            # remove the scratch directory
            scratch.rmdir()

        except FileExistsError:
            print(f"File {k} already exists. Skipping...")
            continue
        
    return None
    
    
