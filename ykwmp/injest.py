from pathlib import Path
from typing import Union
import geopandas as gpd
import ee
import pandas as pd


# -- internal libs
from eesys import monitor_task 


def shapefile_to_feature_collection(shapefile_path: Union[str, Path]):
    """ 
    Generic function that converts a single shapefile to ee.FeatureCollection 
    """
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapfile not found {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    elif gdf.crs != 4326:
        gdf.to_crs(4326, inplace=True)
    return ee.FeatureCollection(gdf.__geo_interface__)


def merge_shapefiles_to_feature_collection(
    train_shapefile_path: Union[str, Path],
    validation_shapefile_path: Union[str, Path],
    id_column: str = "is_training"
) -> ee.FeatureCollection:
    """
    Merges training and validation shapefiles into an Earth Engine FeatureCollection.
    """
    # cast paths to Path object
    train_shapefile_path, validation_shapefile_path = map(Path, [train_shapefile_path, validation_shapefile_path])
    
    # Ensure the files exists
    if not train_shapefile_path.exists():
        raise FileNotFoundError(f"Training shapefile not found at: {train_shapefile_path}")
    
    if not validation_shapefile_path.exists():
        raise FileNotFoundError(f"Training shapefile not found at: {validation_shapefile_path}")
    
    # Read the shapefile into a GeoDataFrame
    gdf_train = gpd.read_file(train_shapefile_path)
    gdf_test = gpd.read_file(validation_shapefile_path)
 
    # Ensure the CRS is EPSG:4326 (WGS84)
    for gdf in [gdf_train, gdf_test]:
        if gdf.crs is None:
            gdf.set_crs(4326, inplace=True)
        elif gdf.crs != 4326:
            gdf.to_crs(4326, inplace=True)

    # add code or if column for if orig table is for train or test
    id_column = 'is_training'
    gdf_train[id_column] = 1
    gdf_test[id_column] = 2
    
    # merge tables into a single dataframe
    gdf = gpd.GeoDataFrame(pd.concat([gdf_train, gdf_test], ignore_index=True))


    # Convert the GeoDataFrame to a FeatureCollection
    feature_collection = ee.FeatureCollection(gdf.__geo_interface__)
    
    return feature_collection 


def ingest_shapefile_to_asset(
    shapefile_path: Union[str, Path],
    validation_shapefile_path: Union[str, Path],
    asset_id: str,
    asset_type: str = 'shapefile',
    merge: bool = False
) -> None:
    """
    Upload a shapefile (or merged shapefiles) to Earth Engine as an asset.
    
    Parameters:
    - shapefile_path: Path to the asset file (shapefile).
    - validation_shapefile_path: Path to the validation shapefile (optional, required for merging).
    - asset_id: The ID under which the asset will be saved in Earth Engine.
    - asset_type: Type of asset (default is 'shapefile').
    - merge: Whether to merge training and validation shapefiles (default is False).
    
    Returns:
    - None
    """
    if asset_type != 'shapefile':
        raise ValueError(f"Unsupported asset type: {asset_type}")
        
    if merge:
        if validation_shapefile_path is None:
            raise ValueError("validation_shapefile_path is required for merging shapefiles.")
        
        feature_collection = merge_shapefiles_to_feature_collection(
            train_shapefile_path=shapefile_path,
            validation_shapefile_path=validation_shapefile_path
        )
    else:
        feature_collection = shapefile_to_feature_collection(shapefile_path)

    try:
        # Upload the asset to Earth Engine
        task = ee.batch.Export.table.toAsset(
            collection=feature_collection,
            description=f"Upload {asset_type}",
            assetId=asset_id
        )
        task.start()
        print(f"Uploading asset to {asset_id}...")

        # Monitor the task status
        monitor_task(task)
    
    except Exception as e:
        print(f"Error uploading asset: {e}")