from pathlib import Path
from typing import Union
import geopandas as gpd
import ee


def shapefile_to_feature_collection(shapefile_path: Union[str, Path]) -> ee.FeatureCollection:
    """
    Convert a point shapefile to an Earth Engine FeatureCollection.
    
    Parameters:
    - shapefile_path: Path to the shapefile.
    
    Returns:
    - ee.FeatureCollection: Converted Earth Engine FeatureCollection.
    """
    # Ensure the file exists
    shapefile_path = Path(shapefile_path)
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found at: {shapefile_path}")
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure the CRS is EPSG:4326 (WGS84)
    if gdf.crs is None:
        gdf.set_crs(4326, inplace=True)
    elif gdf.crs != 4326:
        gdf.to_crs(4326, inplace=True)

    # Convert the GeoDataFrame to a FeatureCollection
    feature_collection = ee.FeatureCollection(gdf.__geo_interface__)
    
    return feature_collection 


def load_asset_from_path(path: Union[str, Path], asset_type: str = 'shapefile') -> ee.FeatureCollection:
    """
    Loads an asset from the given path, given the type of asset (e.g., shapefile).
    
    Parameters:
    - path: Path to the asset file (e.g., shapefile path).
    - asset_type: Type of the asset (e.g., shapefile).
    
    Returns:
    - ee.FeatureCollection: The corresponding FeatureCollection.
    """
    # Check asset type and process accordingly
    if asset_type == 'shapefile':
        return shapefile_to_feature_collection(path)
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")


def ingest_shapefile_to_asset(
    shapefile_path: Union[str, Path],
    asset_id: str,
    asset_type: str = 'shapefile'
) -> None:
    """
    Upload a shapefile (or other assets) to Earth Engine as an asset.
    
    Parameters:
    - shapefile_path: Path to the asset file (shapefile).
    - asset_id: The ID under which the asset will be saved in Earth Engine.
    - asset_type: Type of asset (default is 'shapefile').
    
    Returns:
    - None
    """
    if asset_type == 'shapefile':
        # Convert shapefile to FeatureCollection
        feature_collection = shapefile_to_feature_collection(shapefile_path)
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")

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