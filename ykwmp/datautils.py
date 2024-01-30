import ee
import geopandas as gpd


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


