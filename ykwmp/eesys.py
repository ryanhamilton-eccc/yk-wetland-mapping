import ee

def asset_exists(asset_id: str) -> bool:
    """
    Check if an asset exists in Google Earth Engine.
    
    Parameters:
    - asset_id: The asset ID to check.
    
    Returns:
    - bool: True if the asset exists, False otherwise.
    """
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.ee_exception.EEException:
        return False


def export_if_not_exists(collection: ee.FeatureCollection, asset_id: str, description: str = "export_task"):
    """
    Export a FeatureCollection to an asset if it doesn't already exist.
    
    Parameters:
    - collection: The FeatureCollection to export.
    - asset_id: The target asset ID.
    - description: Task description (default is "export_task").
    """
    if not asset_exists(asset_id):
        task = ee.batch.Export.table.toAsset(
            collection=collection,
            description=description,
            assetId=asset_id
        )
        task.start()
        print(f"Export started for asset: {asset_id}")
    else:
        print(f"Asset {asset_id} already exists. Skipping export.")
        

def list_assets_in_path(path: str) -> list:
    """
    List all assets under a specific path in Google Earth Engine.
    
    Parameters:
    - path: The asset path.
    
    Returns:
    - list: A list of asset metadata.
    """
    return ee.data.listAssets({"parent": path}).get('assets', [])


def delete_asset(asset_id: str):
    """
    Delete an asset from Google Earth Engine.
    
    Parameters:
    - asset_id: The ID of the asset to delete.
    """
    try:
        ee.data.deleteAsset(asset_id)
        print(f"Asset {asset_id} deleted successfully.")
    except ee.ee_exception.EEException as e:
        print(f"Failed to delete asset {asset_id}: {e}")
