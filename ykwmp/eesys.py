import ee
import time


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


# -- Tasks
def export_classification_result(classified_image: ee.Image, task_name: str, region: ee.Geometry, scale: int = 30):
    export_task = ee.batch.Export.image.toDrive(
        image=classified_image,
        description=task_name,
        scale=scale,
        region=region,
        fileFormat='GeoTIFF'
    )
    export_task.start()
    return export_task

def export_error_matrix(error_matrix: ee.ConfusionMatrix, task_name: str):
    # Convert the confusion matrix to a FeatureCollection or other format as needed
    cm_fc = ee.FeatureCollection([ee.Feature(None, {
        'confusion_matrix': error_matrix.getInfo()  # Or process accordingly
    })])
    
    export_task = ee.batch.Export.table.toDrive(
        collection=cm_fc,
        description=task_name,
        fileFormat='CSV',
        folder='error_matrix_exports'
    )
    export_task.start()
    return export_task

def export_model(model: ee.Classifier, task_name: str):
    model_dict = {
        'model': model.getInfo()  # Serialize the model as needed
    }
    
    model_fc = ee.FeatureCollection([ee.Feature(None, model_dict)])
    
    export_task = ee.batch.Export.table.toDrive(
        collection=model_fc,
        description=task_name,
        fileFormat='CSV',
        folder='model_exports'
    )
    export_task.start()
    return export_task

def export_features(features: ee.FeatureCollection, task_name: str):
    export_task = ee.batch.Export.table.toDrive(
        collection=features,
        description=task_name,
        fileFormat='CSV',
        folder='features_exports'
    )
    export_task.start()
    return export_task


# -- Monitor Tasks
def monitor_task(task: ee.batch.Task, check_interval: int = 10):
    """
    Monitors the Earth Engine task until it is completed or fails.
    Prints the status periodically.
    
    Parameters:
        task: ee.batch.Task - The task to monitor
        check_interval: int - Time (in seconds) between status checks
    
    Returns:
        None
    """
    while task.active():
        print(f"Task status: {task.status()['state']}")
        time.sleep(check_interval)  # Wait for the next check
    
    print(f"Final task status: {task.status()['state']}")
    if task.status()['state'] == 'COMPLETED':
        print("Task completed successfully!")
    else:
        print(f"Task failed with error: {task.status().get('error_message', 'No error message available')}")