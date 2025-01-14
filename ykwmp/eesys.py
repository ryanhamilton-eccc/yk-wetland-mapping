from datetime import datetime
from typing import Any, Dict, List
import ee
import re
import time

import ee.batch


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


def create_asset_folder(folder_id: str):
    """
    Create a folder in the Earth Engine file system.
    
    Parameters:
    - folder_id: The asset ID of the folder you want to create (e.g., 'users/your_username/my_folder/').
    
    Returns:
    - None
    """
    try:
        # Create the folder in Earth Engine by using createAssetHome
        ee.data.createAssetHome(folder_id)
        print(f"Folder {folder_id} created successfully.")
        
    except Exception as e:
        print(f"Error creating folder: {e}")


def list_assets_in_path(path: str, pattern: str = None) -> List[Dict[str, Any]]:
    """ 
    List all assets under a specific path in Google Earth Engine.
    
    Parameters:
    - path: The asset path.
    
    Returns:
    - list: A list of asset metadata.
    """
    assets = ee.data.listAssets({"parent": path}).get('assets', [])
    if pattern is None:
        return assets
    
    regex = re.compile(pattern)
    matching_assets = [asset for asset in assets if regex.search(asset.get("id", ""))]
    return matching_assets

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


# -- Asset tasks
def create_table_asset_task(table: ee.FeatureCollection, asset_id: str, description: str = "") -> ee.batch.Task:
    task = ee.batch.Export.table.toAsset(
        collection=table,
        description=description,
        assetId=asset_id
    )
    return task


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
        # Monitor the task status
        monitor_task(task)
    else:
        print(f"Asset {asset_id} already exists. Skipping export.")


def export_model_if_not_exits(model: ee.Classifier, asset_id: str, description: str = "model_task"):
    if not asset_exists(asset_id=asset_id):
        task = ee.batch.Export.classifier.toAsset(classifier=model, assetId=asset_id, description=description)
        task.start()
        print(f"Export started for asset: {asset_id}")
        # Monitor the task status
        monitor_task(task)
    else:
        print(f"Asset {asset_id} already exists. Skipping export.")


# -- Drive Tasks
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

def export_error_matrix(table: ee.FeatureCollection, folder: str, filename: str):
    
    export_task = ee.batch.Export.table.toDrive(
        collection=table,
        description='ErrorMatrixs',
        fileFormat='GeoJSON',
        folder=folder,
        fileNamePrefix=filename
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


def monitor_tasks(tasks: List[ee.batch.Task], interval: int = 10):
    """
    Monitors Earth Engine tasks and prints their statuses.

    Parameters:
    - tasks (list): List of ee.batch.Task objects to monitor.
    - interval (int): Time interval (in seconds) between status checks.
    """
    print("Monitoring tasks...")
    while True:
        print(f"Status At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        all_done = True
        for task in tasks:
            status = task.status()
            state = status.get('state', 'UNKNOWN')
            print(f"Task {task.id}: {state}")
            
            if state not in ('COMPLETED', 'FAILED', 'CANCELLED'):
                all_done = False
        print("#" * 30)
        if all_done:
            print("All tasks are complete.")
            break
        
        time.sleep(interval)


def monitor_tasks_inplace(tasks: List[ee.batch.Task], interval: int = 10):
    """
    Monitors Earth Engine tasks and updates the status in place on the console.

    Parameters:
    - tasks (list): List of ee.batch.Task objects to monitor.
    - interval (int): Time interval (in seconds) between status checks.
    """
    print("Monitoring tasks...\n")
    while True:
        all_done = True
        status_lines = []
        
        for task in tasks:
            status = task.status()
            state = status.get('state', 'UNKNOWN')
            status_lines.append(f"Task {task.id}: {state}")
            
            if state not in ('COMPLETED', 'FAILED', 'CANCELLED'):
                all_done = False
        
        # Clear the console lines and update with the current statuses
        print("\033[F" * len(status_lines), end="")  # Move the cursor up
        for line in status_lines:
            print(line)
        
        if all_done:
            print("\nAll tasks are complete.")
            break
        
        time.sleep(interval)