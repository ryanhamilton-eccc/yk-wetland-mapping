from pathlib import Path

import ee
import ee.data

# -- internal imports
from ykwmp.eesys import create_asset_folder, asset_exists, list_assets_in_path
from ykwmp.injest import batch_ingest_shapefiles, ingest_shapefile_to_asset
from ykwmp.image_processing import process_s1_images, process_s2_images
from ykwmp.image_utils import batch_extract
from ykwmp.classification import batch_model_and_assesment


def setup_workspace_step(workspace: str, region_path: str | Path) -> None:

    # -- create workspace
    # -- injest region shapefile
    if isinstance(region_path, str):
        region_path = Path(region_path)
    
    asset_id = f"{workspace}/{region_path.name.split('.')[0]}_region"
    if asset_exists(asset_id):
        print("Workspace already setup..")
        return
    
    create_asset_folder(workspace)
    
    ingest_shapefile_to_asset(
        shapefile_path=region_path,
        asset_id=asset_id
    )


def ingest_trainval_step(trainval_root: str | Path, ee_workspace: str) -> None:
    # -- run injestor
    batch_ingest_shapefiles(trainval_root, ee_workspace)


def batch_feature_extraction_step(workspace: str):
    # -- get the meta data
    region_id = list_assets_in_path(workspace, "_region$")
    features = list_assets_in_path(workspace, "_raw$")

    # -- extract the ids
    region_id = region_id[0].get('id')
    features = list(map(lambda x: x.get('id'), features))

    # consturct region
    region = ee.FeatureCollection(region_id).geometry()
    # -- s1 inputs
    s1 = process_s1_images(region, dates=("2019-06-20", "2019-09-21"))
    # -- s2 inputs
    s2 = process_s2_images(region)

    # create the stack
    stack = ee.Image.cat(s1, s2) 
    
    # run batch extract tool
    batch_extract(stack=stack, assets=features)

    ee.Reset()
    return

def batch_model_and_assessment_step(workspace):
    if not ee.data._credentials:
        ee.Initialize(project="yk-wetland-mapping")
        print("Re init api")
    inputs = list_assets_in_path(workspace, pattern="_feature$")
    inputs = list(map(lambda x: x.get('id'), inputs))
    batch_model_and_assesment(features=inputs, folder_name="165_metrics")
    ee.Reset()
    return


def batch_predict(workspace):
    if not ee.data._credentials:
        ee.Initialize(project="yk-wetland-mapping")
    region = list_assets_in_path(workspace, pattern="_region$")
    models = list_assets_in_path(workspace, pattern="_model$")

    region = region[0].get('id')
    models = list(map(lambda x: x.get('id'), models))
    # for each model predict on image 
    # image is a constant

    pass


if __name__ == '__main__':
    ee.Initialize(project="yk-wetland-mapping")
    setup_workspace_step(
        workspace="projects/yk-wetland-mapping/assets/test-workspace",
        region_path=r"D:\Yukon-Wetland-Mapping-Data\YK_Seperated_AOI\165.shp"
    )

    # -- step 2: Injest Training and Validation to ee file system
    ingest_trainval_step(
        trainval_root=r"D:\Yukon-Wetland-Mapping-Data\YK_GeeReady\165",
        ee_workspace="projects/yk-wetland-mapping/assets/test-workspace"
    )

    # -- step 3: feature extraction
    batch_feature_extraction_step("projects/yk-wetland-mapping/assets/test-workspace")

    # -- step 4: modeling and assessment
    batch_model_and_assessment_step("projects/yk-wetland-mapping/assets/test-workspace")

    # -- step 5: batch prediction