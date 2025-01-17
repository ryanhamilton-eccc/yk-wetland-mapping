from pathlib import Path
import logging
import ee
import ee.data
import ee.data

# -- internal imports
from ykwmp.eesys import (
    create_asset_folder,
    asset_exists,
    list_assets_in_path,
    monitor_tasks,
    create_table_asset_task,
    export_error_matrix
)
from ykwmp.injest import batch_ingest_shapefiles, ingest_shapefile_to_asset
from ykwmp.image_processing import process_s1_images, process_s2_images
from ykwmp.classification import (
    compute_accuracy_metrics,
    partition_feature_collection,
    get_ee_predictors,
    FeatureInputs,
    RandomForestHyperparameters,
    randomforest,
    create_accuracy_feature_collection,
)
from ykwmp.image_utils import extract


# -- set up logger in global space
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pipeline_step(func):
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Starting step: {func.__name__}")
            result = func(*args, **kwargs)
            logger.info
            logger.info(f"Step {func.__name__} completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error during step {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


@pipeline_step
def setup_workspace_step(workspace: str, region_path: str | Path) -> None:
    logger.info(f"Setting up workspace at {workspace} with region {region_path}")
    if isinstance(region_path, str):
        region_path = Path(region_path)

    asset_id = f"{workspace}/{region_path.name.split('.')[0]}_region"
    if asset_exists(asset_id):
        print("Workspace already setup..")
        return

    create_asset_folder(workspace)

    ingest_shapefile_to_asset(shapefile_path=region_path, asset_id=asset_id)


@pipeline_step
def batch_ingest_trainval_step(trainval_root: str | Path, ee_workspace: str) -> None:
    # -- run injestor
    batch_ingest_shapefiles(trainval_root, ee_workspace)


@pipeline_step
def batch_feature_extraction_step(workspace: str):
    # -- get the meta data
    region_id = list_assets_in_path(workspace, "_region$")
    features = list_assets_in_path(workspace, "_raw$")

    # -- extract the ids
    region_id = region_id[0].get("id")
    features = list(map(lambda x: x.get("id"), features))

    # consturct region
    region = ee.FeatureCollection(region_id).geometry()
    # -- s1 inputs
    s1 = process_s1_images(region, dates=("2019-06-20", "2019-09-21"))
    # -- s2 inputs
    s2 = process_s2_images(region)
    # -- if you need to add additional layers here

    # create the stack
    stack = ee.Image.cat(s1, s2)

    # run batch extract tool
    assets = features
    tasks = []
    for asset in assets:
        # -- extract
        if not asset_exists(asset):
            continue
        raw = ee.FeatureCollection(asset)
        features = extract(stack, raw)

        # -- task
        asset_id = asset.replace("_raw", "_feature")
        task = create_table_asset_task(
            features, asset_id=asset_id, description=f"Export_Features"
        )

        task.start()
        print(f"Started export task for {asset_id}")
        tasks.append(task)

    monitor_tasks(tasks)
    return


@pipeline_step
def batch_model_and_assessment_step(workspace, gdirve_folder: str):
    inputs = list_assets_in_path(workspace, pattern="_feature$")
    inputs = list(map(lambda x: x.get("id"), inputs))

    features = inputs

    tasks = []
    for feature in features:
        inputs = ee.FeatureCollection(feature)
        train, test = partition_feature_collection(
            features=inputs,
            partition_column="is_training",
            train_value=1,
            validation_value=2,
        )

        predictors = get_ee_predictors(train)

        model_inputs = FeatureInputs(
            features=train, class_property="class_value", input_properties=predictors
        )

        model = randomforest(
            feature_inputs=model_inputs, hyperparameters=RandomForestHyperparameters()
        )

        error_matrix = compute_accuracy_metrics(
            model=model, validation=test, actual="class_value"
        )

        serialized_matrix = create_accuracy_feature_collection(metrics=error_matrix)

        # -- model asset task
        model_asset_id = feature.replace("_feature", "_model")
        task = ee.batch.Export.classifier.toAsset(
            classifier=model, description="ExportModel", assetId=model_asset_id
        )
        task.start()
        print(f"Started export task for {model_asset_id}")
        tasks.append(task)

        # -- export to drive
        name = feature.split("/")[-1].replace("_feature", "_metrics")
        export_error_matrix(table=serialized_matrix, folder=gdirve_folder, filename=name)
    monitor_tasks(tasks)
    return
    

@pipeline_step
def batch_predict(workspace, model_workspace: str = None):
    if not ee.data._credentials:
        ee.Initialize(project="yk-wetland-mapping")
    region = list_assets_in_path(workspace, pattern="_region$")
    if model_workspace is None:
        models = list_assets_in_path(workspace, pattern="_model$")
    else:
        models = list_assets_in_path(model_workspace, pattern="_model$")

    region = region[0].get("id")
    models = list(map(lambda x: x.get("id"), models))

    # consturct region
    region = ee.FeatureCollection(region).geometry()

    # -- s1 inputs
    s1 = process_s1_images(region, dates=("2019-06-20", "2019-09-21"))
    # -- s2 inputs
    s2 = process_s2_images(region)
    # -- if you need to add additional layers here

    # create the stack
    stack = ee.Image.cat(s1, s2)
    


    pass


def reset_and_init_api(projectid: str):
    ee.Reset()
    ee.Initialize(project=projectid)


def run_pipeline(
    project_id: str,
    workspace: str, # optional
    region_path: str, # required
    trainval_root: str, # optional
    gdrive_foldername: str # optional
) -> None:
    """
    we can run this with just the input models and the region but everything 
    else is optional

    the logic i want to use is that if the train val is empty it might be 
    fair to assume that we only want to run the pipeline with just the prediction
    """
    workspace = f"projects/{project_id}/assets/{workspace}"
    logger.info("Setting up workspace...")
    setup_workspace_step(workspace, region_path)

    logger.info("Ingesting training and validation data...")
    batch_ingest_trainval_step(trainval_root, workspace)
    # reset_and_init_api(project_id)
    
    logger.info("Running batch feature extraction...")
    batch_feature_extraction_step(workspace)
    reset_and_init_api(project_id)

    logger.info("Running batch model and assessment...")
    batch_model_and_assessment_step(workspace, gdrive_foldername)
    reset_and_init_api(project_id)

    logger.info("Running batch prediction...")

    logger.info("Pipeline completed.")


def run_pipeline_without_feature_extration(
    project_id: str, 
    workspace: str, 
    model_workspace: str
) -> None:
    pass

if __name__ == '__main__':
    import ee

    ee.Initialize(project="yk-wetland-mapping")

    run_pipeline(
        project_id='yk-wetland-mapping',
        workspace="test-workspace",
        region_path=r"D:\Yukon-Wetland-Mapping-Data\YK_Seperated_AOI\165.shp",
        trainval_root=r"D:\Yukon-Wetland-Mapping-Data\YK_GeeReady\165",
        gdrive_foldername="165"
    )