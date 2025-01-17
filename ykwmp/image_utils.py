from typing import Callable, List, Tuple
import ee


from ykwmp.eesys import asset_exists, create_table_asset_task, monitor_tasks


# -- Stacking
def stack(*images) -> ee.Image:
    return ee.Image.cat(*images)


# -- Raster Calculators
def compute_ndvi(nir: str, red: str, name: str = "NDVI") -> Callable:
    pass


def compute_tasseled_cap(bands: Tuple[str, ...]) -> Callable:
    pass


# -- Masks
def s2_cloud_mask(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


def edge_mask(image) -> ee.Image:
    pass


# -- Feature Extraction
def extract(image: ee.Image, collection: ee.FeatureCollection, properties: List[str] = None) -> ee.FeatureCollection:
    return image.sampleRegions(
        collection=collection,
        scale=10,
        tileScale=16,
        geometries=True,
        properties=properties
    )


def batch_extract(stack: ee.Image, assets: List[str]):
    tasks = []
    for asset in assets:
        # -- extract
        if not asset_exists(asset):
            continue
        raw = ee.FeatureCollection(asset)
        features = extract(stack, raw)

        # -- task
        asset_id = asset.replace("_raw", "_feature")
        task  = create_table_asset_task(
            features,
            asset_id=asset_id,
            description=f"Export_Features"
        )

        task.start()
        print(f"Started export task for {asset_id}")
        tasks.append(task)

    monitor_tasks(tasks)
    return 