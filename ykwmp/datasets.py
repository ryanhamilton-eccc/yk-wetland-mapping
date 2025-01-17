from typing import Any, Tuple
import ee


def dataset_factory(dataset_id: str, aoi: ee.Geometry | None = None, dates: Tuple[str, str] | None = None) -> ee.ImageCollection:
    ids = {
        's1': "COPERNICUS/S1_GRD",
        's2': "COPERNICUS/S2_HARMONIZED"
    }
    dataset = ee.ImageCollection(ids[dataset_id])
    if aoi is not None:
        dataset = dataset.filterBounds(aoi)
    if dates is not None:
        dataset = dataset.filterDate(*dates)
    return dataset

