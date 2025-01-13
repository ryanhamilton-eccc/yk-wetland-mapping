from typing import Callable, Tuple
import ee

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
def extract(image, collection) -> ee.FeatureCollection:
    return image.sampleRegions(
        collection=collection,
        scale=10,
        tileScale=16
    )

