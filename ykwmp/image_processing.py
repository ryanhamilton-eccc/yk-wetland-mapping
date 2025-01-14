from typing import Optional, Tuple, Union
import ee


# -- internal imports
from ykwmp.datasets import dataset_factory
from ykwmp.image_utils import s2_cloud_mask, compute_ndvi, compute_tasseled_cap


def get_s1_images(
    aoi: ee.Geometry, 
    date: Tuple[str, str], 
    look_direction: Optional[Union[str, None]] = "DESCENDING", 
    relative_orbit_number: int | None = None,
    platform_number: str | None = "A",
) -> ee.Image | ee.ImageCollection:
     
    dataset = dataset_factory('s1', aoi, date)
    dataset = (
        dataset
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )

    if look_direction is not None:
        dataset = dataset.filter(ee.Filter.eq("orbitProperties_pass", look_direction))
    
    if relative_orbit_number is not None:
        dataset = dataset.filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit_number))

    if platform_number is not None:
        dataset = dataset.filter(ee.Filter.eq('platform_number', platform_number))

    return dataset


def process_s1_images(aoi, dates) -> ee.Image:
    return get_s1_images(aoi, dates).median().convolve(ee.Kernel.square(1)).select('V.*|H.*')

    
# -- S2 Data Collection and pre processing
def get_s2_images(
    aoi: ee.Geometry, 
    date: Tuple[str, str], 
    cloudy_pixel: Optional[float] = 20.0, 
    cloud_mask: bool = True, 
) -> ee.ImageCollection:
    dataset = dataset_factory('s2', aoi, date)

    if cloudy_pixel >= 0:
        dataset = dataset.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloudy_pixel))
    
    if cloud_mask:
        dataset = dataset.map(s2_cloud_mask)
    
    return dataset


def process_s2_images(
    aoi,  
    add_ndvi: bool = True, 
    add_tasseled_cap: bool = True, 
    start_date: str = "2018-06-01", 
    end_date: str = "2021-10-31",
    chunk_dates: bool = True,
    cloudy_pixel: Optional[float] = 20.0, 
    cloud_mask: bool = True
) -> ee.Image:
    if chunk_dates:
        sy,sm,sd = start_date.split("-")
        ey, em, ed = end_date.split("-")
        date_chunks = [(f"{y}-{sm}-{sd}", f"{y}-{em}-{ed}") for y in range(int(sy), int(ey) + 1)]
        datasets = [get_s2_images(aoi=aoi, date=date_chunk, cloudy_pixel=cloudy_pixel, cloud_mask=cloud_mask) for date_chunk in date_chunks]

        current_dateset = datasets.pop(0)
        for x in datasets:
            current_dateset = current_dateset.merge(x)
        dataset = current_dateset
    else:
        dataset = get_s2_images(aoi, (start_date, end_date), cloud_mask=cloud_mask, cloudy_pixel=cloudy_pixel)

    if add_ndvi:
        pass

    if add_tasseled_cap:
        pass
    # TODO add selectors for NDVI and Tasseled Cap
    return dataset.median().select("B[2-9]|B8A|B[0-1][0-2]|NDVI")