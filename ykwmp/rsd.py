from __future__ import annotations
import ee
import geopandas as gpd

from math import pi


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

## Sentinel-2

class Sentinel2:
    """
    Represents a Sentinel-2 satellite image dataset.

    Attributes:
        BANDS (list): List of band names in the Sentinel-2 dataset.
    """

    BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
    
    def __init__(self, aoi, start, end):
        """
        Initializes a Sentinel2 object.

        Args:
            aoi: Area of interest.
            start: Start date of the image collection.
            end: End date of the image collection.
        """
        self.aoi = aoi
        self.start = start
        self.end = end
        self.dataset = ee.ImageCollection("COPERNICUS/S2")
        
    def mask_clouds(self, image: ee.Image) -> ee.Image:
        """
        Masks clouds in a Sentinel-2 image.

        Args:
            image (ee.Image): The input Sentinel-2 image.

        Returns:
            ee.Image: The input image with clouds masked.
        """
        qa = image.select("QA60")
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask)
        
    def add_tasseled_cap(self, image: ee.Image) -> ee.Image:
        """
        Computes the Tasseled Cap transformation on a Sentinel-2 TOA image.

        Args:
            image (ee.Image): The input Sentinel-2 TOA image.

        Returns:
            ee.Image: The input image with additional bands representing the Tasseled Cap components.
        """
    
        coefficients = ee.Array(
            [
                [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],
                [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],
                [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
                [-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773],
                [-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085],
                [0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252],
            ]
        )   

        image = image.select(["B2", "B3", "B4", "B8", "B11", "B12"])
        array_image = image.toArray()
        array_image_2d = array_image.toArray(1)

        components = (
            ee.Image(coefficients)
            .matrixMultiply(array_image_2d)
            .arrayProject([0])
            .arrayFlatten(
                [["brightness", "greenness", "wetness", "fourth", "fifth", "sixth"]]
            )
        )
        components = components.select(["brightness", "greenness", "wetness"])
        return image.addBands(components)

    
    def add_ndvi(self, image: ee.Image) -> ee.Image:
        """
        Adds the Normalized Difference Vegetation Index (NDVI) band to a Sentinel-2 image.

        Args:
            image (ee.Image): The input Sentinel-2 image.

        Returns:
            ee.Image: The input image with the NDVI band added.
        """
        return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("NDVI"))

    def merge(self, other: Sentinel2) -> Sentinel2:
        """
        Merges the image collection of another Sentinel2 object with the current object.

        Args:
            other (Sentinel2): The other Sentinel2 object to merge.

        Returns:
            Sentinel2: The merged Sentinel2 object.
        """
        self.dataset = self.dataset.merge(other.dataset)
        return self

    def process(self, cloudy_pix: int = 20):
        """
        Processes the Sentinel-2 image collection by applying various filters and transformations.

        Args:
            cloudy_pix (int): The maximum percentage of cloudy pixels allowed.

        Returns:
            Sentinel2: The processed Sentinel2 object.
        """
        self.dataset = (
            self.dataset
            .filterDate(self.start, self.end)
            .filterBounds(self.aoi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloudy_pix))
            .map(self.mask_clouds)
            .select(self.BANDS)
            .map(self.add_ndvi)
            .map(self.add_tasseled_cap)
        )
        return self

    def get_dataset(self) -> ee.ImageCollection:
        """
        Returns the processed Sentinel-2 image collection.

        Returns:
            ee.ImageCollection: The processed Sentinel-2 image collection.
        """
        return self.dataset
    
    def composite(self) -> ee.Image:
        """
        Returns a composite image of the processed Sentinel-2 image collection.

        Returns:
            ee.Image: The composite image.
        """
        return self.dataset.median()


class Sentinel2TimeSeries:
    """
    A class representing a time series analysis of Sentinel-2 satellite data.
    
    Attributes:
        dataset (Sentinel2): The Sentinel2 dataset used for the analysis.
    """
    
    def __init__(self, dataset: Sentinel2) -> None:
        """
        Initializes a new instance of the Sentinel2TimeSeries class.
        
        Args:
            dataset (Sentinel2): The Sentinel2 dataset used for the analysis.
        """
        self.dataset: Sentinel2 = dataset
    
    def add_time(self, element: ee.computedobject):
        """
        Adds a time band to an image element.
        
        Args:
            element (ee.computedobject): The image element to add the time band to.
        
        Returns:
            ee.computedobject: The image element with the time band added.
        """
        date = ee.Date(element.get('system:time_start'))
        years = date.difference(ee.Date("1970-01-01"), "year")
        time_radians = ee.Image(years.multiply(2 * pi))
        return element.addBands(time_radians.rename("t").float())
    
    def add_harmonics(self, cos: list[str], sin: list[str]):
        """
        Adds harmonic bands to each image in the dataset.
        
        Args:
            cos (list[str]): The names of the cosine bands.
            sin (list[str]): The names of the sine bands.
        
        Returns:
            function: A function that adds harmonic bands to an image.
        """
        def wrapper(element: ee.computedobject):
            frequencies = ee.Image.constant(list(range(1, 4)))
            time = ee.Image(element).select("t")
            cos_img = time.multiply(frequencies).cos().rename(cos)
            sin_img = time.multiply(frequencies).sin().rename(sin)
            return element.addBands(cos_img).addBands(sin_img)    
        return wrapper
    
    def add_phase(self, mode: int):
        """
        Adds a phase band to each image in the dataset.
        
        Args:
            mode (int): The harmonic mode.
        """
        def compute_phase(image: ee.Image):
            cos = image.select(f"cos_{mode}_coef")
            sin = image.select(f"sin_{mode}_coef")
            arctan = cos.atan2(sin)
            return image.addBands(arctan.rename(f"phase_{mode}"))
        return compute_phase
    
    def add_amplitude(self, mode: int):
        def compute_amplitude(image: ee.Image):
            cos = image.select(f"cos_{mode}_coef")
            sin = image.select(f"sin_{mode}_coef")
            amplitude = cos.hypot(sin)
            return image.addBands(amplitude.rename(f"amplitude_{mode}"))
        return compute_amplitude

   
    def process(self, modes: int = 3):
        """
        Processes the Sentinel2 dataset by adding time and harmonic bands, computing coefficients, and adding them to the dataset.
        
        Args:
            modes (int): The number of harmonic modes to consider.
        
        Returns:
            Sentinel2TimeSeries: The processed Sentinel2TimeSeries object.
        """
        DEPENDENT = 'NDVI'
        INDPENDENT = []
        
        dataset = self.dataset.process().get_dataset().select("NDVI")
        
        INDPENDENT.append('constant')
    
        # add the time
        dataset = dataset.map(self.add_time)
        INDPENDENT.append('t')
        
        # add the harmonics
        cos_freq = [f"cos_{i}" for i in range(1, modes + 1)]
        sin_freq = [f"sin_{i}" for i in range(1, modes + 1)]
        
        dataset = dataset.map(self.add_harmonics(cos_freq, sin_freq))
        INDPENDENT.extend(cos_freq)
        INDPENDENT.extend(sin_freq)
        
        # compute the coefficients
        linear_regression = (dataset.select(INDPENDENT + [DEPENDENT])
            .reduce(ee.Reducer.linearRegression(len(INDPENDENT), 1))
            .select("coefficients")
            .arrayFlatten([INDPENDENT, ["coef"]])
        )
        
        # add the coefficients to the dataset
        coef = linear_regression.select(".*coef")
        dataset = dataset.map(lambda image: image.addBands(coef))
        
        for mode in range(1, modes + 1):
            dataset = dataset.map(self.add_phase(mode))
            dataset = dataset.map(self.add_amplitude(mode))
        
        self.dataset = dataset
        
        return self
    
    def transform(self) -> ee.Image:
        """
        Transforms the Sentinel2 dataset by computing the median and scaling the values between -1 and 1.
        
        Returns:
            ee.Image: The transformed Sentinel2 dataset.
        """
        return self.dataset.median().unitScale(-1, 1)
    

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


## Sentinel - 2: helpers

def s2_compositing(geometry: ee.Geometry, cloudy_pix: int = 20) -> ee.Image:
 
    s2_2018 = Sentinel2(geometry, "2018-06-20", "2018-09-21") 
    s2_2019 = Sentinel2(geometry, "2019-06-20", "2019-09-21")
    s2_2020 = Sentinel2(geometry, "2020-06-20", "2020-09-21")
    
    s2 = s2_2018.merge(s2_2019).merge(s2_2020)
    
    s2.process(cloudy_pix)
    
    return s2.get_dataset().select(Sentinel2.BANDS + ["NDVI"]).median()


def s2_time_series_processing(aoi: ee.Geometry, modes: int = 3) -> ee.Image:
    """
    Process Sentinel-2 time series data for a given area of interest (AOI).

    Args:
        aoi (ee.Geometry): The area of interest.
        modes (int, optional): The number of harmonic modes to consider. Defaults to 3.

    Returns:
        ee.Image: The processed time series data. with phase 1 and phase 2 selected.
    """
    
    dataset = Sentinel2(aoi, "2018-06-20", "2020-09-21")
    time_series = Sentinel2TimeSeries(dataset=dataset)
    time_series = time_series.process(modes=modes)
    return time_series.transform().select("phase_1|phase_2")
    
    
    

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

## Sentinel-1

class Sentinel1:
    def __init__(self, aoi, start, end) -> None:
        self.aoi = aoi
        self.start = start
        self.end = end
        self.dataset = ee.ImageCollection("COPERNICUS/S1_GRD")

    def process(self, orbit: str = "DESCENDING"):
        look_direction = ['DESCENDING', 'ASCENDING']

        if orbit not in look_direction:
            raise ValueError(f"orbit must be one of {look_direction}")

        self.dataset = (
            self.dataset
            .filterDate(self.start, self.end)
            .filterBounds(self.aoi)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        return self

    def get_dataset(self) -> ee.ImageCollection:
        return self.dataset


def get_s1_inputs(aoi: ee.Geometry) -> ee.Image:
    from ykwmp import S1_DATASET
    """ The resulting dataframes need to have the same shape. or number of rows. """
    target_orbits = [50, 108, 79]
    # load the 2019 dataset
    gdf = gpd.read_file(S1_DATASET)
    # the target orbits # are 50, 108, 79
    swath_images: list[ee.Image] = []
    for idxs, gdf in gdf.groupby(["relativeOrbitNumber_start", "date"]):
        if idxs[0] in target_orbits:
            swath_images.append(ee.ImageCollection(gdf["system_id"].tolist()).mosaic())
    # We are only using images from SAT A
    mosaic = ee.ImageCollection(swath_images).filterBounds(aoi).mosaic()
    return mosaic


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## CDEM data

def cedm_processing(aoi: ee.Geometry) -> ee.Image:
    """ Process the CDEM data for a given area of interest (AOI). Adds elevation and slope bands."""
    dem = ee.ImageCollection("NRCan/CDEM").select('elevation').mosaic()
    slope = ee.Terrain.slope(dem)
    dem = dem.addBands(slope).clip(aoi)
    return dem

    
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##