from __future__ import annotations
import ee

from math import pi


class RemoteSensingDatasetProcessor(object):
    def __init__(self, dataset: str | list[str], start_date: str, end_date: str, bounds: ee.Geometry) -> None:
        self.dataset = ee.ImageCollection(dataset)
        self.start_date = start_date
        self.end_date = end_date
        self.bounds = bounds
    
    def __add__(self, other: RemoteSensingDatasetProcessor) -> RemoteSensingDatasetProcessor:
        self.dataset = self.dataset.merge(other.dataset)
        return self
    
    def filter_by_date(self) -> RemoteSensingDatasetProcessor:
        self.dataset = self.dataset.filterDate(self.start_date, self.end_date)
        return self

    def filter_by_bounds(self) -> RemoteSensingDatasetProcessor:
        self.dataset = self.dataset.filterBounds(self.bounds)
        return self
    
    def filter_by_cloud_cover(self, meta_flag: str, max_cloud_cover: float) -> RemoteSensingDatasetProcessor:
        self.dataset = self.dataset.filter(ee.Filter.lt(meta_flag, max_cloud_cover))
        return self
    
    def select_bands(self, bands: list[str]|str) -> RemoteSensingDatasetProcessor:
        self.dataset = self.dataset.select(bands)
        return self


class Sentinel2(RemoteSensingDatasetProcessor):
    def __init__(self, start_date: str, end_date: str, bounds: ee.Geometry) -> None:
        super().__init__("COPERNICUS/S2_HARMONIZED", start_date, end_date, bounds)
    
    def add_ndvi(self) -> RemoteSensingDatasetProcessor:
        def compute_ndvi(image: ee.Image):
            return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        self.dataset = self.dataset.map(compute_ndvi)
        return self
    
    def process(self):
        return (self
            .filter_by_date()
            .filter_by_bounds()
            .filter_by_cloud_cover("CLOUDY_PIXEL_PERCENTAGE", 20)
            .select_bands(["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"])
            .add_ndvi()
        )


class FourierTransform:
    def __init__(self, collection, dependent, modes: int = 3) -> None:
        self.rsd = collection
        self.dependent = dependent
        self.modes = modes
        self.independent = []
    
    def add_constant(self) -> FourierTransform:
        self.rsd = self.rsd.map(lambda image: image.addBands(ee.Image.constant(1)))
        self.independent.append('constant')
        return self

    def add_time(self) -> FourierTransform:
        def wrapper(image: ee.Image):
            date = image.date()
            years = date.difference(ee.Date("1970-01-01"), "year")
            time_radians = ee.Image(years.multiply(2 * pi))
            return image.addBands(time_radians.rename("t").float())
        self.add_independent('t')
        self.rsd = self.rsd.map(wrapper)
        return self
    
    def add_harmonics(self):
        cos = self._mk_frequency_name("cos", self.modes)
        sin = self._mk_frequency_name("sin", self.modes)
        def wrapper(image: ee.Image):
            frequencies = ee.Image.constant(list(range(1, self.modes + 1)))
            time = ee.Image(image).select("t")
            cos_img = time.multiply(frequencies).cos().rename(cos)
            sin_img = time.multiply(frequencies).sin().rename(sin)
            return image.addBands(cos_img).addBands(sin_img)
        self.independent.extend(cos)
        self.independent.extend(sin)
        self.rsd = self.rsd.map(wrapper)
        return self
    
    def add_coefficients(self) -> FourierTransform:
        linear_regression = (self.rsd.select(self.independent + [self.dependent])
            .reduce(ee.Reducer.linearRegression(len(self.independent), 1))
            .select("coefficients")
            .arrayFlatten([self.independent, ["coef"]])
        )
        coef = linear_regression.select(".*coef")
        self.rsd = self.rsd.map(lambda image: image.addBands(coef))
        return self
    
    def add_phase(self, mode):
        name = f"phase_{mode}"
        def compute_phase(image: ee.Image):
            sin = image.select(f"sin_{mode}_coef")
            cos = image.select(f"cos_{mode}_coef")
            phase = sin.atan2(cos).rename(name)
            return image.addBands(phase)
        self.rsd = self.rsd.map(compute_phase)
        return self
    
    def add_amplitude(self, mode):
        name = f"amp_{mode}"
        def compute_amplitude(image: ee.Image):
            sin = image.select(f"sin_{mode}_coef")
            cos = image.select(f"cos_{mode}_coef")
            amplitude = sin.hypot(cos).rename(name)
            return image.addBands(amplitude)
        self.rsd = self.rsd.map(compute_amplitude)
        return self
    
    def compute(self) -> FourierTransform:
        self.add_constant()
        self.add_time()
        self.add_harmonics()
        self.add_coefficients()
        for i in range(1, self.modes + 1):
            self.add_phase(i)
            self.add_amplitude(i)
        return self
        
    @staticmethod
    def _mk_frequency_name(name, modes):
        return [f"{name}_{i}" for i in range(1, modes + 1)]

    