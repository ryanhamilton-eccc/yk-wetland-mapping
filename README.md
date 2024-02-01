# yk-wetland-mapping
Yukon Wetland Mapping 

## Installation

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution)
- [Google Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)

### Conda Dependencies
- [earthengine-api](https://anaconda.org/conda-forge/earthengine-api)
- [geopandas](https://geopandas.org/)

### Steps

```bash
1. git clone https://github.com/ryanhamilton-eccc/yk-wetland-mapping.git
```
```bash
2. cd yk-wetland-mapping
```
```bash
3. conda env create -n ykwmp -f environment.yml
```
```bash
4. conda activate ykwmp
```
```bash
5. python -m pip install -e .
```

## Usage
To launch the application, run the following command:

```bash
(ykwmp) $ python -m ykwmp <feature_asset_id> <region_asset_id>
```

## Remote Sensing Datasets
### [Sentinel-1](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD)
#### Processing Steps
- input year: 2019
- Query Date Range: 2019-06-20 to 2019-09-21
- Polarization: VV, VH
- Resolution: 10m
- Mode: IW
- Orbit: Ascending (only complete coverage)
- 3x3 boxcar filter
### Notes
- Image Collection was converted to a feature collection. and then written to an disk as a geojson file.
- geopandas was used to group the data by the relative orbit number start and the date. Only dataframes that had the same shape (i.e. the lengths matched) where retained as valid images to consturct a mosaic from. If the number of images differed between two swaths the mosaic could not be constructed (as in memeory the images need to be the same). This methodolgy works for 95% of the Yukon territory. The remaining 5% of the territory is covered by another swath (137) which is missing two accaqusions.
### [Sentinel-2](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2)
#### Processing Steps
- input years: 2018, 2019, 2020
- Query Date Range: 2019-06-20 to 2019-09-21
- Bands: B2, B3, B4, B8, B11, B12
- Resolution: 10m (B2, B3, B4, B8), 20m (B11, B12)
- Cloud Masking: 20% cloud cover
- Added NDVI, Brightness, Greenness, Wetness
### Notes
- 3 year median composite was created. This was done to reduce the impact of cloud cover and to create a more representative image of the landscape.
- Sentinel 2 Top of Atmosphere (TOA) collection was used because it reaches farther back in time than SR and is a more complete collection when compared to the SR.
### [Elevation](https://yukon.ca/en/statistics-and-data/mapping/view-yukon-elevation-data)
#### Processing Steps
- CDEM tiles over the Yukon were downloaded from the [Yukon Elevation Data FTP](https://map-data.service.yukon.ca/Elevation/Canadian%20Digital%20Elevation%20Model%20(CDEM)/).
- The tiles were converted to cog using the `gdal_translate` command. a helper script was used to automate the process.
- The cog files were then uploaded to a google cloud storage bucket for the project.
### Notes
- CDEM collection nativly available in GEE because it does not play well when trying to create additional bands.
### Fourier Transform
#### Processing Steps
- Fourier Transform was applied to NDVI from the Sentinel-2 collection.
- 3 modes were computed Phase 1 and Phase 2 where retained.
### Notes
- Fourier Transform was computed using the CDEM grid as the bounding geometry. This was done to ensure that the Fourier Transform was computed on the same grid as the other datasets.
