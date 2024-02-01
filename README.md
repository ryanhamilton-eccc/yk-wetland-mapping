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