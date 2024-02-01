import sys

import ee
import geopandas as gpd

from pathlib import Path


def get_s1(aoi: ee.Geometry, start_date: str, end_date: str, lookdir: str = None) -> ee.ImageCollection:
    """Get Sentinel-1 GRD data for a given area and time period.

    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        ee.ImageCollection: Sentinel-1 GRD data.
    """
    if lookdir not in ['ASCENDING', 'DESCENDING']:
        raise ValueError('lookdir must be either ASCENDING or DESCENDING')
    
    s1 = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('orbitProperties_pass', lookdir))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .select(['VV', 'VH'])
        .map(lambda img: img.set({'date':ee.Image(img).date().format('YYYY-MM-dd')}))
        .map(lambda img: img.set('system_id', ee.String('COPERNICUS/S1_GRD/').cat(ee.String(ee.Image(img).id()))))
    )
    return s1


def img2feature(element: ee.computedobject):
    img = ee.Image(element)
    return ee.Feature(img.geometry(), img.toDictionary())
    
    
def main(args: list = None):
    if len(args) != 2:
        print('Usage: python extract_s1.py <year>')
        return 1
    print(f'Extracting S1 data for {args[1]}')
    aoi = ee.FeatureCollection("projects/yk-wetland-mapping/assets/yukon-bounds").geometry().bounds()
    col = None
    
    start_mm_dd = f'{args[1]}-06-20'
    end_mm_dd = f'{args[1]}-09-22'
    s1 = get_s1(aoi, start_mm_dd, end_mm_dd, lookdir='ASCENDING')
    
    # convert to feature collection
    
    fc = ee.FeatureCollection(s1.toList(s1.size()).map(img2feature))
    
    out_path = Path('../data/geojson')
    if not out_path.exists():
        out_path.mkdir(parents=True)

    gdf = gpd.GeoDataFrame.from_features(fc.getInfo()['features'])
    gdf['transmitterReceiverPolarisation'] = gdf['transmitterReceiverPolarisation'].astype(str)
    gdf.to_file(out_path / f's1_asc_176_{sys.argv[1]}.geojson', driver='GeoJSON')


if __name__ == '__main__':
    ee.Initialize()
    sys.exit(main(sys.argv))
        

        
