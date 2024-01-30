import os

import ee
import geopandas as gpd
import pandas as pd

from pathlib import Path
from pprint import pprint


pd.set_option("display.max_columns", None)


def main():
    data = Path("../data/geojson/s1_asc_2019.geojson")
    
    gdf = gpd.read_file(data)
    data = {}
    # handle organizing the swaths by relative orbit, and Platform number and date
    group = gdf.groupby(["relativeOrbitNumber_start", 'platform_number', 'date'])
    for _, group in group:
        rel, platform, date = _[0], _[1], _[2]

        bands = ['VV', 'VH'], [f'VV_{"".join(date.split("-"))}', f'VH_{"".join(date.split("-"))}']
        mosaic = ee.ImageCollection(group['system_id'].tolist()).mosaic().select(*bands)
        
        # if the key does not exist, create it
        
        # sets the data structure for storage        
        if data.get(rel) is None:
            data[rel] = {}

        if data[rel].get(platform) is None:
            # add the platform number
            data[rel][platform] = []
        # add the date
        d = {'date': date, 'mosaic': mosaic}
        data[rel][platform].append(d)
         
    # handles the stacking of time series images
    swaths = []
    for rel, platforms in data.items():
        stack = None
        for platform, d in platforms.items():
            print(platform)
            for i in d:
                if stack is None:
                    stack = i['mosaic'] 
                else:
                    stack = stack.addBands(i['mosaic'])
        swaths.append(stack)
    
    mosaic = ee.ImageCollection(swaths).mosaic()
    return mosaic    
        
if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    ee.Initialize()
    main()