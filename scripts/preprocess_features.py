"""
preprocess_features.py

This script is used to preprocess the psudo features for the wetlands project.
"""
import os
import sys


from pathlib import Path
from shutil import copy2

import geopandas as gpd
import pandas as pd


def main():
    """Main entry point for the script."""
    
    if not Path("../sratch/data").exists():
        Path("../scratch/data").mkdir(parents=True)
    
    gdf = gpd.read_file("../data/shapefiles/175_120_pnts.shp")
    
    gdf['class_name'] = ""
    class_names = ['wetland', 'non-wetland', 'water']
    
    for i in class_names:
        _ = gdf[gdf['class_name'] == ""]
        subset = _.sample(n=40, random_state=1)
        subset['class_name'] = i
        gdf.update(subset)
    
    gdf.reset_index(drop=True, inplace=True)
    
    gdf['split'] = 'train'
    gdf.loc[gdf.sample(frac=0.3, random_state=1).index, 'split'] = 'test'
    
    # save the dataframes
    gdf_train = gdf[gdf['split'] == 'train']
    gdf_train.reset_index(inplace=True, drop=True)
    gdf_train.to_file("../scratch/data/trainingPoints.shp")
    
    gdf_val = gdf[gdf['split'] == 'test']
    gdf_val.reset_index(inplace=True, drop=True)
    gdf_val.to_file("../scratch/data/validationPoints.shp")
    
    region = gpd.read_file("../data/shapefiles/175.shp")
    region.to_file("../scratch/data/region.shp")
    
    return 0

if __name__ == '__main__':
    # os.chdir(Path(__file__).parent)
    sys.exit(main())