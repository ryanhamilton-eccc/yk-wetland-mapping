import os
import sys

from pathlib import Path
from pprint import pprint

import ee

def main(args: list[str]) -> int:
    if len(args) != 2:
        print('Usage: python create_cloud_backed_asset.py <uris_file>')
        return 1

    
    uris_file = args[1]
    with open(uris_file, 'r') as f:
        uris = f.read().splitlines()
    
    cogs = [ee.Image.loadGeoTIFF(uri) for uri in uris]
    collection = ee.ImageCollection(cogs)
    
    pprint(collection.first().getInfo())


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    ee.Initialize()
    sys.exit(main(sys.argv))