import os
import sys
from pathlib import Path

import ee

from ykwmp import rsd    

PROJECT_ROOT = "projects/yk-wetland-mapping/assets/tmp"


def make_asset_root(asset_id: str) -> None:
    try:
        ee.data.createAsset({'type': 'FOLDER'}, asset_id)
    except ee.ee_exception.EEException:
        print(f'Warning: {asset_id} already exists')


def dataset_from_textfile(textfile: str) -> ee.ImageCollection:
    with open(textfile, 'r') as f:
        uris = f.read().splitlines()
    
    images = [ee.Image.loadGeoTIFF(_) for _ in uris]
    return ee.ImageCollection(images)


def image_processing(aoi) -> ee.Image:
    stack = []
    
    ## s1 processing
    s1 = rsd.s1_mosaics()
    stack.append(s1)
    
    ## s2 processing
    s2_2018 = rsd.Sentinel2(aoi, start='2018-06-20', end='2018-09-21').process()
    s2_2019 = rsd.Sentinel2(aoi, start='2019-06-20', end='2019-09-21').process()
    s2_2020 = rsd.Sentinel2(aoi, start='2020-06-20', end='2020-09-21').process()
    s2_toa_composite = s2_2018.merge(s2_2019).merge(s2_2020).get_dataset().median()
    stack.append(s2_toa_composite)
    
    ## CDEM processing
    cdem = dataset_from_textfile("../data/cdem.txt").filterBounds(aoi).select(['Band_1'], ['elevation'])
    cdem = cdem.map(lambda x: x.addBands(ee.Terrain.slope(x)))
    stack.append(cdem.mosaic())

    ## Fourier Transform
    # fourier = dataset_from_textfile('fourier.txt').filterBounds(aoi).mosaic()
    
    return ee.Image.cat(*stack)


def monitor_task(task: ee.batch.Task) -> int:
    import time
    while task.status()['state'] in ['READY', 'RUNNING']:
        time.sleep(5)
    
    status_code = {
        'COMPLETED': 0,
        'FAILED': 1,
        'CANCELLED': 2
    }
    
    return status_code[task.status()['state']]


def reset_processing_env():
    ee.Reset()
    ee.Initialize()
    return None


def main() -> int:
    # step 1: reset processing environment
    # load the assets
    
    features_asset_id = "projects/yk-wetland-mapping/assets/yk-tmp/features"
    aoi_asset_id = "projects/yk-wetland-mapping/assets/yk-tmp/region"
    
    make_asset_root(PROJECT_ROOT)
    
    features = ee.FeatureCollection(features_asset_id)
    aoi = ee.FeatureCollection(aoi_asset_id).geometry()

    # step 2: create the features to model
    # TODO: add some logic to skip this step if the samples already exist
    samples_asset_id = f"{PROJECT_ROOT}/samples_{features_asset_id.split('/')[-1]}"
    stack = image_processing(features)   
    samples = stack.sampleRegions(**{
        'collection': features,
        'scale': 10,
        'geometries': True,
        'tileScale': 16
    })
    samples_task = ee.batch.Export.table.toAsset(**{
        'collection': samples,
        'description': '',
        'assetId': samples_asset_id
    })
    
    samples_task.start()
    print('Exporting samples...')
    status = monitor_task(samples_task)
    if status != 0:
        print('Error: Failed to export samples')
        sys.exit(1)
    print('Samples exported successfully')
    
    reset_processing_env()
    
    # step 3: train the model
    model = ee.Classifier.smileRandomForest(1000).train(**{
        'features': samples.filter(ee.Filter.eq('split', 'train')),
        'classProperty': 'class_name',
        'inputProperties': stack.bandNames()
    })
    
    ## save model to asset
    model_asset_id = f"{PROJECT_ROOT}/model_{features_asset_id.split('/')[-1]}"
    model_task = ee.batch.Export.classifier.toAsset(**{
        'classifier': model,
        'description': '',
        'assetId': model_asset_id
    })
    
    model_task.start()
    
    # TODO: Implement assessment of model
    
    print('Exporting model...')
    status = monitor_task(model_task)
    if status != 0:
        print('Error: Failed to export model')
        sys.exit(1)
    print('Model exported successfully')
    
    reset_processing_env()
    
    load_model = ee.Classifier.load(model_asset_id)
    aoi = ee.FeatureCollection(aoi_asset_id).geometry()
    stack = image_processing(aoi)
    classified = stack.classify(load_model)
    
    image_task = ee.batch.Export.image.toDrive(
        image=classified,
        description='',
        folder='yk-wetland-mapping-testing',
        formatOptions={
            'cloudOptimized': True
        },
        fileDimensions=[2048, 2048],
        region=aoi,
        maxPixels=1e13,
        skipEmptyTiles=True,
        scale=10,
        fileNamePrefix='yk-wetland-mapping-testing-'
    )
    
    image_task.start()
    print('Exporting image...')
    print('This may take a while')
    
    return 0


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    ee.Initialize()
    sys.exit(main())
    