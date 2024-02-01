import ee

from . import rsd


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

def assess_model(model, samples: ee.FeatureCollection):
    test_predictions = samples.classify(model)
    order = samples.aggregate_array('class_name').distinct()
    error_matrix = test_predictions.errorMatrix('class_name', 'classification', order)
    overall_accuracy = error_matrix.accuracy()
    producers = error_matrix.producersAccuracy()
    consumers = error_matrix.consumersAccuracy()
    
    table = [
        ee.Feature(None, {"matrix": error_matrix.array()}),
        ee.Feature(None, {"overall": overall_accuracy}),
        ee.Feature(None, {"producers": producers.toList().flatten()}),
        ee.Feature(None, {"consumers": consumers.toList().flatten()}),
        ee.Feature(None, {"order": order}),
    ]
    
    return ee.FeatureCollection(table)

def reset_processing_env():
    ee.Reset()
    ee.Initialize()
    return None
    
    