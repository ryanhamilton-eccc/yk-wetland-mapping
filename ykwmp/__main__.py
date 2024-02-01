import os
import sys

from pathlib import Path
from pprint import pprint

import ee

from . import helpers

DEBUG = False # if set to True, the working directory will be changed to the directory of this file

# print('DEBUG_MODE:', DEBUG)

def main(args: list[str]) -> int:
    """
    Info that needs to be injected into the main function
    
    feature dataset we want to model and the target region
    
    the program will infer the output asset folder from the root of the feature dataset
    
    i.e. porjects/yk-wetland-mapping/assets/yk-tmp/features
    the output destination will be projects/yk-wetland-mapping/assets/yk-tmp/
    the output asset ids will be build off this.
    
    the asset will need to have a meaningful name, so we can infer the name of the 
    output asset from the input asset
    
    nameing convention:
    {name}_{object_type}
    for example:
        features -> features_176
        sample -> samples_features_176
        model -> model_features_176
        assessment -> assessment_features_176
        classification -> classification_features_176
    """
    # step 0: validate user input
    if len(args) < 2:
        print('Usage: ykwmp <features_asset_id> <aoi_asset_id>')
        return 1
    
    # step 1: to inital setup
    features_id = args[0]
    aoi_id = args[1]
    destination = '/'.join(features_id.split('/')[:-1])
    
    # step 2: create the features to model
    ## load the assets
    print('Loading Features to Extract from...')
    features = ee.FeatureCollection(features_id)
    ## create the datasets we want to extract from
    print('Creating Samples...')
    stack = helpers.image_processing(features)
    ## extract the samples
    samples = stack.sampleRegions(**{
        'collection': features,
        'scale': 10,
        'geometries': True,
        'tileScale': 16
    })
    
    print('Exporting samples...')
    samples_asset_id = f"{destination}/samples_{features_id.split('/')[-1]}"
    
    # save the samples to asset
    samples_task = ee.batch.Export.table.toAsset(**{
        'collection': samples,
        'description': '',
        'assetId': samples_asset_id
    })
    
    samples_task.start()
    print('Exporting samples...')
    status = helpers.monitor_task(samples_task)
    if status != 0:
        print('Error: Failed to export samples')
        sys.exit(1)
    print('Samples exported successfully')
    
    helpers.reset_processing_env()
    
    # step 3: train the model
    model = ee.Classifier.smileRandomForest(1000).train(**{
        'features': samples.filter(ee.Filter.eq('split', 'train')),
        'classProperty': 'class_name',
        'inputProperties': stack.bandNames()
    })
    
    # do assessment
    test = samples.filter(ee.Filter.eq('split', 'test'))
    helpers.assess_model(model, test)
    
    model_asset_id = f"{destination}/model_{features_id.split('/')[-1]}"
    model_task = ee.batch.Export.classifier.toAsset(**{
        'classifier': model,
        'description': '',
        'assetId': model_asset_id
    })
    
    model_task.start()
    status = helpers.monitor_task(model_task)
    
    print('Exporting model...')
    if status != 0:
        print('Error: Failed to export model')
        sys.exit(1)
    print('Model exported successfully')
    
    helpers.reset_processing_env()
    
    # step 4: classify the image
    load_model = ee.Classifier.load(model_asset_id)
    aoi = ee.FeatureCollection(aoi_id).geometry()
    stack = helpers.image_processing(aoi)
    classified = stack.classify(load_model)
    
    image_task = ee.batch.Export.image.toDrive(
        image=classified,
        description=f'classification_{features_id.split("/")[-1]}',
        folder="",
        formatOptions={
            'cloudOptimized': True
        },
        fileDimensions=[2048, 2048],
        region=aoi,
        maxPixels=1e13,
        skipEmptyTiles=True,
        scale=10,
        fileNamePrefix=f'classification_{features_id.split("/")[-1]}-'
    )
    
    image_task.start()
    print('Exporting image...')
    print("to monitor the task, go to: https://code.earthengine.google.com/tasks")
    print("Pipeline Done: Exiting...")

    return 0


if __name__ == '__main__':
    if DEBUG:
        os.chdir(Path(__file__).parent)
    
    ee.Initialize()
    sys.exit(main(sys.argv[1:]))