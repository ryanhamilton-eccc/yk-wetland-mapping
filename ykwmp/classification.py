from dataclasses import dataclass
from typing import Any, List, Union
import ee


def randomforest(
    features: ee.FeatureCollection,
    class_property: str, 
    input_properties: Union[List[str], ee.List], 
    number_of_trees: int = 1000,
) -> ee.Classifier:
    rf_model = ee.Classifier.smileRandomForest(number_of_trees).train(features, class_property, input_properties)
    return rf_model


def predict(X: Union[ee.Image, ee.FeatureCollection], model: ee.Classifier):
    return X.classify(model)


# -- Accuracy Asessment
@dataclass(frozen=True)
class AccuracyMetrics:
    confusion_matrix: ee.ConfusionMatrix
    accuracy: ee.Number
    kappa: ee.Number
    producers: ee.Array
    consumers: ee.Array


def compute_accuracy_metrics(
    model: ee.Classifier, 
    validation: ee.FeatureCollection, 
    actual: str, 
    predicted: str = "classification"
) -> ee.ConfusionMatrix:
    """
    Computes the confusion matrix and accuracy metrics within Earth Engine.
    """
    predictions = model.classify(validation)  # Classify the validation set
    error_matrix = predictions.errorMatrix(actual, predicted)
    
    # Extract additional metrics like overall accuracy, Kappa, etc.
    accuracy = error_matrix.accuracy()  # Overall accuracy
    kappa = error_matrix.kappa()  # Kappa coefficient
    producers = None
    consumers = None
    
    # Optionally, you can return the confusion matrix along with metrics
    return AccuracyMetrics(
        confusion_matrix=error_matrix,
        accuracy=accuracy,
        kappa=kappa,
        producers=producers,
        consumers=consumers
    )


def create_accuracy_feature_collection(metrics: AccuracyMetrics):
    pass