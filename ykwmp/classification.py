from dataclasses import dataclass
from typing import Any, List, Union, Tuple
import ee


# -- Random Forest
@dataclass(frozen=True)
class RandomForestHyperparameters:
    number_of_trees: int = 1000


@dataclass(frozen=True)
class FeatureInputs:
    features: ee.FeatureCollection
    class_property: str
    input_properties: Union[List[str], ee.List]


def partition_feature_collection(
    features: ee.FeatureCollection,
    partition_column: str,
    train_value: Any,
    validation_value: Any
) -> Tuple[ee.FeatureCollection, ee.FeatureCollection]:
    """
    Partitions a FeatureCollection into training and validation sets based on a column.

    Parameters:
    - features: ee.FeatureCollection containing the data.
    - partition_column: The column used to identify train and validation rows.
    - train_value: The value in `partition_column` that indicates training rows.
    - validation_value: The value in `partition_column` that indicates validation rows.

    Returns:
    - Tuple of ee.FeatureCollection (train, validation)
    """
    train_features = features.filter(ee.Filter.eq(partition_column, train_value))
    validation_features = features.filter(ee.Filter.eq(partition_column, validation_value))
    return train_features, validation_features


def get_predictors(object: ee.List | ee.Image | ee.Feature | ee.FeatureCollection):
    if isinstance(object, ee.Image):
        return object.bandNames()
    # TODO implement the rest

def randomforest(
    feature_inputs: FeatureInputs,
    hyperparameters: RandomForestHyperparameters,
) -> ee.Classifier:
    """
    Trains a Random Forest model using Earth Engine's smileRandomForest classifier.

    Parameters:
    - feature_inputs: FeatureInputs object containing training features, target property, and predictors.
    - hyperparameters: RandomForestHyperparameters object containing model configuration.

    Returns:
    - ee.Classifier: A trained Random Forest classifier.
    """
    rf_model = (
        ee.Classifier.smileRandomForest(hyperparameters.number_of_trees)
        .train(
            feature_inputs.features,
            feature_inputs.class_property,
            feature_inputs.input_properties,
        )
    )
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