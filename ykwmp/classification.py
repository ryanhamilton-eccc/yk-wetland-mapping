from dataclasses import dataclass
from typing import Any, List, Union, Tuple
import json
import ee
import pandas as pd
import numpy as np


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


def get_ee_predictors(object: ee.List | ee.Image | ee.Feature | ee.FeatureCollection):
    props_to_remove = ['is_training', 'class_name', 'class_value', 'system:index']
    if isinstance(object, ee.Image):
        return object.bandNames()
    if isinstance(object, ee.FeatureCollection):
        object = object.first().propertyNames()
    if isinstance(object, ee.Feature):
        object = object.propertyNames()
    return object.removeAll(props_to_remove)


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
    order: ee.List


def compute_accuracy_metrics(
    model: ee.Classifier, 
    validation: ee.FeatureCollection, 
    actual: str, 
    predicted: str = "classification"
) -> ee.ConfusionMatrix:
    """
    Computes the confusion matrix and accuracy metrics within Earth Engine.
    """
    order = validation.aggregate_array(actual).distinct()
    predictions = validation.classify(model)  # Classify the validation set
    error_matrix = predictions.errorMatrix(actual, predicted, order=order)
    
    # Extract additional metrics like overall accuracy, Kappa, etc.
    accuracy = error_matrix.accuracy()  # Overall accuracy
    kappa = error_matrix.kappa()  # Kappa coefficient
    producers = error_matrix.producersAccuracy()
    consumers = error_matrix.consumersAccuracy()
    
    # Optionally, you can return the confusion matrix along with metrics
    return AccuracyMetrics(
        confusion_matrix=error_matrix,
        accuracy=accuracy,
        kappa=kappa,
        producers=producers,
        consumers=consumers,
        order=order
    )


def create_accuracy_feature_collection(metrics: AccuracyMetrics):
    table_compoents = [
        ee.Feature(None, {"confusion_matrix": metrics.confusion_matrix.array()}),
        ee.Feature(None, {"overall": metrics.accuracy}),
        ee.Feature(None, {"producers": metrics.producers.toList().flatten()}),
        ee.Feature(None, {"consumers": metrics.consumers.toList().flatten()}),
        ee.Feature(None, {"order": metrics.order}),
        ee.Feature(None, {"kappa": metrics.kappa})
    ]
    return ee.FeatureCollection(table_compoents)


def build_metrics_table(datafile: str, labels: list[str] = None) -> pd.DataFrame:
    with open(datafile, "r") as f:
        data = json.load(f)

    features = data["features"]
    props = [_.get("properties") for _ in features]
    data = {k: v for _ in props for k, v in _.items()}

    # if labels are not provided, use the order of the classes
    if not labels:
        labels = data.get("order")

    # this construct the base table for the cond
    cfm = pd.DataFrame(
        data=data.get("matrix"),
        columns=labels,
        index=labels,
    )

    # add producers to the base table
    producers = data.get("producers")
    cfm = cfm.reindex(columns=cfm.columns.tolist() + ["Producers"])
    pro = list(map(lambda x: round(x * 100, 2), producers))
    cfm["Producers"] = pro

    # add consumers to the base table
    consumers = data.get("consumers")
    new_index = pd.Index(cfm.index.tolist() + ["Consumers"])
    cfm = cfm.reindex(new_index).fillna(value=np.nan)
    cfm.iloc[-1, 0:-1] = list(map(lambda x: round(x * 100), consumers))

    # insert overall accuracy
    overall = data.get("overall")
    cfm = cfm.reindex(cfm.index.tolist() + ["Overall Accuracy"]).fillna(value=np.nan)
    cfm.iloc[-1, 0] = round(overall * 100, 2)

    return cfm