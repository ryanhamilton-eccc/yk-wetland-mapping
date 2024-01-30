from dataclasses import dataclass
import ee


class SmileRandomForest:
    model = None
    
    def fit(self, features, class_col, predictors, **kwargs):
        self.model = ee.Classifier.smileRandomForest(**kwargs).train(features, class_col, predictors)
        return self
    
    def predict(self, ee_object: ee.Image | ee.FeatureCollection):
        if self.model is None:
            raise ValueError("Model is not trained yet")
        return ee_object.classify(self.model)

    def save(self, path):
        if self.model is None:
            raise ValueError("Model is not trained yet")
        return ee.Export.classifier.toAsset(
            classifier=self.model,
            description="SmileRandomForest",
            assetId=path
        ).start()
    
    @classmethod
    def load(cls, path):
        cls.model = ee.Classifier.load(path)
        return cls


import json
import ee
import pandas as pd
import numpy as np


class Metrics:
    """
    A class for computing metrics from predictions and actual values using Earth Engine.

    Args:
        predictions (ee.FeatureCollection): The predicted values.
        actual (str): The actual values.
        predicted (str, optional): The predicted values column name. Defaults to "classification".
        class_order (list[str] | ee.List, optional): The order of classes. Defaults to None.

    Attributes:
        actual (str): The actual values.
        predicted (str): The predicted values column name.
        class_order (list[str] | ee.List): The order of classes.
        matrix (ee.FeatureCollection): The error matrix.

    Properties:
        producers (ee.List): The producer's accuracy.
        consumers (ee.List): The consumer's accuracy.
        overall (ee.Number): The overall accuracy.
        kappa (ee.Number): The kappa coefficient.

    Methods:
        create_metrics(): Creates a feature collection containing the metrics.

    """

    def __init__(
        self,
        predictions: ee.FeatureCollection,
        actual: str,
        predicted: str = None,
        class_order: list[str] | ee.List = None,
    ) -> None:
        self.actual = actual
        self.predicted = predicted or "classification"
        self.class_order = class_order
        self.matrix = predictions

    @property
    def matrix(self):
        """
        Get the error matrix.

        Returns:
            ee.FeatureCollection: The error matrix.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, predictions: ee.FeatureCollection) -> None:
        """
        Set the error matrix.

        Args:
            predictions (ee.FeatureCollection): The predicted values.
        """
        self._matrix = predictions.errorMatrix(
            self.actual, self.predicted, self.class_order
        )

    @property
    def producers(self) -> ee.List:
        """
        Get the producer's accuracy.

        Returns:
            ee.List: The producer's accuracy.
        """
        return self.matrix.producersAccuracy().toList().flatten()

    @property
    def consumers(self) -> ee.List:
        """
        Get the consumer's accuracy.

        Returns:
            ee.List: The consumer's accuracy.
        """
        return self.matrix.consumersAccuracy().toList().flatten()

    @property
    def overall(self) -> ee.Number:
        """
        Get the overall accuracy.

        Returns:
            ee.Number: The overall accuracy.
        """
        return self.matrix.accuracy()

    @property
    def kappa(self) -> ee.Number:
        """
        Get the kappa coefficient.

        Returns:
            ee.Number: The kappa coefficient.
        """
        return self.matrix.kappa()

    def create_metrics(self) -> ee.FeatureCollection:
        """
        Create a feature collection containing the metrics.

        Returns:
            ee.FeatureCollection: The feature collection containing the metrics.
        """
        return ee.FeatureCollection(
            [
                ee.Feature(None, {"matrix": self.matrix.array()}),
                ee.Feature(None, {"overall": self.overall}),
                ee.Feature(None, {"producers": self.producers}),
                ee.Feature(None, {"consumers": self.consumers}),
                ee.Feature(None, {"order": self.class_order}),
            ]
        )


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