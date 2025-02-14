from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DatasetTransformer(ABC):
    """Abstract base class for preprocessing strategies."""

    @abstractmethod
    def fit(self, data):
        """Fits the preprocessing strategy on the dataset."""

    @abstractmethod
    def transform(self, data):
        """Applies the preprocessing strategy."""

    def fit_transform(self, data):
        """Convenience method to fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class NoDataTransform(DatasetTransformer):
    """Dummy strategy (does nothing)."""

    def fit(self, data):
        pass

    def transform(self, data):
        return data


class StandardScalerTransform(DatasetTransformer):
    """StandardScaler (Mean 0, Std 1)"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)


class MinMaxScalerTransform(DatasetTransformer):
    """Min-Max Scaler (0 to 1)"""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)
