from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DatasetTransformer(ABC):
    """Abstract base class for preprocessing strategies."""

    @property
    @abstractmethod
    def scaler(self):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data  # Default: Do nothing (e.g., for NoDataTransform)


class NoDataTransform(DatasetTransformer):
    """Dummy strategy (does nothing)."""

    @property
    def scaler(self):
        return None

    def fit(self, data):
        return NoDataTransform()

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardScalerTransform(DatasetTransformer):
    """StandardScaler (Mean 0, Std 1)"""

    def __init__(self):
        self._scaler = StandardScaler()

    @property
    def scaler(self):
        return self._scaler

    def fit(self, data):
        self.scaler.fit(data)
        return self.scaler

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MinMaxScalerTransform(DatasetTransformer):
    """Min-Max Scaler (0 to 1)"""

    def __init__(self):
        self._scaler = MinMaxScaler()

    @property
    def scaler(self):
        return self._scaler

    def fit(self, data):
        self.scaler.fit(data)
        return self.scaler

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
