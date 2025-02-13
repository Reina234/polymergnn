from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class PreprocessorStrategy(ABC):
    """Abstract base class for preprocessing strategies."""

    @abstractmethod
    def fit(self, data):
        """Fits the preprocessing strategy on the dataset."""
        pass

    @abstractmethod
    def transform(self, data):
        """Applies the preprocessing strategy."""
        pass

    def fit_transform(self, data):
        """Convenience method to fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class NoDataPreprocessing(PreprocessorStrategy):
    """Dummy strategy (does nothing)."""

    def fit(self, data):
        pass

    def transform(self, data):
        return data


class StandardScalerPreprocessor(PreprocessorStrategy):
    """StandardScaler (Mean 0, Std 1)"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)


class MinMaxScalerPreprocessor(PreprocessorStrategy):
    """Min-Max Scaler (0 to 1)"""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)
