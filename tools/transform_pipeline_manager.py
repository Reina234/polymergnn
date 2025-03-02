import numpy as np
from sklearn.pipeline import Pipeline
from typing import List, Optional, Tuple
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import copy


def log_transform(x):
    return np.log(x)  # log(1 + x) to avoid log(0) issues


def inverse_log_transform(x):
    return np.exp(x)  # exp(x) - 1 to reverse log1p


def no_log(x):
    return x


def no_inverse_log(x):
    return x


class TransformPipelineManager:

    def __init__(
        self,
        feature_indexes: Optional[List[int]] = None,
        target_indexes: Optional[List[int]] = None,
        log_transform_targets: Optional[bool] = True,
    ):
        self.feature_indexes = feature_indexes if feature_indexes is not None else []
        self.target_indexes = target_indexes if target_indexes is not None else []

        self.num_features = len(self.feature_indexes)
        self.num_targets = len(self.target_indexes)

        self.feature_pipelines = (
            [None] * self.num_features if self.num_features > 0 else []
        )
        self.target_pipelines = (
            [None] * (2 * self.num_targets) if self.num_targets > 0 else []
        )

        self.fitted_feature_pipelines = (
            [None] * self.num_features if self.num_features > 0 else []
        )
        self.fitted_target_pipelines = (
            [None] * (2 * self.num_targets) if self.num_targets > 0 else []
        )
        if log_transform_targets:
            self.log_transform = log_transform
            self.inverse_log_transform = inverse_log_transform
        else:
            self.log_transform = no_log
            self.inverse_log_transform = no_inverse_log

    def set_feature_pipeline(self, pipeline: Pipeline) -> None:
        """Sets the same pipeline (deep-copied) for each feature column."""
        if self.num_features > 0:
            self.feature_pipelines = [
                copy.deepcopy(pipeline) for _ in range(self.num_features)
            ]

    def set_feature_pipelines(self, pipeline_list: List[Pipeline]) -> None:
        """Sets individual pipelines for each feature."""
        if self.num_features == 0:
            return
        if len(pipeline_list) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} feature pipelines, got {len(pipeline_list)}"
            )
        self.feature_pipelines = [copy.deepcopy(p) for p in pipeline_list]

    def set_target_pipeline(self, pipeline: Pipeline) -> None:
        """Sets a single pipeline for all target columns, including log versions (using deep copies)."""
        if self.num_targets > 0:
            # For the non-log targets, use a deep copy for each.
            normal_pipelines = [
                copy.deepcopy(pipeline) for _ in range(self.num_targets)
            ]
            # For the log targets, build a pipeline that includes the log transform, then a copy of pipeline.
            log_pipelines = [
                Pipeline(
                    [
                        (
                            "log_transform",
                            FunctionTransformer(
                                self.log_transform,
                                inverse_func=self.inverse_log_transform,
                            ),
                        ),
                        ("scaler", copy.deepcopy(pipeline)),
                    ]
                )
                for _ in range(self.num_targets)
            ]
            self.target_pipelines = normal_pipelines + log_pipelines

    def set_target_pipelines(self, pipeline_list: List[Pipeline]) -> None:
        """Sets individual pipelines for each target & log target."""
        if self.num_targets == 0:
            return
        if len(pipeline_list) != 2 * self.num_targets:
            raise ValueError(
                f"Expected {2 * self.num_targets} target pipelines, got {len(pipeline_list)}"
            )
        self.target_pipelines = [copy.deepcopy(p) for p in pipeline_list]

    def fit(self, df: pd.DataFrame, log_indexes: Optional[List[int]] = None) -> None:
        """
        Fits feature and target pipelines.
        If log_indexes is None, assumes the last num_targets columns are the log-transformed targets.
        """
        if self.num_features > 0:
            for i, idx in enumerate(self.feature_indexes):
                self.fitted_feature_pipelines[i] = self.feature_pipelines[i].fit(
                    df.iloc[:, idx].values.reshape(-1, 1)
                )
        if self.num_targets > 0:
            if log_indexes is None:
                log_indexes = list(
                    range(len(df.columns) - self.num_targets, len(df.columns))
                )
            for i, idx in enumerate(self.target_indexes):
                self.fitted_target_pipelines[i] = self.target_pipelines[i].fit(
                    df.iloc[:, idx].values.reshape(-1, 1)
                )
                self.fitted_target_pipelines[i + self.num_targets] = (
                    self.target_pipelines[i + self.num_targets].fit(
                        df.iloc[:, log_indexes[i]].values.reshape(-1, 1)
                    )
                )

    def transform(
        self, df: pd.DataFrame, log_indexes: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Transforms features & targets in place, keeping the original DataFrame structure.
        Only the specified feature and target columns are updated; all other columns remain unchanged.

        Args:
        df (pd.DataFrame): The dataset.
        log_indexes (list or None): If `None`, assumes the last n columns are the log-transformed targets.

        Returns:
        pd.DataFrame: The transformed DataFrame with original column order.
        """
        # Make a copy of the DataFrame; do not cast entire df to float so that non-numeric columns remain unchanged.
        transformed_df = df.copy()

        if self.num_targets > 0 and log_indexes is None:
            log_indexes = list(
                range(len(df.columns) - self.num_targets, len(df.columns))
            )

        # Transform only the feature columns
        for i, idx in enumerate(self.feature_indexes):
            col_values = df.iloc[:, idx].values.reshape(-1, 1)
            transformed_features = (
                self.fitted_feature_pipelines[i].transform(col_values).flatten()
            )
            # Use pd.Series to force the column to float
            transformed_df.iloc[:, idx] = pd.Series(
                transformed_features.astype(np.float64), index=transformed_df.index
            )

        # Transform only the target columns (and their log-transformed duplicates)
        for i, idx in enumerate(self.target_indexes):
            # Regular target
            col_values = df.iloc[:, idx].values.reshape(-1, 1)
            transformed_values = (
                self.fitted_target_pipelines[i].transform(col_values).flatten()
            )
            transformed_df.iloc[:, idx] = pd.Series(
                transformed_values.astype(np.float64), index=transformed_df.index
            )

            log_col_index = log_indexes[i]
            col_values_log = df.iloc[:, log_col_index].values.reshape(-1, 1)
            transformed_values_log = (
                self.fitted_target_pipelines[i + self.num_targets]
                .transform(col_values_log)
                .flatten()
            )

            transformed_df.iloc[:, log_col_index] = pd.Series(
                transformed_values_log.astype(np.float64), index=transformed_df.index
            )

        return transformed_df

    def get_transformers(self) -> Tuple[List[Pipeline], List[Pipeline]]:
        """
        Returns the stored transformers (for external reference).
        """
        return self.fitted_feature_pipelines, self.fitted_target_pipelines
