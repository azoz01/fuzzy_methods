from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from skfuzzy import cmeans_predict
from sklearn.base import TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import tqdm

from utils import (
    TreeBasedPipeline,
    get_generic_pipeline,
    get_optimal_clustering,
)


class FuzzyImputer(TransformerMixin):
    """
    Source: https://link.springer.com/article/10.1007/s10115-019-01427-1
    """

    def __init__(self):
        self.estimators = dict()

    def fit(self, X: pd.DataFrame, y: Any = None) -> FuzzyImputer:
        # Resolve column types
        self.categorical_columns = X.select_dtypes(include=["object"]).columns.to_list()
        self.numerical_columns = X.select_dtypes(exclude=["object"]).columns.to_list()
        # Extract only complete data rows
        self.fitting_data = X.loc[X.index[~(X.isna().any(axis=1))]].reset_index(drop=True)

        # Fit predictors on complete data
        for column in self.categorical_columns:
            clf = TreeBasedPipeline(get_generic_pipeline(), DecisionTreeClassifier(max_depth=5))
            clf.fit(
                self.fitting_data.drop(columns=[column]),
                self.fitting_data[column],
            )
            self.estimators[column] = clf
        for column in self.numerical_columns:
            clf = TreeBasedPipeline(get_generic_pipeline(), DecisionTreeRegressor(max_depth=5))
            clf.fit(
                self.fitting_data.drop(columns=[column]),
                self.fitting_data[column],
            )
            self.estimators[column] = clf

        # Calculate nodes of complete data inside imputers
        self.training_data_leaf_nodes = dict()
        for column in self.categorical_columns + self.numerical_columns:
            leaf_nodes = self.estimators[column].apply(self.fitting_data.drop(columns=column))
            self.training_data_leaf_nodes[column] = dict()
            for node in np.unique(leaf_nodes):
                self.training_data_leaf_nodes[column][node] = np.where(leaf_nodes == node)[
                    0
                ].tolist()

        return self

    def transform(
        self, X: pd.DataFrame, learning_rate: float = 0.1, max_iterations: int = 3
    ) -> pd.DataFrame:
        X = deepcopy(X)
        # Extract data with incomplete rows
        original_idx = X.index
        complete_df = X.loc[X.index[~(X.isna().any(axis=1))]]
        incomplete_df = (
            X.loc[X.index[X.isna().any(axis=1)]]
            .reset_index(drop=False)
            .rename(columns={"index": "original_index"})
        )
        # If no missing data, then nothing to do
        if incomplete_df.shape[0] == 0:
            return X
        original_incomplete_df = deepcopy(incomplete_df)

        missing_data_to_nodes_mapping = defaultdict(
            dict
        )  # Mapping (row, column) -> node_of_imputer
        nodes_to_missing_data_mapping = defaultdict(
            lambda: defaultdict(list)
        )  # Mapping (column, node) -> list of rows from incomplete data
        # First imputation step
        for row_idx in incomplete_df.index:
            row = incomplete_df.loc[[row_idx]]
            for column in self.categorical_columns + self.numerical_columns:
                value = row.loc[row_idx, column]
                if not pd.isnull(value) and not pd.isna(value):
                    continue
                # Get the imputer for the current column
                imputer = self.estimators[column]
                # Get the leaf node of the incomplete data
                leaf_node = imputer.apply(row.drop(columns=column))
                # Save information about the incomplete data
                missing_data_to_nodes_mapping[row_idx][column] = leaf_node[0].item()
                nodes_to_missing_data_mapping[column][leaf_node[0].item()].append(row_idx)
                # Get the indices of the training data that belong to the same leaf node
                training_data_idx = self.training_data_leaf_nodes[column][leaf_node[0].item()]
                # Get the most frequent value in the same leaf node
                if column in self.categorical_columns:
                    # For categorical columns, use mode
                    incomplete_df.loc[row_idx, column] = self.fitting_data.loc[
                        training_data_idx, column
                    ].mode()[0]
                else:
                    # For numerical columns, use mean
                    incomplete_df.loc[row_idx, column] = self.fitting_data.loc[
                        training_data_idx, column
                    ].mean(skipna=True)

        # Second imputation step
        av = float("inf")
        iteration_counter = 0
        while av >= 0.005 and iteration_counter < max_iterations:
            clustering_cache = defaultdict(dict)
            prev_iteration_df = deepcopy(incomplete_df)
            pbar = tqdm(incomplete_df.index, leave=False)
            pbar.set_postfix_str(f"{av=:.4f}, {iteration_counter=}")
            for row_idx in pbar:
                row = incomplete_df.loc[[row_idx]].drop(columns=["original_index"])
                for column in self.categorical_columns + self.numerical_columns:
                    original_value = original_incomplete_df.loc[row_idx, column]
                    # If this value is not missing in original data then proceed
                    if not pd.isnull(original_value) and not pd.isna(original_value):
                        continue
                    leaf_node = self.estimators[column].apply(row.drop(columns=column))[0].item()
                    # If row is not already clustered then cluster it
                    if not (column in clustering_cache and row_idx in clustering_cache[column]):
                        clustering_data = np.concat(
                            [
                                self.fitting_data.loc[
                                    self.training_data_leaf_nodes[column][leaf_node]
                                ]
                                .select_dtypes(np.number)
                                .values,
                                prev_iteration_df.drop(columns=["original_index"])
                                .loc[nodes_to_missing_data_mapping[column][leaf_node]]
                                .select_dtypes(np.number)
                                .values,
                            ],
                            axis=0,
                        )
                        cluster_centers, clusters, _, _, _, _, _ = get_optimal_clustering(
                            clustering_data.T,
                        )
                        if column in self.categorical_columns:
                            highest_degree_clusters = clusters.argmax(axis=0)
                            most_frequent_values = {
                                cluster: Counter(
                                    pd.concat(
                                        [
                                            self.fitting_data.loc[
                                                self.training_data_leaf_nodes[column][leaf_node],
                                                column,
                                            ],
                                            prev_iteration_df.loc[
                                                nodes_to_missing_data_mapping[column][leaf_node],
                                                column,
                                            ],
                                        ]
                                    )
                                ).most_common(1)[0][0]
                                for cluster in np.unique(highest_degree_clusters)
                            }
                            target_column_idx = None
                        else:
                            most_frequent_values = None
                            target_column_idx = self.fitting_data.select_dtypes(
                                np.number
                            ).columns.get_loc(column)
                        clustering_cache[column][row_idx] = (
                            cluster_centers,
                            clusters,
                            target_column_idx,
                            most_frequent_values,
                        )

                    cluster_centers, clusters, target_column_idx, most_frequent_values = (
                        clustering_cache[column][row_idx]
                    )
                    row_cluster, _, _, _, _, _ = cmeans_predict(
                        row.select_dtypes(np.number).values.reshape(-1, 1),
                        cluster_centers,
                        m=1.5,
                        error=0.001,
                        maxiter=500,
                    )
                    if column in self.numerical_columns:
                        incomplete_df.loc[row_idx, [column]] = incomplete_df.loc[
                            row_idx, [column]
                        ] + learning_rate * (
                            (cluster_centers[:, target_column_idx] * row_cluster).sum()
                            - incomplete_df.loc[row_idx, [column]]
                        )
                    else:
                        membership_degrees_of_cat_column = defaultdict(lambda: 0)
                        for cluster in range(clusters.shape[0]):
                            membership_degrees_of_cat_column[most_frequent_values[cluster]] = (
                                membership_degrees_of_cat_column[most_frequent_values[cluster]]
                                + row_cluster[cluster][0]
                            )
                        incomplete_df.loc[row_idx, [column]] = max(
                            membership_degrees_of_cat_column.items(), key=lambda item: item[1]
                        )[0]
            av = (
                (
                    incomplete_df.select_dtypes(include=np.number)
                    - prev_iteration_df.select_dtypes(include=np.number)
                )
                .abs()
                .mean()
                .mean()
            )
            iteration_counter += 1

        # Final postprocessing
        incomplete_df.index = incomplete_df["original_index"]
        incomplete_df = incomplete_df.drop(columns=["original_index"])
        return pd.concat([incomplete_df, complete_df]).loc[original_idx]
