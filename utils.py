from __future__ import annotations

import numpy as np
import pandas as pd
from skfuzzy import cmeans
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class TreeBasedPipeline:

    def __init__(self, transforms: Pipeline, model: DecisionTreeClassifier | DecisionTreeRegressor):
        self.transforms = transforms
        self.model = model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TreeBasedPipeline:
        self.transforms.fit(X, y)
        X_transformed = self.transforms.transform(X)
        self.model.fit(X_transformed, y)

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.transforms.transform(X)
        return self.model.apply(X_transformed)


def get_generic_pipeline() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one-hot",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    ).set_output(transform="pandas")

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    ).set_output(transform="pandas")

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(dtype_include=("object", "category")),
                    ),
                    (
                        num_pipeline,
                        make_column_selector(dtype_include=np.number),
                    ),
                ),
            )
        ]
    ).set_output(transform="pandas")
    return pipeline


def fuzzy_silhouette(data, partition_matrix, prototypes, distance="euclidean", alpha=1.0):
    N = data.shape[0]
    c = partition_matrix.shape[0]

    sorted_memberships = np.sort(partition_matrix, axis=0)[::-1]
    sorted_indices = np.argsort(partition_matrix, axis=0)[::-1]

    mu_p = sorted_memberships[0]
    mu_q = sorted_memberships[1]

    best_cluster_indices = sorted_indices[0]

    silhouette_values = np.zeros(N)
    weights = np.zeros(N)

    for j in range(N):

        p = best_cluster_indices[j]

        if c < 2 or np.sum(partition_matrix[p] > 0.5) <= 1:
            silhouette_values[j] = 0
            weights[j] = 0
            continue

        if distance == "euclidean":
            a_pj = np.sqrt(np.sum((data[j] - prototypes[p]) ** 2))
        else:
            raise ValueError(f"Distance metric '{distance}' not supported")

        distances_to_other_prototypes = []
        for i in range(c):
            if i != p:
                if distance == "euclidean":
                    dist = np.sqrt(np.sum((data[j] - prototypes[i]) ** 2))
                    distances_to_other_prototypes.append(dist)

        b_pj = min(distances_to_other_prototypes)

        if a_pj == 0 and b_pj == 0:
            silhouette_values[j] = 0
        else:
            silhouette_values[j] = (b_pj - a_pj) / max(a_pj, b_pj)

        weights[j] = (mu_p[j] - mu_q[j]) ** alpha

    if np.sum(weights) == 0:
        return 0
    else:
        return np.sum(weights * silhouette_values) / np.sum(weights)


def get_optimal_clustering(X: np.ndarray):
    silhouettes = []
    for n_clusters in range(min(2, X.shape[1]), min(8, X.shape[1])):
        centers, partitioning, _, _, _, _, _ = cmeans(
            X.T, n_clusters, m=1.5, error=0.001, maxiter=500
        )
        silhouettes.append(fuzzy_silhouette(X, partitioning, centers))

    if len(silhouettes) == 0:
        best_number_of_clusters = 1
    else:
        best_number_of_clusters = np.argmax(silhouettes) + 2
    return cmeans(X, best_number_of_clusters, m=1.5, error=0.001, maxiter=500)
