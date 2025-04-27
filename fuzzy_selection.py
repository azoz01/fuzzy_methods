from functools import partial
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


class FuzzySelector(TransformerMixin):
    """
    Source: https://sci2s.ugr.es/keel/pdf/specific/articulo/ramze_rezaee_goedhart_99_fuzzy%20feature%20selection.pdf
    """  # noqa: E501

    def __init__(self, n_features_to_select: int | Literal["auto"] = "auto"):
        self.fuzzifier = FeatureFuzzifier()
        # TODO: if required, then change cv to bagging
        self.selector = SequentialFeatureSelector(
            KNeighborsClassifier(),
            n_features_to_select=n_features_to_select,
            scoring=make_scorer(accuracy_score),
            direction="backward",
            cv=5,
        ).set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        X_fuzzified = self.fuzzifier.fit_transform(X)
        self.selector.fit(X_fuzzified, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_fuzzified = self.fuzzifier.transform(X)
        return self.selector.transform(X_fuzzified)


class FeatureFuzzifier(TransformerMixin):
    """
    A transformer that applies fuzzy matching to a specified column in a DataFrame.
    """

    def fit(self, X: Any, y: Any = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numerical_columns = X.select_dtypes(include=np.number)
        categorical_columns = X.select_dtypes(exclude=np.number)
        numerical_fuzzified = pd.concat(
            [self.__fuzzify_feature(X[feature]) for feature in numerical_columns.columns], axis=1
        )
        categorical_fuzzified = (
            OneHotEncoder(sparse_output=False)
            .set_output(transform="pandas")
            .fit_transform(categorical_columns)
        )
        return pd.concat([numerical_fuzzified, categorical_fuzzified], axis=1)

    def __fuzzify_feature(self, feature: pd.Series) -> pd.DataFrame:
        feature_min = feature.min()
        feature_max = feature.max()
        feature = (feature - feature_min) / (feature_max - feature_min)
        output = pd.DataFrame(
            {
                f"{feature.name}_set_1": feature.apply(partial(min_set, 0.0, 0.5)),
                f"{feature.name}_set_2": feature.apply(partial(middle_set, 0.0, 0.5, 1.0)),
                f"{feature.name}_set_3": feature.apply(partial(max_set, 0.5, 1.0)),
            }
        )
        return output


def min_set(feature_min: float, feature_middle: float, feature: float) -> float:
    if feature <= feature_min:
        return 1
    if feature > feature_middle:
        return 0
    return 1 - feature / feature_middle


def max_set(feature_middle: float, feature_max: float, feature: float) -> float:
    if feature <= feature_middle:
        return 0
    if feature > feature_max:
        return 1
    return feature / feature_middle - 1


def middle_set(
    feature_min: float, feature_middle: float, feature_max: float, feature: float
) -> float:
    if feature > feature_max or feature < feature_min:
        return 0
    if feature <= feature_middle:
        return feature / feature_middle
    return -feature / feature_middle + 2
