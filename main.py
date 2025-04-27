import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from fuzzy_imputation import FuzzyImputer
from fuzzy_selection import FuzzySelector


def main():
    logger.info("Parsing args")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="Path to input csv file")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to output csv file")
    parser.add_argument("--impute", action="store_true", help="Perform imputation")
    parser.add_argument("--select-features", action="store_true", help="Perform feature selection")
    args = parser.parse_args()

    logger.info("Loading data")
    df = pd.read_csv(args.input_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    if args.impute:
        logger.info("Imputing")
        X = FuzzyImputer().fit_transform(X)
    if args.select_features:
        logger.info("Selecting features")
        X = FuzzySelector().fit_transform(X, y)

    logger.info("Saving data")
    df = pd.concat([X, y], axis=1)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
