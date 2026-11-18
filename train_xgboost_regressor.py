#!/usr/bin/env python3
"""Train and cross-validate an XGBoost regression model from a CSV file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


TARGET = "P_diff_target0_est_pred"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-validate, train, and save an XGBoost regression model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("long_df_head.csv"),
        help="Input CSV path (default: long_df_head.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/xgb_regressor.joblib"),
        help="Path for the trained model file.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of shuffled K-fold cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used for cross-validation and model training.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Optional feature columns to exclude, such as IDs or leakage-prone fields.",
    )
    parser.add_argument(
        "--dropbox-path",
        help=(
            "Optional destination path in Dropbox (for example "
            "/models/xgb_regressor.joblib). Requires DROPBOX_ACCESS_TOKEN."
        ),
    )
    return parser.parse_args()


def make_model(numeric_columns: list[str], categorical_columns: list[str], seed: int) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(transformers=transformers)
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", regressor),
        ]
    )


def upload_to_dropbox(local_path: Path, dropbox_path: str) -> None:
    token = os.environ.get("DROPBOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "Set the DROPBOX_ACCESS_TOKEN environment variable to upload to Dropbox."
        )
    if not dropbox_path.startswith("/"):
        raise ValueError("--dropbox-path must be an absolute Dropbox path beginning with '/'.")

    import dropbox

    with local_path.open("rb") as model_file:
        dropbox.Dropbox(token).files_upload(
            model_file.read(),
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite,
        )
    print(f"Uploaded model to Dropbox: {dropbox_path}")


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    data = pd.read_csv(args.data)
    if TARGET not in data.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing from {args.data}.")

    unknown_drop_columns = sorted(set(args.drop_columns) - set(data.columns))
    if unknown_drop_columns:
        raise ValueError(f"Columns requested for exclusion were not found: {unknown_drop_columns}")
    if TARGET in args.drop_columns:
        raise ValueError(f"The target column {TARGET!r} cannot be excluded.")

    target = pd.to_numeric(data[TARGET], errors="coerce")
    valid_target = target.notna() & np.isfinite(target)
    if not valid_target.all():
        print(
            f"Excluding {(~valid_target).sum()} rows with missing, non-numeric, "
            "or non-finite targets."
        )

    feature_columns = [
        column
        for column in data.columns
        if column != TARGET and column not in args.drop_columns
    ]
    features = data.loc[valid_target, feature_columns].copy()
    target = target.loc[valid_target].astype(float)
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_columns:
        features[numeric_columns] = features[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    features = features.dropna(axis=1, how="all")
    if features.shape[1] == 0:
        raise ValueError("No usable feature columns remain after exclusions.")
    if len(features) < args.folds:
        raise ValueError(
            f"Need at least {args.folds} rows with numeric targets; found {len(features)}."
        )

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    fold_results = []
    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(features), start=1
    ):
        model = make_model(numeric_columns, categorical_columns, args.random_state)
        X_train = features.iloc[train_indices]
        X_validation = features.iloc[validation_indices]
        y_train = target.iloc[train_indices]
        y_validation = target.iloc[validation_indices]

        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        fold_results.append(
            {
                "fold": fold_number,
                "rmse": np.sqrt(mean_squared_error(y_validation, predictions)),
                "mae": mean_absolute_error(y_validation, predictions),
                "r2": r2_score(y_validation, predictions),
            }
        )

    metrics = pd.DataFrame(fold_results)
    metrics_path = args.output.with_name(f"{args.output.stem}.cv_metrics.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    print("Cross-validation metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nMean and standard deviation across folds:")
    print(metrics[["rmse", "mae", "r2"]].agg(["mean", "std"]).to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved fold metrics to: {metrics_path}")

    final_model = make_model(numeric_columns, categorical_columns, args.random_state)
    final_model.fit(features, target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.output)
    print(f"Saved trained model to: {args.output}")

    if args.dropbox_path:
        upload_to_dropbox(args.output, args.dropbox_path)


if __name__ == "__main__":
    main()
