import os
import json
import gzip
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix


def load_data(path):
    df = pd.read_csv(path, compression="zip")
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns="ID")
    df = df[(df["MARRIAGE"] != 0) & (df["EDUCATION"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df.dropna()


def calc_metrics(name, y_real, y_pred):
    return {
        "type": "metrics",
        "dataset": name,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }


def compute_confusion_matrix(name, y_real, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def main():
    train = load_data("files/input/train_data.csv.zip")
    test = load_data("files/input/test_data.csv.zip")

    X_train = train.drop(columns="default")
    y_train = train["default"]
    X_test = test.drop(columns="default")
    y_test = test["default"]

    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    model = Pipeline([
        ("pre", pre),
        ("sel", SelectKBest(score_func=f_classif, k=20)),
        ("pca", PCA(n_components=None)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(50, 30, 40, 60),
            alpha=0.26,
            learning_rate_init=0.001,
            max_iter=15000,
            random_state=21
        ))
    ])

    grid = GridSearchCV(model, param_grid={}, cv=10, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    model_dir = Path("files/models")
    if model_dir.exists():
        for f in glob(str(model_dir / "*")):
            os.remove(f)
    model_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(model_dir / "model.pkl.gz", "wb") as f:
        pickle.dump(grid, f)

    y_pred_train = grid.predict(X_train)
    y_pred_test = grid.predict(X_test)

    resultados = [
        calc_metrics("train", y_train, y_pred_train),
        calc_metrics("test", y_test, y_pred_test),
        compute_confusion_matrix("train", y_train, y_pred_train),
        compute_confusion_matrix("test", y_test, y_pred_test),
    ]

    out_dir = Path("files/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r) + "\n")



if __name__ == "__main__":
    main()
