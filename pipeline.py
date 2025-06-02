import json
import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --------------------------------------------------
# Data loading utilities
# --------------------------------------------------

try:
    # Preferred: use the group’s shared pre‑processor so train/test
    # splits & encodings are identical across every script.
    from data_preprocessor import load_processed  # type: ignore
except ImportError:
    # Fallback loader.
    def load_processed(split: str = "train") -> Tuple[pd.DataFrame, pd.Series]:
        """Minimal loader in case data_preprocessor is missing.

        Parameters
        ----------
        split : "train" | "holdout"
            Portion of the Adult dataset to return.

        Returns
        -------
        X : DataFrame
            Feature matrix (categorical values are left as strings).
        y : Series
            Binary labels (1 if income > $50K).
        """
        cols = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        df = pd.read_csv(
            "data/adult.data", header=None, names=cols, na_values="?", skipinitialspace=True
        )
        df.dropna(inplace=True)
        df["income"] = (df["income"].str.contains(">50K")).astype(int)

        # Simple encode categoricals numerically so scikit models accept them.
        #cat_cols = df.select_dtypes(include="object").columns.drop("income")
        cat_cols = df.select_dtypes(include="object").columns.difference(["income"])
        for col in cat_cols:
            df[col] = df[col].astype("category").cat.codes

        from sklearn.model_selection import train_test_split

        train_df, holdout_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["income"]
        )
        part = train_df if split == "train" else holdout_df
        return part.drop("income", axis=1), part["income"]


# --------------------------------------------------
# Metric helpers
# --------------------------------------------------

def prop_score_auc(X_all: pd.DataFrame, labels: np.ndarray) -> float:
    """Propensity score AUC.

    Trains a logistic classifier to separate real vs synthetic rows. AUC
    closer to 0.5 means synthetic distribution is hard to distinguish
    (better fidelity / privacy).
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_all, labels)
    probs = lr.predict_proba(X_all)[:, 1]
    return roc_auc_score(labels, probs)


def membership_inference_auc(X_real_train: pd.DataFrame, X_synth: pd.DataFrame) -> float:
    """Distance‑based membership‑inference AUC.

    Uses nearest‑neighbour distance to synthetic data as the attack
    signal. Lower AUC (~0.5) implies less leak.
    """
    # Distances for members (train) and non‑members (synthetic itself)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X_synth)
    dist_train, _ = nbrs.kneighbors(X_real_train)

    # Synthetic to itself gives distance 0 (non‑members baseline)
    dist_synth = np.zeros(len(X_synth))
    attack_features = np.concatenate([dist_train.flatten(), dist_synth])
    attack_labels = np.concatenate([
        np.ones(len(X_real_train)),  # members
        np.zeros(len(X_synth)),      # non‑members
    ])
    # Higher distance => less likely member; negate so larger means member
    return roc_auc_score(attack_labels, -attack_features)


# --------------------------------------------------
# Synthetic loader
# --------------------------------------------------

def load_synthetic(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    return df.drop("income", axis=1), df["income"]


# --------------------------------------------------
# Main evaluation routine
# --------------------------------------------------

def main(epsilon: float, clf_name: str = "random_forest") -> None:
    # Real data
    X_train, y_train = load_processed("train")
    X_test, y_test = load_processed("holdout")

    # Classifier map
    clf_map = {
        "random_forest": RandomForestClassifier(),
        "gb": GradientBoostingClassifier(),
        "logreg": LogisticRegression(max_iter=500),
    }
    if clf_name not in clf_map:
        raise ValueError(f"Unsupported classifier: {clf_name}")

    # Baseline: train on real, test on real
    base_clf = clf_map[clf_name]
    base_clf.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, base_clf.predict(X_test))

    # Paths to synthetic CSVs produced by each pipeline
    synth_paths = {
        "modelfirst": f"synthetic/modelfirst_ε{epsilon}.csv",
        "statfirst": f"synthetic/statfirst_ε{epsilon}.csv",
    }

    results = {"epsilon": epsilon, "baseline_real_acc": baseline_acc}

    for tag, path in synth_paths.items():
        if not os.path.exists(path):
            print(f"[!] Skipping {tag}: {path} not found.")
            continue
        X_syn, y_syn = load_synthetic(path)

        clf = clf_map[clf_name]
        clf.fit(X_syn, y_syn)
        tstr_acc = accuracy_score(y_test, clf.predict(X_test))

        combined = pd.concat([X_train, X_syn], ignore_index=True)
        labels = np.concatenate([
            np.zeros(len(X_train)),
            np.ones(len(X_syn)),
        ])
        ps_auc = prop_score_auc(combined, labels)
        mi_auc = membership_inference_auc(X_train, X_syn)

        results[tag] = {
            "tstr_acc": tstr_acc,
            "propensity_auc": ps_auc,
            "membership_auc": mi_auc,
        }

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    os.makedirs("metrics", exist_ok=True)
    out_path = f"metrics/summary_ε{epsilon}_{clf_name}.json"
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"[✓] Saved evaluation results → {out_path}\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DP synthetic datasets using TSTR, fidelity & privacy metrics."
    )
    parser.add_argument("--epsilon", type=float, required=True, help="Privacy budget ε in filename")
    parser.add_argument(
        "--clf",
        choices=["random_forest", "gb", "logreg"],
        default="random_forest",
        help="Classifier for TSTR evaluation",
    )
    args = parser.parse_args()
    main(args.epsilon, args.clf)


##python evaluate_week4.py --epsilon 1.0 --clf random_forest