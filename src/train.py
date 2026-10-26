"""Train and evaluate Decision Tree + Random Forest deauth-flood detectors.

Key honesty point: the headline test is split BY CAPTURE SESSION, so the test set
contains whole captures the model never saw. A naive random split of windows is
also reported to show how much it inflates the score (neighbouring windows from
the same attack are nearly identical).
"""
import argparse
import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from features import FEATURES

CLASSES = ["normal", "deauth flood"]


def split_by_session(df, test_frac=0.3, seed=42):
    rng = np.random.RandomState(seed)
    sess = df.groupby("session")["label"].max()
    test = []
    for cls in (0, 1):
        names = sorted(sess[sess == cls].index)
        rng.shuffle(names)
        test += names[: max(1, round(len(names) * test_frac))]
    mask = df["session"].isin(test)
    return df[~mask], df[mask], sorted(test)


def scores(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/features.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="model.joblib")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data)
    train, test, test_sessions = split_by_session(df)
    if train["label"].nunique() < 2 or test["label"].sum() == 0:
        raise SystemExit("Not enough attack sessions to split into train/test. "
                         "Add more attack captures/CSV chunks (need at least 2 sessions containing attacks).")
    Xtr, ytr = train[FEATURES], train["label"]
    Xte, yte = test[FEATURES], test["label"]
    print(f"train: {len(train)} windows ({int(ytr.sum())} attack) | "
          f"test: {len(test)} windows ({int(yte.sum())} attack)")
    print(f"held-out test sessions: {test_sessions}\n")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        fitted[name] = m
        results[name] = scores(yte, m.predict(Xte))
        r = results[name]
        print(f"{name}: acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
              f"rec={r['recall']:.4f} f1={r['f1']:.4f} cm={r['confusion_matrix']}")

    # --- stricter / sanity checks ---------------------------------------
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(df[FEATURES], df["label"], test_size=0.3,
                                                  stratify=df["label"], random_state=42)
    naive = scores(yr_te, rf.fit(Xr_tr, yr_tr).predict(Xr_te))
    n_splits = min(5, df["session"].nunique())
    cv_acc, cv_rec = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(df[FEATURES], df["label"], df["session"]):
        mdl = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        mdl.fit(df.iloc[tr][FEATURES], df.iloc[tr]["label"])
        p = mdl.predict(df.iloc[te][FEATURES])
        cv_acc.append(accuracy_score(df.iloc[te]["label"], p))
        if df.iloc[te]["label"].sum() > 0:  # recall is undefined if a fold has no attacks
            cv_rec.append(recall_score(df.iloc[te]["label"], p, zero_division=0))
    print(f"\nnaive random-window split (inflated): acc={naive['accuracy']:.4f} rec={naive['recall']:.4f}")
    print(f"{n_splits}-fold session-grouped CV (RF): acc={np.mean(cv_acc):.4f}+/-{np.std(cv_acc):.4f} "
          f"recall={np.mean(cv_rec):.4f}+/-{np.std(cv_rec):.4f}")
    # trivial baseline: always predict "normal"
    base = accuracy_score(yte, np.zeros(len(yte), dtype=int))
    print(f"baseline 'always normal' accuracy on test: {base:.4f}  <- why accuracy alone is misleading")

    # --- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (name, m) in zip(axes, fitted.items()):
        ConfusionMatrixDisplay.from_predictions(yte, m.predict(Xte), display_labels=CLASSES,
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"{name}\n(held-out sessions)")
    plt.tight_layout()
    plt.savefig(f"{a.out}/confusion_matrix.png", dpi=200)
    plt.close()

    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 4))
    imp.plot.barh(color="#2b6cb0")
    plt.xlabel("Importance (Random Forest)")
    plt.title("Which features matter?")
    plt.tight_layout()
    plt.savefig(f"{a.out}/feature_importance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 6))
    plot_tree(fitted["Decision Tree"], feature_names=FEATURES, class_names=CLASSES,
              filled=True, rounded=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{a.out}/decision_tree.png", dpi=200)
    plt.close()
    with open(f"{a.out}/decision_tree_rules.txt", "w") as f:
        f.write(export_text(fitted["Decision Tree"], feature_names=FEATURES))

    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump({"test_sessions": test_sessions, "models": results,
                   "naive_random_split_rf": naive,
                   "grouped_cv_rf": {"folds": n_splits, "acc_mean": float(np.mean(cv_acc)),
                                     "acc_std": float(np.std(cv_acc)),
                                     "recall_mean": float(np.mean(cv_rec)),
                                     "recall_std": float(np.std(cv_rec))},
                   "always_normal_baseline_acc": base,
                   "feature_importance": imp.sort_values(ascending=False).to_dict()}, f, indent=2)

    joblib.dump(fitted["Random Forest"], a.model)
    print(f"\nSaved {a.model} (Random Forest) and figures/metrics in {a.out}/")
    print("\nFeature importance (RF):\n" + imp.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
