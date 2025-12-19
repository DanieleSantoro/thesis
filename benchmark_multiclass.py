import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Classifiers """
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

""" Dataset and Metrics """
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.utils import resample
from netcal.metrics import ECE

""" Confidence Ensembles """
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting as ConfidenceBoostingClassifier
from confens.classifiers.ConfidenceBagging import ConfidenceBagging as ConfidenceBaggingClassifier

def load_fashion_mnist_dataset():
    print("Loading Fashion-MNIST...")
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    X, y = resample(X, y, n_samples=10000, random_state=42, stratify=y)
    X = X / 255.0 # Normalizzazione
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def evaluate_multiclass(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # 1. Accuracy
    acc = accuracy_score(y_test, y_pred)

    # 2. ECE
    ece = ECE(bins=10).measure(y_prob, y_test)

    return {"Model": name, "Accuracy": acc, "ECE": ece}, y_prob

def plot_multiclass_reliability_aggregated(y_test, y_prob, name, ax):
    """
    Computes the Class-Agnostic Reliability Diagram.
    It bins predictions based on the confidence of the argmax class and
    calculates the expected accuracy within each bin. This aggregates
    calibration performance across all classes into a single curve.
    """

    # 1. Taking the maximum probability for each sample
    confidences = np.max(y_prob, axis=1)

    # 2. Check if the predicted class matches the true label.
    # Doing this, I converted the multiclass problem into a binary task (Correct vs Incorrect)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_test).astype(int)

    # 3. Generating the reliability curve points
    prob_true, prob_pred = calibration_curve(accuracies, confidences, n_bins=10, strategy='uniform')

    ax.plot(prob_pred, prob_true, "s-", label=f"{name}")

if __name__ == "__main__":
    # 1. Loading data
    X_train, X_test, y_train, y_test = load_fashion_mnist_dataset()
    results = []

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

    competitors = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Neural Network (MLP)", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=100, random_state=42)),
    ]

    base_dt = DecisionTreeClassifier(max_depth=8)
    conf_ensembles = [
        ("ConfBag (Native)", ConfidenceBaggingClassifier(clf=base_dt, n_base=20, conf_thr=0.75, weighted=True)),
        ("ConfBoost (Native)",
         ConfidenceBoostingClassifier(clf=base_dt, n_base=20, conf_thr=0.75, weighted=True, learning_rate=1.0,
                                      sampling_ratio=0.8))
    ]

    # LOOP 1: Classics (Native + Isotonic + Platt Scaling)
    for name, clf in competitors:
        print(f"\nProcessing {name}...")

        # A. Native
        clf.fit(X_train, y_train)
        res, probs = evaluate_multiclass(name, clf, X_test, y_test)
        results.append(res)

        plot_multiclass_reliability_aggregated(y_test, probs, f"{name} (Native)", ax)

        # B. Isotonic (Post-Hoc)
        iso_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        iso_clf.fit(X_train, y_train)
        res_iso, _ = evaluate_multiclass(f"{name} + Isotonic", iso_clf, X_test, y_test)
        results.append(res_iso)

        # C. Platt Scaling
        platt_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
        platt_clf.fit(X_train, y_train)
        res_platt, _ = evaluate_multiclass(f"{name} + Platt", platt_clf, X_test, y_test)
        results.append(res_platt)

    # LOOP 2: Confidence Ensembles
    for name, clf in conf_ensembles:
        print(f"\nProcessing {name}...")
        clf.fit(X_train, y_train)
        res, probs = evaluate_multiclass(name, clf, X_test, y_test)
        results.append(res)

        plot_multiclass_reliability_aggregated(y_test, probs, f"{name}", ax)

    # FINAL OUTPUT
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="ECE")

    print("\n" + "=" * 50)
    print(" MULTICLASS LEADERBOARD (SORTED BY ECE)")
    print("=" * 50)
    print(df_results.to_string(index=False))

    df_results.to_csv("benchmark_multiclass_extended.csv", index=False)

    ax.set_ylabel("Accuracy (Fraction of Correct Predictions)")
    ax.set_xlabel("Confidence (Predicted Probability)")
    ax.set_title("Multiclass Reliability Diagram (Aggregated)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.savefig("reliability_multiclass.png")
    print("\n Saving as 'reliability_multiclass.png'")