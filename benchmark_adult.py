import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" --- sklearn classifiers ---"""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

""" --- XGBoost ---"""
from xgboost import XGBClassifier

""" --- calibration & metrics ---"""
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score
from netcal.metrics import ECE

""" --- external libraries ---"""
from confens.classifiers.ConfidenceBoosting import ConfidenceBoosting as ConfidenceBoostingClassifier
from confens.classifiers.ConfidenceBagging import ConfidenceBagging as ConfidenceBaggingClassifier

def load_adult_dataset():
    print("--- Loading Dataset ---")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
               'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country', 'income']

    df = pd.read_csv(url, header=None, names=columns, na_values='?', skipinitialspace=True)
    df = df.dropna() # Removing missing values

    # Encoding of the target
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    if df['income'].sum() == 0:
        raise ValueError('ERRORE: Nessuna classe positiva trovata. Controlla la stringa target.')

    """
    Separating the dataset into the Feature Matrix (X) and the Target Vector (y):
    - X: input features used by the model.
    - y: target variables we want to predict. They contains 0 or 1.  
    I chose to drop the 'income' column from X using axis=1 in order to prevent "Data Leakage",
    ensuring the model does not have access to the label during the training process.
    """
    X = df.drop('income', axis=1)
    y = df['income']

    # One-hot encoding in order to convert text variables into binary columns
    X = pd.get_dummies(X, drop_first=True)

    """
    1. train_test_split: it splits the data into training and test sets:
        - the random_state ensures reproducibility of the split;
        - stratify = y maintans the same proportion of classes in both sets as in the original dataset
    2. to_numpy(): it converts pandas DataFrames/Series into NumPy arrays. This prevent sklearn warnings
        about feature names mismatch during cross-validation or model fitting.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    ece = ECE(bins=10).measure(y_prob, y_test)

    return {"Model ": name, "ACC": acc, "MCC": mcc, "ECE": ece}, y_prob

if __name__ == '__main__':
    # 1. Loading data
    X_train, X_test, y_train, y_test = load_adult_dataset()
    results = []

    # Graph setup
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")  # plotting the perfect calibration

    # _____________________________________________________________________________
    # __________________________COMPETITOR DEFINITION______________________________
    # _____________________________________________________________________________

    # 1. Classic classificators (scikit-learn & XGBoost)
    standard_classifiers = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Extra trees", ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=42)),
        ("GaussianNB", GaussianNB()), # Naive Bayes
        ("LDA", LinearDiscriminantAnalysis()),
        ("XGBoost", XGBClassifier(n_estimators=100, base_score=0.5, random_state=42))
    ]

    # 2. Confidence ensembles
    # They use a Decision Tree as a start base
    base_dt = DecisionTreeClassifier(max_depth=6)

    conf_ensembles = [
        ("ConfBag (Native)", ConfidenceBaggingClassifier(clf=base_dt, n_base=20, conf_thr=0.75, weighted=True)),
        ("ConfBoost (Native)", ConfidenceBoostingClassifier(clf=base_dt, n_base=20, conf_thr=0.75, weighted=True, learning_rate=2.0, sampling_ratio=0.5))
    ]

    # _____________________________________________________________________________
    # _______________________________EVALUTATION LOOP_______________________________
    # _____________________________________________________________________________

    # A. Loop over Standard Classifiers (Native + Isotonic + Platt)
    for name, clf in standard_classifiers:
        print(f"\n--- Processing {name} ---")

        # 1. Native
        clf.fit(X_train, y_train)
        res, probs = evaluate_model(name, clf, X_test, y_test)
        results.append(res)

        if name == "XGBoost":
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
            plt.plot(mean_pred, frac_pos, "s--", color="red", alpha=0.6, label=f"{name} (Native)")

        # 2. Calibration with Isotonic Regression
        iso_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
        iso_clf.fit(X_train, y_train)
        res_iso, probs_iso = evaluate_model(f"{name} + Isotonic", iso_clf, X_test, y_test)
        results.append(res_iso)

        # 3. Calibration with Platt Scaling (Sigmoid)
        platt_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
        platt_clf.fit(X_train, y_train)
        res_platt, probs_platt = evaluate_model(f"{name} + Platt", platt_clf, X_test, y_test)
        results.append(res_platt)

    # B. Loop over Confidence Ensembles (Native Only)
    for name, clf in conf_ensembles:
        print(f"\n--- Processing {name} ---")

        clf.fit(X_train, y_train)
        res, probs = evaluate_model(name, clf, X_test, y_test)
        results.append(res)

        # Updating the plot
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        color = "green" if "Boost" in name else "orange"
        plt.plot(mean_pred, frac_pos, "o-", linewidth=2.5, color = color, label=f"{name} (ECE={res['ECE']:.3f})")

    # _____________________________________________________________________________
    # _______________________________FINAL OUTPUT__________________________________
    # _____________________________________________________________________________

    # Create DataFrame and sort by ECE (from best to worst)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="ECE")

    # Print the final output
    print("\n" + "=" * 60)
    print("FINAL OUTPUT (Sorted for best calibration / low ECE)")
    print("=" * 60)
    print(df_results.to_string(index=False))

    # Saving CSV
    df_results.to_csv("benchmark_results_extended.csv", index=False)

    # Saving the graph
    plt.ylabel("Fraction of Positives (Reality)")
    plt.xlabel("Mean Predicted Probability (Confidence)")
    plt.title("Reliability Diagram: ConfBoost vs Standard & Calibrated Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig("reliability_extended.png")

