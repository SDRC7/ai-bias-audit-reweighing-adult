import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from src.config import get_config
from src.data_load import load_adult_openml
from src.preprocess import add_binary_columns, preprocess_fit_transform
from src.utils_repro import ensure_dir

def main():
    cfg = get_config()

    ensure_dir("results/models")
    ensure_dir("results/predictions")

    # Load raw dataset
    df_raw = load_adult_openml(data_home="data/raw/openml")

    # Preprocess
    X, y, sex_bin, _ = preprocess_fit_transform(df_raw, cfg)

    # Load split indices
    train_idx = np.load("results/splits/train_idx.npy")
    test_idx = np.load("results/splits/test_idx.npy")

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Load reweighing weights
    w_train = np.load("results/metrics/reweighing_sample_weight.npy")

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("w_train:", w_train.shape)

    # Train Logistic Regression WITH weights
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train, sample_weight=w_train)

    # Save model
    joblib.dump(clf, "results/models/mitigated_logreg.joblib")

    # Predict on test
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Save predictions
    np.save("results/predictions/mitigated_test_preds.npy", y_pred)
    np.save("results/predictions/mitigated_test_probs.npy", y_prob)

    print("Phase 5 complete.")
    print("Mitigated model trained using sample_weight.")

if __name__ == "__main__":
    main()
