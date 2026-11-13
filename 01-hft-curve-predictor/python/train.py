"""
Train multi-horizon mid-price "curve" prediction models.

For each horizon h in {1,5,10,20,50} ticks ahead, trains a gradient-boosted
regressor predicting forward return, and a classifier predicting direction.
Uses LightGBM if installed, else falls back to sklearn's
HistGradientBoosting (no external deps needed either way).

Exports the direction-classifier ensemble to ONNX for the C++ inference engine.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error

try:
    from sklearn.metrics import root_mean_squared_error

    def rmse(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:  # older sklearn
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

HORIZONS = (1, 5, 10, 20, 50)

FEATURE_COLS = None  # set at runtime (everything except mid_price/targets)


def get_feature_cols(df):
    exclude = {"mid_price"} | {f"fwd_ret_{h}" for h in HORIZONS} | {f"fwd_dir_{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in exclude]


def train_one_horizon(df, feat_cols, h, out_dir):
    X = df[feat_cols].values
    y_dir = df[f"fwd_dir_{h}"].astype(int).values
    y_ret = df[f"fwd_ret_{h}"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]  # last fold = most realistic (trailing test)
    X_train, X_test = X[train_idx], X[test_idx]
    y_dir_train, y_dir_test = y_dir[train_idx], y_dir[test_idx]
    y_ret_train, y_ret_test = y_ret[train_idx], y_ret[test_idx]

    t0 = time.time()
    if HAVE_LGB:
        clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, verbosity=-1)
        reg = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, verbosity=-1)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.05)
        reg = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05)

    clf.fit(X_train, y_dir_train)
    reg.fit(X_train, y_ret_train)
    fit_time = time.time() - t0

    dir_pred = clf.predict(X_test)
    ret_pred = reg.predict(X_test)

    acc = accuracy_score(y_dir_test, dir_pred)
    rmse_val = rmse(y_ret_test, ret_pred)
    # naive baseline: predict "no change" / zero return every time
    naive_acc = max(np.mean(y_dir_test == 1), np.mean(y_dir_test == -1), np.mean(y_dir_test == 0))

    print(
        f"[h={h:>3}] fit={fit_time:5.2f}s  dir_acc={acc:.4f} "
        f"(naive={naive_acc:.4f})  ret_rmse={rmse_val:.6f}  backend={'lgb' if HAVE_LGB else 'sklearn'}"
    )

    import joblib

    joblib.dump(clf, f"{out_dir}/clf_h{h}.joblib")
    joblib.dump(reg, f"{out_dir}/reg_h{h}.joblib")

    return {
        "horizon": h,
        "dir_accuracy": acc,
        "naive_accuracy": naive_acc,
        "ret_rmse": rmse_val,
        "fit_time_sec": fit_time,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features.csv")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--export-onnx", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    feat_cols = get_feature_cols(df)
    print(f"Loaded {len(df):,} rows, {len(feat_cols)} features")
    print("Features:", feat_cols)

    results = []
    for h in HORIZONS:
        results.append(train_one_horizon(df, feat_cols, h, args.out_dir))

    with open(f"{args.out_dir}/metrics.json", "w") as f:
        json.dump({"feature_cols": feat_cols, "results": results}, f, indent=2)
    print(f"\nSaved metrics to {args.out_dir}/metrics.json")

    if args.export_onnx:
        export_onnx(args.out_dir, feat_cols)


def export_onnx(out_dir, feat_cols):
    """Export the h=1 direction classifier to ONNX for the C++ inference engine."""
    try:
        import joblib
        from skl2onnx import to_onnx
    except ImportError:
        print("skl2onnx not installed -- skipping ONNX export. "
              "pip install skl2onnx onnxruntime to enable the C++ inference path.")
        return

    clf = joblib.load(f"{out_dir}/clf_h1.joblib")
    n_features = len(feat_cols)

    if HAVE_LGB:
        # LightGBM needs onnxmltools, not skl2onnx -- keep this path optional/best-effort
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType

            onnx_model = convert_lightgbm(clf, initial_types=[("input", FloatTensorType([None, n_features]))])
        except ImportError:
            print("onnxmltools not installed -- skipping ONNX export for LightGBM model. "
                  "pip install onnxmltools, or rerun training without lightgbm installed "
                  "to use the sklearn backend (exports via skl2onnx).")
            return
    else:
        onnx_model = to_onnx(clf, np.zeros((1, n_features), dtype=np.float32))

    with open(f"{out_dir}/clf_h1.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    with open(f"{out_dir}/feature_cols.json", "w") as f:
        json.dump(feat_cols, f)
    print(f"Exported ONNX model to {out_dir}/clf_h1.onnx")


if __name__ == "__main__":
    main()
