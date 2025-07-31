import os
import warnings
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, confusion_matrix)
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model", "output")


def load_and_prepare_data() -> pd.DataFrame:
    data_path = os.path.join(PROJECT_ROOT, "data", "model", "model_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected file not found: {data_path}")

    return pd.read_csv(data_path)


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
    exclude_cols = ["tesla_close", "timestamp"]
    if "tesla_close_original" in df.columns:
        exclude_cols.append("tesla_close_original")
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df["tesla_close"]
    timestamps = df.get("timestamp")

    return X, y, feature_cols, timestamps


def normalize_target_variable(y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    mean_, std_ = y_train_log.mean(), y_train_log.std()
    y_train_norm = (y_train_log - mean_) / std_
    y_test_norm = (y_test_log - mean_) / std_
    return y_train_norm, y_test_norm, {"mean": mean_, "std": std_}


def inverse_transform_target(y_normalized: np.ndarray, target_scaler: Dict[str, float]) -> np.ndarray:
    y_log = y_normalized * target_scaler["std"] + target_scaler["mean"]
    return np.exp(y_log)


def optimized_preprocessing(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                            feature_cols: List[str]):
    X_train_final = X_train.astype(float).copy()
    X_test_final = X_test.astype(float).copy()
    y_train_final, y_test_final, scaler = normalize_target_variable(y_train, y_test)

    normalization_objects = {
        "target_scaler": scaler,
        "features": "raw_unscaled",
        "optimization": "target_only"
    }

    return (X_train_final, X_test_final, y_train_final, y_test_final, normalization_objects)


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 scaler: Dict[str, float] | None = None) -> float:
    if scaler is not None:
        y_true = inverse_transform_target(y_true, scaler)
        y_pred = inverse_transform_target(y_pred, scaler)

    thresh = np.mean(y_true)
    return np.mean((y_pred > thresh) == (y_true > thresh)) * 100.0


def evaluate_model(y_true, y_pred, model_name: str) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model_name": model_name,
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "accuracy": calculate_direction_accuracy(y_true, y_pred)
    }


def create_models(optimized_xgb=None, optimized_rf=None) -> Dict[str, object]:
    models = {}

    if optimized_rf:
        models["Random Forest (Optimized)"] = optimized_rf
    else:
        models["Random Forest (Optimized)"] = RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

    if optimized_xgb:
        models["XGBoost (Optimized)"] = optimized_xgb
    else:
        models["XGBoost (Optimized)"] = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric="rmse"
        )

    models["SVR"] = SVR(kernel="linear", C=0.01, epsilon=10.0)
    return models


def train_single_model(name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
                       y_test: pd.Series):
    try:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        metrics_test = evaluate_model(y_test, y_pred_test, f"{name}_test")

        result = {
            "model": name,
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": metrics_test.pop("r2"),
            "r2": metrics_test.pop("r2", None),
            "test_rmse": metrics_test["rmse"],
            "test_mae": metrics_test["mae"],
            "test_accuracy": metrics_test["accuracy"],
        }
        return result, model

    except Exception as exc:
        print(f"❌ {name} failed: {exc}")
        return None, None


def train_all_models(X_train, X_test, y_train, y_test, target_scaler=None, timestamps_test=None, optimized_xgb=None,
                     optimized_rf=None):
    models = create_models(optimized_xgb, optimized_rf)
    results, trained = [], {}

    for name, mdl in models.items():
        res, fitted = train_single_model(name, mdl, X_train, X_test, y_train, y_test)
        if res:
            results.append(res)
            trained[name] = fitted
            create_model_plots(
                fitted,
                name,
                X_train,
                y_train,
                X_test,
                y_test,
                target_scaler,
                timestamps_test,
            )

    return results, trained


# Plotting
def create_regression_confusion_matrix(y_true, y_pred, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mean_val = np.mean(y_true)
    cm = confusion_matrix(y_true > mean_val, y_pred > mean_val)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Below", "Above"],
        yticklabels=["Below", "Above"]
    )
    plt.title(f"{model_name}: Direction Confusion Matrix")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def create_loss_curve(model, model_name, X_train, y_train, save_dir):
    if not isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        return

    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    # RandomForest
    if isinstance(model, RandomForestRegressor):
        param_name, param_range = "n_estimators", [50, 100, 150, 200, 250, 300]
        train_sc, val_sc = validation_curve(
            RandomForestRegressor(**{k: v for k, v in model.get_params().items() if k != param_name}),
            X_train,
            y_train,
            param_name=param_name,
            param_range=param_range,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        plt.plot(param_range, np.sqrt(-train_sc.mean(1)), "o-", label="TrainRMSE")
        plt.plot(param_range, np.sqrt(-val_sc.mean(1)), "o-", label="ValRMSE")

    # XGBoost
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        booster = xgb.XGBRegressor(**model.get_params())
        booster.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        res = booster.evals_result()
        plt.plot(res["validation_0"]["rmse"], label="TrainRMSE")
        plt.plot(res["validation_1"]["rmse"], label="ValRMSE")

    plt.title(f"{model_name}: Loss Curve")
    plt.ylabel("RMSE")
    plt.xlabel("Boosting round" if isinstance(model, xgb.XGBRegressor) else "n_estimators")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()


def create_prediction_line_chart(y_true, y_pred, model_name: str, save_dir: str, target_scaler: dict | None = None,
                                 timestamps: pd.Series | None = None, smooth_window: int = 14):
    os.makedirs(save_dir, exist_ok=True)

    if target_scaler:
        y_true_plot = inverse_transform_target(y_true, target_scaler)
        y_pred_plot = inverse_transform_target(y_pred, target_scaler)
        ylabel = "Tesla Close Price ($)"
        title_suffix = "Tesla Stock Price ($)"
    else:
        y_true_plot, y_pred_plot = y_true, y_pred
        ylabel = "Normalized Target Value"
        title_suffix = "Normalized Values"

    df_plot = pd.DataFrame(
        {
            "timestamp": (
                pd.to_datetime(timestamps)
                if timestamps is not None
                else pd.RangeIndex(len(y_true_plot))
            ),
            "actual": y_true_plot,
            "predicted": y_pred_plot,
        }
    ).sort_values("timestamp")

    df_plot["predicted_smooth"] = (df_plot["predicted"].rolling(smooth_window, center=True).mean())

    plt.figure(figsize=(15, 8))
    plt.plot(
        df_plot["timestamp"],
        df_plot["actual"],
        label="Actual Tesla Price",
        linewidth=2,
        alpha=0.9,
    )
    plt.plot(
        df_plot["timestamp"],
        df_plot["predicted_smooth"],
        label=f"Predicted (rolling {smooth_window}d)",
        linewidth=2,
        alpha=0.9,
        color="tab:orange",
    )

    plt.fill_between(
        df_plot["timestamp"],
        df_plot["actual"],
        df_plot["predicted_smooth"],
        color="grey",
        alpha=0.25,
        linewidth=0,
        label="Residual",
    )

    plt.xlabel("Date" if timestamps is not None else "Sample Index")
    plt.ylabel(ylabel)
    plt.title(f"{model_name}: Actual vs Predicted {title_suffix}")
    if timestamps is not None:
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    corr = np.corrcoef(df_plot["actual"], df_plot["predicted"])[0, 1]
    info = (
        f"Correlation: {corr:.4f}\n"
        f"Actual mean: {df_plot['actual'].mean():.2f}\n"
        f"Pred mean:   {df_plot['predicted'].mean():.2f}"
    )
    plt.text(
        0.02,
        0.98,
        info,
        transform=plt.gca().transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_line_chart.png"), dpi=300)
    plt.close()


def create_feature_importance_plot(model, model_name, feature_cols, save_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return

    os.makedirs(save_dir, exist_ok=True)
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .nlargest(top_n, "importance")
        .iloc[::-1]
    )

    plt.figure(figsize=(12, 8))
    plt.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"{model_name}: Top {top_n} Features")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300)
    plt.close()


def create_model_plots(
        model,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_scaler: dict | None = None,
        timestamps_test: pd.Series | None = None,
):
    from sklearn.linear_model import LinearRegression

    model_dir = os.path.join(OUTPUT_DIR, model_name.lower().replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)

    y_pred_train = model.predict(X_train)
    y_pred_test_raw = model.predict(X_test)

    lr = LinearRegression().fit(y_pred_train.reshape(-1, 1), y_train)
    y_pred_test = lr.predict(y_pred_test_raw.reshape(-1, 1))

    create_regression_confusion_matrix(y_test, y_pred_test, model_name, model_dir)
    create_loss_curve(model, model_name, X_train, y_train, model_dir)
    create_prediction_line_chart(
        y_test,
        y_pred_test,
        model_name,
        model_dir,
        target_scaler,
        timestamps_test,
        smooth_window=7
    )
    if hasattr(model, "feature_importances_"):
        create_feature_importance_plot(model, model_name, X_train.columns.tolist(), model_dir)


def apply_smote_regression(X_train, y_train, target_size_multiplier: float = 1.5, k_neighbors: int = 5):
    X_arr = X_train.values
    y_arr = y_train.values
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_arr)
    n_syn = int(len(X_arr) * (target_size_multiplier - 1))
    synth_X, synth_y = [], []

    for _ in range(n_syn):
        idx = np.random.randint(len(X_arr))
        neigh_idx = np.random.choice(knn.kneighbors([X_arr[idx]])[1][0][1:])
        alpha = np.random.random()
        synth_X.append(X_arr[idx] + alpha * (X_arr[neigh_idx] - X_arr[idx]))
        synth_y.append(y_arr[idx] + alpha * (y_arr[neigh_idx] - y_arr[idx]))

    X_aug = pd.DataFrame(np.vstack([X_arr, synth_X]), columns=X_train.columns, index=None)
    y_aug = pd.Series(np.concatenate([y_arr, synth_y]), index=None)
    return X_aug, y_aug


def prepare_data_split(X, y, timestamps=None):
    if timestamps is not None:
        X_tr, X_te, y_tr, y_te, ts_tr, ts_te = train_test_split(X, y, timestamps, test_size=0.2,
                                                                random_state=42, shuffle=True)
        return X_tr, X_te, y_tr, y_te, ts_tr, ts_te
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_tr, X_te, y_tr, y_te, None, None


def save_best_model(best_model, best_model_name, results_df, normalization_objects, feature_cols):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)

    meta = {
        "model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "performance": results_df.iloc[0].to_dict(),
        "params": best_model.get_params(),
        "preprocessing": normalization_objects,
        "n_features": len(feature_cols),
    }
    pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "model_metadata.json"))
    pd.DataFrame({"feature": feature_cols}).to_csv(os.path.join(OUTPUT_DIR, "feature_columns.csv"), index=False)
    joblib.dump(normalization_objects, os.path.join(OUTPUT_DIR, "normalization_objects.joblib"))
    print(f"✅ Saved best model ({best_model_name}) to {model_path} – testR²={meta['performance']['test_r2']:.3f}")


def display_results_summary(results: list[dict]) -> pd.DataFrame:
    results_df = pd.DataFrame(results).sort_values("test_r2", ascending=False)

    print(f"{'Model':<25} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'Acc':<8}")
    print("-" * 65)
    for _, row in results_df.iterrows():
        print(
            f"{row['model']:<25} "
            f"{row['test_r2']:<8.4f} "
            f"{row['test_rmse']:<8.4f} "
            f"{row['test_mae']:<8.4f} "
            f"{row['test_accuracy']:<8.1f}%"
        )
    return results_df


def main() -> None:
    df = load_and_prepare_data()
    X, y, feature_cols, timestamps = prepare_features_target(df)
    X_tr, X_te, y_tr, y_te, ts_tr, ts_te = prepare_data_split(X, y, timestamps)

    X_tr_aug, y_tr_aug = apply_smote_regression(X_tr, y_tr, 1.5, 5)

    (Xtr_final, Xte_final, ytr_final, yte_final, norm_objs,) = optimized_preprocessing(X_tr_aug, X_te, y_tr_aug, y_te,
                                                                                       feature_cols)

    results, models = train_all_models(Xtr_final, Xte_final, ytr_final, yte_final, norm_objs["target_scaler"], ts_te)

    res_df = display_results_summary(results)

    res_df = res_df.sort_values("test_r2", ascending=False).reset_index(drop=True)
    best_name = res_df.loc[0, "model"]
    best_model = models[best_name]

    save_best_model(best_model, best_name, res_df, norm_objs, feature_cols)


if __name__ == "__main__":
    main()
