import os
import warnings
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, confusion_matrix)
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model", "output")


def load_and_prepare_data() -> pd.DataFrame:
    data_path = os.path.join(PROJECT_ROOT, "data", "model", "model_data.csv")
    return pd.read_csv(data_path)


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
    exclude_cols = ["tesla_close", "timestamp"]
    if "tesla_close_original" in df.columns:
        exclude_cols.append("tesla_close_original")
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    y = df["tesla_close"]
    timestamps = df["timestamp"] if "timestamp" in df.columns else None
    
    return X, y, feature_cols, timestamps


def apply_target_normalization(y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.Series, pd.Series, Dict]:
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    mean_val = y_train_log.mean()
    std_val = y_train_log.std()
    
    y_train_norm = (y_train_log - mean_val) / std_val
    y_test_norm = (y_test_log - mean_val) / std_val
    
    target_scaler = {"mean": mean_val, "std": std_val}
    return y_train_norm, y_test_norm, target_scaler


def inverse_transform_target(y_normalized: np.ndarray, target_scaler: Dict) -> np.ndarray:
    y_log = y_normalized * target_scaler["std"] + target_scaler["mean"]
    return np.exp(y_log)


def apply_smote_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                          target_multiplier: float = 1.5, k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
    X_array = X_train.values
    y_array = y_train.values
    
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    knn.fit(X_array)
    
    n_original = len(X_array)
    n_synthetic = int(n_original * (target_multiplier - 1))
    
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        idx = np.random.randint(0, n_original)
        sample_x = X_array[idx]
        sample_y = y_array[idx]
        
        distances, indices = knn.kneighbors([sample_x])
        neighbor_indices = indices[0][1:]
        
        neighbor_idx = np.random.choice(neighbor_indices)
        neighbor_x = X_array[neighbor_idx]
        neighbor_y = y_array[neighbor_idx]
        
        alpha = np.random.random()
        synthetic_x = sample_x + alpha * (neighbor_x - sample_x)
        synthetic_y_val = sample_y + alpha * (neighbor_y - sample_y)
        
        synthetic_X.append(synthetic_x)
        synthetic_y.append(synthetic_y_val)
    
    X_augmented = np.vstack([X_array, np.array(synthetic_X)])
    y_augmented = np.concatenate([y_array, np.array(synthetic_y)])
    
    X_augmented = pd.DataFrame(X_augmented, columns=X_train.columns)
    y_augmented = pd.Series(y_augmented)
    
    return X_augmented, y_augmented


def calculate_direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_val = np.mean(y_true)
    pred_direction = y_pred > mean_val
    true_direction = y_true > mean_val
    return np.mean(pred_direction == true_direction) * 100


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy = calculate_direction_accuracy(y_true, y_pred)
    
    return {
        "model": model_name,
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_accuracy": accuracy
    }


def train_single_model(name: str, model, X_train, X_test, y_train, y_test) -> Tuple[Dict, object]:
    try:
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        result = evaluate_model_performance(y_test, y_pred_test, name)
        return result, model
    except Exception as exc:
        print(f"ERROR {name} failed: {exc}")
        return None, None


def create_confusion_matrix_plot(y_true, y_pred, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    mean_val = np.mean(y_true)
    y_true_class = (y_true > mean_val).astype(int)
    y_pred_class = (y_pred > mean_val).astype(int)
    
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Below Mean", "Above Mean"],
                yticklabels=["Below Mean", "Above Mean"])
    plt.title(f"{model_name}: Direction Prediction Confusion Matrix")
    plt.xlabel("Predicted Direction")
    plt.ylabel("Actual Direction")
    
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_loss_curve_plot(model, model_name, X_train, y_train, save_dir):
    if not hasattr(model, "n_estimators"):
        return
        
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    if isinstance(model, xgb.XGBRegressor):
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        eval_set = [(X_train_split, y_train_split), (X_val_split, y_val_split)]
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train_split, y_train_split, eval_set=eval_set, verbose=False)
        
        results = model_copy.evals_result()
        train_rmse = results["validation_0"]["rmse"]
        val_rmse = results["validation_1"]["rmse"]
        
        boosting_rounds = range(len(train_rmse))
        plt.plot(boosting_rounds, train_rmse, color="blue", label="Training RMSE", linewidth=2)
        plt.plot(boosting_rounds, val_rmse, color="red", label="Validation RMSE", linewidth=2)
        
        plt.xlabel("Boosting Round")
        plt.ylabel("RMSE")
        plt.title(f"{model_name}: Training vs Validation RMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    elif isinstance(model, RandomForestRegressor):
        param_name = "n_estimators"
        param_range = [50, 100, 150, 200, 250]
        
        train_scores, val_scores = validation_curve(
            type(model)(**{k: v for k, v in model.get_params().items() if k != param_name}),
            X_train, y_train, param_name=param_name, param_range=param_range,
            cv=3, scoring="neg_mean_squared_error", n_jobs=-1
        )
        
        train_rmse = np.sqrt(-train_scores.mean(axis=1))
        val_rmse = np.sqrt(-val_scores.mean(axis=1))
        train_std = np.sqrt(-train_scores).std(axis=1)
        val_std = np.sqrt(-val_scores).std(axis=1)
        
        plt.plot(param_range, train_rmse, "o-", color="blue", label="Training RMSE")
        plt.fill_between(param_range, train_rmse - train_std, train_rmse + train_std, alpha=0.2, color="blue")
        plt.plot(param_range, val_rmse, "o-", color="red", label="Validation RMSE")
        plt.fill_between(param_range, val_rmse - val_std, val_rmse + val_std, alpha=0.2, color="red")
        
        plt.xlabel("Number of Estimators")
        plt.ylabel("RMSE")
        plt.title(f"{model_name}: Training vs Validation RMSE")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_prediction_line_chart(y_true, y_pred, model_name, save_dir, target_scaler=None, timestamps=None):
    os.makedirs(save_dir, exist_ok=True)
    
    if target_scaler is not None:
        y_true_dollars = inverse_transform_target(y_true, target_scaler)
        y_pred_dollars = inverse_transform_target(y_pred, target_scaler)
        ylabel = "Tesla Stock Price ($)"
        title_suffix = "Tesla Stock Price ($)"
    else:
        y_true_dollars = y_true
        y_pred_dollars = y_pred
        ylabel = "Normalized Target Value"
        title_suffix = "Normalized Values"
    
    plt.figure(figsize=(15, 8))
    
    if timestamps is not None:
        df_plot = pd.DataFrame({
            "timestamp": pd.to_datetime(timestamps),
            "actual": y_true_dollars,
            "predicted": y_pred_dollars
        })
        
        df_plot = df_plot.sort_values("timestamp")
        
        x_axis = df_plot["timestamp"]
        y_true_sorted = df_plot["actual"]
        y_pred_sorted = df_plot["predicted"]
        xlabel = "Date"
    else:
        x_axis = range(len(y_true_dollars))
        y_true_sorted = y_true_dollars
        y_pred_sorted = y_pred_dollars
        xlabel = "Sample Index (Time Order)"
    
    plt.plot(x_axis, y_true_sorted, color="blue", label="Actual Tesla Price", alpha=0.7, linewidth=2)
    plt.plot(x_axis, y_pred_sorted, color="red", label="Predicted Tesla Price", alpha=0.7, linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{model_name}: Actual vs Predicted {title_suffix} Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if timestamps is not None:
        plt.xticks(rotation=45)
    
    correlation = np.corrcoef(y_true_sorted, y_pred_sorted)[0, 1]
    mean_actual = np.mean(y_true_sorted)
    mean_pred = np.mean(y_pred_sorted)
    
    info_text = f"Correlation: {correlation:.4f}\nActual Mean: ${mean_actual:.2f}\nPredicted Mean: ${mean_pred:.2f}"
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "prediction_line_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_feature_importance_plot(model, model_name, feature_cols, save_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    
    top_features = importance_df.tail(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features["importance"], color="steelblue", alpha=0.7)
    
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title(f"{model_name}: Top {top_n} Most Important Features", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, top_features["importance"])):
        plt.text(value + max(top_features["importance"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{value:.4f}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_all_model_plots(model, model_name, X_train, y_train, X_test, y_test, target_scaler=None, timestamps_test=None):
    model_dir = os.path.join(OUTPUT_DIR, model_name.lower().replace(" ", "_"))
    
    y_pred_test = model.predict(X_test)
    
    create_confusion_matrix_plot(y_test, y_pred_test, model_name, model_dir)
    create_loss_curve_plot(model, model_name, X_train, y_train, model_dir)
    create_prediction_line_chart(y_test, y_pred_test, model_name, model_dir, target_scaler, timestamps_test)
    
    if hasattr(model, "feature_importances_"):
        feature_cols = X_train.columns.tolist() if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(X_train.shape[1])]
        create_feature_importance_plot(model, model_name, feature_cols, model_dir)
    
    return model_dir


def initialize_models():
    models = {
        "XGBoost (Optimized)": xgb.XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric="rmse"
        ),
        "Random Forest (Optimized)": RandomForestRegressor(
            n_estimators=400,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        "SVR": SVR(
            kernel="linear",
            C=0.01,
            epsilon=10.0
        )
    }
    return models


def train_all_models(X_train, X_test, y_train, y_test, target_scaler=None, timestamps_test=None):
    models = initialize_models()
    results = []
    trained_models = {}
    
    for name, model in models.items():
        result, trained_model = train_single_model(name, model, X_train, X_test, y_train, y_test)
        if result is not None:
            results.append(result)
            trained_models[name] = trained_model
            create_all_model_plots(trained_model, name, X_train, y_train, X_test, y_test, target_scaler, timestamps_test)
    
    return results, trained_models


def display_results_summary(results: List[Dict]) -> pd.DataFrame:
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


def save_best_model(best_model, best_model_name: str, results_df: pd.DataFrame, target_scaler: Dict, feature_cols: List[str]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(best_model, model_path)
    
    best_result = results_df.iloc[0]
    meta = {
        "model_name": best_model_name,
        "performance": {
            "test_r2": float(best_result["test_r2"]),
            "test_rmse": float(best_result["test_rmse"]),
            "test_mae": float(best_result["test_mae"]),
            "test_accuracy": float(best_result["test_accuracy"])
        },
        "preprocessing": target_scaler,
        "n_features": len(feature_cols),
    }
    pd.Series(meta).to_json(os.path.join(OUTPUT_DIR, "model_metadata.json"))
    pd.DataFrame({"feature": feature_cols}).to_csv(os.path.join(OUTPUT_DIR, "feature_columns.csv"), index=False)
    normalization_objects = {"target_scaler": target_scaler}
    joblib.dump(normalization_objects, os.path.join(OUTPUT_DIR, "normalization_objects.joblib"))
    print(f"SUCCESS Saved best model ({best_model_name}) to {model_path} – testR²={meta['performance']['test_r2']:.3f}")


def main():
    df = load_and_prepare_data()
    X, y, feature_cols, timestamps = prepare_features_target(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    timestamps_train, timestamps_test = None, None
    if timestamps is not None:
        timestamps_train, timestamps_test = train_test_split(timestamps, test_size=0.2, random_state=42, shuffle=True)
    
    X_train_aug, y_train_aug = apply_smote_regression(X_train, y_train, target_multiplier=1.5, k_neighbors=5)
    
    y_train_norm, y_test_norm, target_scaler = apply_target_normalization(y_train_aug, y_test)
    
    results, trained_models = train_all_models(X_train_aug, X_test, y_train_norm, y_test_norm, target_scaler, timestamps_test)
    
    results_df = display_results_summary(results)
    
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    
    save_best_model(best_model, best_model_name, results_df, target_scaler, feature_cols)


if __name__ == "__main__":
    main()
