import os
import warnings
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans 

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import joblib

GLOBAL_RANDOM_STATE = 42
np.random.seed(GLOBAL_RANDOM_STATE)


class PipelineConfig:
    SUBSET = "FD002" 

    DATA_DIR = os.path.join("data", "processed")
    TRAIN_FILE = os.path.join(DATA_DIR, f"train_{SUBSET}_cleaned.csv")
    TEST_FILE  = os.path.join(DATA_DIR, f"test_{SUBSET}_cleaned.csv")
    RUL_FILE   = os.path.join(DATA_DIR, f"rul_{SUBSET}.csv")

    OPERATIONAL_COLUMNS = ['op_setting1', 'op_setting2', 'op_setting3']
    N_OPERATING_CLUSTERS = 6

    WINDOW_SIZE = 30
    MIN_SHIFT = 5
    MAX_SHIFT = 20
    TEST_STRIDE = 3

    RUL_CAP = 125
    NORMALIZATION_CYCLES = 10

    PCA_VARIANCE = 0.5

    # --- CQR ---
    ALPHA = 0.1
    CALIBRATION_SPLIT = 0.3
    CQR_OVER_TO_UNDER_RATIO = 3.0

    FFT_COEFF_MAX = 10 if SUBSET == "FD004" else 3

    RANDOM_STATE = GLOBAL_RANDOM_STATE
    CV_FOLDS = 5
    N_JOBS = 3

    @classmethod
    def validate(cls):
        assert cls.SUBSET in {"FD002", "FD004"}, "SUBSET must be FD002 or FD004"
        assert cls.WINDOW_SIZE > 0, "Window size must be positive"
        assert 0 < cls.ALPHA < 1, "Alpha must be between 0 and 1"
        assert 0 < cls.PCA_VARIANCE <= 1, "PCA variance must be between 0 and 1"
        assert cls.MIN_SHIFT < cls.MAX_SHIFT, "MIN_SHIFT must be less than MAX_SHIFT"

Config = PipelineConfig
Config.validate()


def load_raw_data():
    for file_path in [Config.TRAIN_FILE, Config.TEST_FILE, Config.RUL_FILE]:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")

    df_train = pd.read_csv(Config.TRAIN_FILE, dtype={"engine_id": int, "cycle": int})
    df_test = pd.read_csv(Config.TEST_FILE, dtype={"engine_id": int, "cycle": int})
    df_rul = pd.read_csv(Config.RUL_FILE, header=None, names=["engine_id", "true_RUL"])

    df_rul["engine_id"] = df_rul["engine_id"].astype(int)
    df_rul["true_RUL"] = pd.to_numeric(df_rul["true_RUL"], errors="coerce")

    if df_rul["true_RUL"].isna().any():
        raise ValueError("NaN values in RUL labels")

    missing_ops = [col for col in Config.OPERATIONAL_COLUMNS if col not in df_train.columns]
    if missing_ops:
        raise ValueError(f"Missing operational columns: {missing_ops}")

    return df_train, df_test, df_rul


def create_training_rul_labels(df_train, rul_cap=Config.RUL_CAP):
    df = df_train.copy()

    last_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    last_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(last_cycles, on='engine_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=rul_cap)
    df = df.drop(columns=['max_cycle'])

    return df

def create_test_final_samples(df_test, df_rul):
    final_samples = df_test.loc[df_test.groupby('engine_id')['cycle'].idxmax()].copy()
    final_samples = final_samples.sort_values('engine_id').reset_index(drop=True)

    final_samples = final_samples.merge(df_rul, on='engine_id', how='left')
    final_samples['true_RUL'] = final_samples['true_RUL'].clip(upper=Config.RUL_CAP)

    return final_samples

def identify_operating_regimes_and_normalize(df_train, df_test, n_clusters=Config.N_OPERATING_CLUSTERS):
    df_train_norm = df_train.copy()
    df_test_norm = df_test.copy()

    train_ops = df_train_norm[Config.OPERATIONAL_COLUMNS]
    test_ops = df_test_norm[Config.OPERATIONAL_COLUMNS]

    kmeans = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=10)
    train_regimes = kmeans.fit_predict(train_ops)
    test_regimes = kmeans.predict(test_ops)

    df_train_norm['regime_id'] = train_regimes
    df_test_norm['regime_id'] = test_regimes

    sensor_cols = [col for col in df_train_norm.columns if col.startswith('sensor')]

    for sensor in sensor_cols:
        df_train_norm[sensor] = df_train_norm[sensor].astype('float64')
        df_test_norm[sensor] = df_test_norm[sensor].astype('float64')

        regime_stats = df_train_norm.groupby('regime_id')[sensor].agg(['mean', 'std']).reset_index()
        regime_stats.columns = ['regime_id', f'{sensor}_mean', f'{sensor}_std']

        regime_stats[f'{sensor}_std'] = regime_stats[f'{sensor}_std'].fillna(1.0)
        regime_stats.loc[regime_stats[f'{sensor}_std'] == 0, f'{sensor}_std'] = 1.0

        df_with_stats = df_train_norm.merge(regime_stats, on='regime_id')
        df_train_norm[sensor] = (df_with_stats[sensor] - df_with_stats[f'{sensor}_mean']) / df_with_stats[f'{sensor}_std']

        df_test_with_stats = df_test_norm.merge(regime_stats, on='regime_id', how='left')

        missing_mask = df_test_with_stats[f'{sensor}_mean'].isna()
        if missing_mask.any():
            global_mean = df_train[sensor].mean()
            global_std = df_train[sensor].std()
            df_test_with_stats.loc[missing_mask, f'{sensor}_mean'] = global_mean
            df_test_with_stats.loc[missing_mask, f'{sensor}_std'] = global_std

        df_test_norm[sensor] = (df_test_with_stats[sensor] - df_test_with_stats[f'{sensor}_mean']) / df_test_with_stats[f'{sensor}_std']

    for op_col in Config.OPERATIONAL_COLUMNS:
        train_mean = df_train_norm[op_col].mean()
        train_std = df_train_norm[op_col].std()
        if train_std == 0:
            train_std = 1.0

        df_train_norm[op_col] = (df_train_norm[op_col] - train_mean) / train_std
        df_test_norm[op_col] = (df_test_norm[op_col] - train_mean) / train_std

    return df_train_norm, df_test_norm, kmeans


def generate_training_windows(
    df,
    window_size=Config.WINDOW_SIZE,
    min_shift=Config.MIN_SHIFT,
    max_shift=Config.MAX_SHIFT
):
    np.random.seed(Config.RANDOM_STATE)
    windows = []
    window_counter = 0

    for engine_id in sorted(df['engine_id'].unique()):
        engine_df = df[df['engine_id'] == engine_id].reset_index(drop=True)
        max_start_idx = len(engine_df) - window_size

        if max_start_idx < 0:
            continue

        stride = np.random.randint(min_shift, max_shift + 1)

        for start_idx in range(0, max_start_idx + 1, stride):
            window = engine_df.iloc[start_idx:start_idx + window_size].copy()

            window['window_id'] = f"train_{window_counter}"
            window['window_start_cycle'] = window['cycle'].iloc[0]
            window['window_end_cycle'] = window['cycle'].iloc[-1]
            window['window_target_rul'] = window['RUL'].iloc[-1]

            windows.append(window)
            window_counter += 1

    df_windowed = pd.concat(windows, ignore_index=True)
    return df_windowed


def generate_test_windows(df, window_size=Config.WINDOW_SIZE, stride=Config.TEST_STRIDE):
    windows = []

    for engine_id in sorted(df['engine_id'].unique()):
        engine_df = df[df['engine_id'] == engine_id].reset_index(drop=True)
        engine_length = len(engine_df)

        if engine_length < window_size:
            window = engine_df.copy()
            window['window_id'] = f"{engine_id}_0_short"
            window['window_start_cycle'] = window['cycle'].iloc[0]
            window['window_end_cycle'] = window['cycle'].iloc[-1]
            windows.append(window)
        else:
            max_start_idx = engine_length - window_size
            for start_idx in range(0, max_start_idx + 1, stride):
                window = engine_df.iloc[start_idx:start_idx + window_size].copy()
                window['window_id'] = f"{engine_id}_{start_idx}"
                window['window_start_cycle'] = window['cycle'].iloc[0]
                window['window_end_cycle'] = window['cycle'].iloc[-1]
                windows.append(window)

    df_windowed = pd.concat(windows, ignore_index=True)
    return df_windowed

def convert_to_tsfresh_format(df_windowed, dataset_name):

    sensor_cols = [col for col in df_windowed.columns if col.startswith('sensor')]
    columns_needed = ['window_id', 'cycle', 'window_start_cycle'] + sensor_cols
    df_subset = df_windowed[columns_needed].copy()

    df_subset['relative_time'] = df_subset['cycle'] - df_subset['window_start_cycle']

    df_long = df_subset.melt(
        id_vars=['window_id', 'relative_time'],
        value_vars=sensor_cols,
        var_name='kind',
        value_name='value'
    )

    df_long = df_long.rename(columns={
        'window_id': 'id',
        'relative_time': 'time'
    })

    return df_long


def create_tsfresh_feature_settings():
    _corr_lags = [1, 3] if Config.SUBSET == "FD004" else [1]
    feature_settings = {
        # Basic statistical features
        "mean": None,
        "standard_deviation": None,
        "root_mean_square": None,
        "maximum": None,
        "minimum": None,

        # Change-based features
        "mean_change": None,

        # Temporal correlation features
        "autocorrelation": [{"lag": l} for l in _corr_lags],
        "partial_autocorrelation": [{"lag": l} for l in _corr_lags],
        "time_reversal_asymmetry_statistic": [{"lag": l} for l in _corr_lags],

        # Extrema location features
        "first_location_of_maximum": None,
        "first_location_of_minimum": None,
        "last_location_of_maximum": None,
        "last_location_of_minimum": None,

        # Trend analysis
        "linear_trend": [
            {"attr": "intercept"},
            {"attr": "slope"},
            {"attr": "stderr"}
        ],
        "linear_trend_timewise": [
            {"attr": "intercept"},
            {"attr": "slope"}
        ],

        # Stationarity test
        "augmented_dickey_fuller": [
            {"attr": "teststat"},
            {"attr": "pvalue"}
        ],

        # Complexity measures
        "lempel_ziv_complexity": [
            {"bins": 5}
        ],
        "permutation_entropy": [
            {"dimension": 3, "tau": 1}
        ],

        # Frequency domain features
        "fft_coefficient": [
            {"coeff": i, "attr": "abs"} for i in range(Config.FFT_COEFF_MAX)
        ],
        "fft_aggregated": [
            {"aggtype": "centroid"},
            {"aggtype": "variance"},
            {"aggtype": "skew"},
            {"aggtype": "kurtosis"}
        ],

        "c3": [{"lag": 1}],
        "cid_ce": [{"normalize": True}]
    }

    return feature_settings


def extract_tsfresh_features(df_long, dataset_name, feature_settings):

    X_features = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=feature_settings,
        impute_function=impute,
        show_warnings=False,
        n_jobs=3,
        disable_progressbar=False
    )

    X_features = X_features.dropna(axis=1, how='all')
    return X_features


def pca_transform(X_train, X_test, pca_variance=Config.PCA_VARIANCE):
    common_features = sorted(set(X_train.columns).intersection(set(X_test.columns)))
    X_train = X_train[common_features]
    X_test = X_test[common_features]

    clean_columns = X_train.columns[~X_train.isna().any()].tolist()
    X_train_clean = X_train[clean_columns]
    X_test_clean = X_test[clean_columns]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train_clean.index, columns=X_train_clean.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test_clean.index, columns=X_test_clean.columns)

    pca = PCA(n_components=pca_variance, svd_solver='full', random_state=Config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_columns = [f'PC_{i+1}' for i in range(X_train_pca.shape[1])]
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train_scaled.index, columns=pca_columns)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test_scaled.index, columns=pca_columns)

    explained_variance = pca.explained_variance_ratio_.sum()

    preprocessing_objects = {
        'scaler': scaler,
        'pca': pca,
        'clean_columns': clean_columns,
        'explained_variance': explained_variance
    }

    return X_train_pca, X_test_pca, preprocessing_objects


def apply_random_forest_selection(X_train_pca, X_test_pca, y_train, n_estimators=100):
    rf_selector = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=Config.RANDOM_STATE,
        n_jobs=Config.N_JOBS,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        oob_score=True
    )

    rf_selector.fit(X_train_pca, y_train)

    feature_importances = pd.Series(
        rf_selector.feature_importances_,
        index=X_train_pca.columns
    ).sort_values(ascending=False)

    threshold = feature_importances.median()
    selector = SelectFromModel(rf_selector, threshold=threshold, prefit=True)
    selected_mask = selector.get_support()

    X_train_selected = X_train_pca.loc[:, selected_mask]
    X_test_selected = X_test_pca.loc[:, selected_mask]

    return X_train_selected, X_test_selected, rf_selector


def train_gbm_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.8, 1.0],
    }

    gbm_model = GradientBoostingRegressor(random_state=Config.RANDOM_STATE)

    random_search = RandomizedSearchCV(
        estimator=gbm_model,
        param_distributions=param_grid,
        n_iter=20,
        cv=Config.CV_FOLDS,
        scoring='neg_mean_absolute_error',
        n_jobs=Config.N_JOBS,
        random_state=Config.RANDOM_STATE,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    return best_model


def s_score_function(y_true, y_pred, a1=10, a2=13):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    diff = y_true - y_pred

    score = np.where(
        diff < 0,
        np.exp(-diff / a1) - 1,
        np.exp(diff / a2) - 1
    )

    return np.sum(score)

def evaluate_point_predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    s_score = s_score_function(y_test, y_pred)

    results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        's_score': s_score,
        'predictions': y_pred
    }

    return results


class ConformalizedQuantileRegressor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.lower_model = None
        self.upper_model = None
        self.adjustment = None
        self.is_fitted = False

        self._adj_low = None
        self._adj_high = None

        self._ratio = getattr(Config, "CQR_OVER_TO_UNDER_RATIO", 1.0)
        if not isinstance(self._ratio, (int, float)) or self._ratio <= 0:
            self._ratio = 1.0

    def fit_and_calibrate(self, X_train, y_train, X_cal, y_cal):
        model_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": Config.RANDOM_STATE,
        }

        alpha_low = self.alpha / (1.0 + self._ratio)
        alpha_high = self.alpha - alpha_low

        tau_low = alpha_low
        tau_high = 1.0 - alpha_high

        from sklearn.ensemble import GradientBoostingRegressor

        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=tau_low, **model_params)
        self.lower_model.fit(X_train, y_train)

        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=tau_high, **model_params)
        self.upper_model.fit(X_train, y_train)

        y_cal_true = np.asarray(y_cal)
        y_cal_lower = self.lower_model.predict(X_cal)
        y_cal_upper = self.upper_model.predict(X_cal)

        s_low = np.maximum(y_cal_lower - y_cal_true, 0.0)
        s_high = np.maximum(y_cal_true - y_cal_upper, 0.0)

        n_cal = len(y_cal_true)
        q_low_level = min((1.0 - alpha_low) * (1.0 + 1.0 / n_cal), 1.0)
        q_high_level = min((1.0 - alpha_high) * (1.0 + 1.0 / n_cal), 1.0)

        self._adj_low = float(np.quantile(s_low, q_low_level))
        self._adj_high = float(np.quantile(s_high, q_high_level))

        self.adjustment = max(self._adj_low, self._adj_high)
        self.is_fitted = True

    def predict_intervals(self, X_test):
        if not self.is_fitted:
            raise ValueError("CQR model must be fitted first")

        y_lower_base = self.lower_model.predict(X_test)
        y_upper_base = self.upper_model.predict(X_test)

        adj_low = self._adj_low if self._adj_low is not None else self.adjustment
        adj_high = self._adj_high if self._adj_high is not None else self.adjustment

        y_lower = np.maximum(y_lower_base - adj_low, 0.0)
        y_upper = y_upper_base + adj_high

        return y_lower, y_upper


def evaluate_prediction_intervals(y_true, y_lower, y_upper):
    y_true = np.array(y_true)

    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    interval_widths = y_upper - y_lower
    mpiw = np.mean(interval_widths)

    rul_range = np.max(y_true) - np.min(y_true)
    nmpiw = mpiw / rul_range if rul_range > 0 else 0

    rul_bins = [0, 20, 40, 60, 80, 100, np.inf]
    rul_labels = ["0-20", "20-40", "40-60", "60-80", "80-100", "100+"]
    conditional_coverage = {}

    for i, (lower_bound, upper_bound) in enumerate(zip(rul_bins[:-1], rul_bins[1:])):
        mask = (y_true >= lower_bound) & (y_true < upper_bound)
        if np.sum(mask) > 0:
            bin_coverage = np.mean(
                (y_true[mask] >= y_lower[mask]) & (y_true[mask] <= y_upper[mask])
            )
            conditional_coverage[rul_labels[i]] = bin_coverage

    results = {
        "coverage": coverage,
        "mpiw": mpiw,
        "nmpiw": nmpiw,
        "conditional_coverage": conditional_coverage,
        "lower_bounds": y_lower,
        "upper_bounds": y_upper,
    }

    return results


def engine_aware_split(df_train_windowed, X_train, y_train, cal_fraction=0.3):
    window_engine_map = df_train_windowed[['window_id', 'engine_id']].drop_duplicates()
    window_to_engine = dict(zip(window_engine_map['window_id'], window_engine_map['engine_id']))

    window_ids = X_train.index.tolist()
    engine_ids = [window_to_engine[wid] for wid in window_ids]

    unique_engines = sorted(set(engine_ids))

    train_engines, cal_engines = train_test_split(
        unique_engines,
        test_size=cal_fraction,
        random_state=Config.RANDOM_STATE,
        shuffle=True
    )

    train_engines = set(train_engines)
    cal_engines = set(cal_engines)

    overlap = train_engines & cal_engines
    assert len(overlap) == 0, f"Leakage: engines in both splits: {sorted(overlap)}"

    train_mask = [eid in train_engines for eid in engine_ids]
    cal_mask   = [eid in cal_engines   for eid in engine_ids]

    X_train_split = X_train[train_mask]
    X_cal_split   = X_train[cal_mask]
    y_train_split = y_train[train_mask]
    y_cal_split   = y_train[cal_mask]

    return X_train_split, X_cal_split, y_train_split, y_cal_split


def run_complete_pipeline():

    # Phase 1: Data Loading and Preprocessing
    df_train, df_test, df_rul = load_raw_data()

    df_train_with_rul = create_training_rul_labels(df_train)
    df_test_final = create_test_final_samples(df_test, df_rul)

    df_train_norm, df_test_norm, kmeans_model = identify_operating_regimes_and_normalize(
        df_train_with_rul, df_test
    )

    # Phase 2: Window Generation
    df_train_windowed = generate_training_windows(df_train_norm)
    df_test_windowed = generate_test_windows(df_test_norm)

    # Phase 3: TSFresh Feature Extraction
    df_train_long = convert_to_tsfresh_format(df_train_windowed, "Training")
    df_test_long = convert_to_tsfresh_format(df_test_windowed, "Test")

    tsfresh_settings = create_tsfresh_feature_settings()
    X_train = extract_tsfresh_features(df_train_long, "Training", tsfresh_settings)
    X_test = extract_tsfresh_features(df_test_long, "Test", tsfresh_settings)

    def generate_window_rul_labels(df_windowed):
        window_rul_labels = {}
        for window_id in df_windowed["window_id"].unique():
            window_data = df_windowed[df_windowed["window_id"] == window_id]
            if "RUL" in window_data.columns:
                window_rul_labels[window_id] = window_data.iloc[-1]["RUL"]
        return pd.Series(window_rul_labels, name="RUL")

    y_train_rul = generate_window_rul_labels(df_train_windowed)

    def extract_final_test_windows(X_test_features, df_test_windowed, df_test_final):
        final_window_ids = []
        for engine_id in sorted(df_test_final["engine_id"].unique()):
            engine_windows = df_test_windowed[
                df_test_windowed["window_id"].str.startswith(f"{engine_id}_")
            ]
            if len(engine_windows) > 0:
                final_window = engine_windows.loc[engine_windows["window_end_cycle"].idxmax()]
                final_window_ids.append(final_window["window_id"])

        X_test_final = X_test_features.loc[final_window_ids]

        engine_rul_map = dict(zip(df_test_final["engine_id"], df_test_final["true_RUL"]))
        y_test_final_values = []
        for window_id in final_window_ids:
            engine_id = int(window_id.split("_")[0])
            y_test_final_values.append(engine_rul_map[engine_id])

        y_test_final = pd.Series(y_test_final_values, index=final_window_ids, name="true_RUL")
        return X_test_final, y_test_final

    X_test_final, y_test_final = extract_final_test_windows(X_test, df_test_windowed, df_test_final)

    X_train_aligned = X_train.loc[y_train_rul.index]
    y_train_aligned = y_train_rul

    # Phase 4: Feature Preprocessing (PCA transform)
    X_train_processed, X_test_processed, preprocessing_objects = pca_transform(
        X_train_aligned, X_test_final
    )

    X_train_final, X_test_final_selected, rf_selector = apply_random_forest_selection(
        X_train_processed, X_test_processed, y_train_aligned
    )

    # Phase 5: Model Training and Evaluation
    point_model = train_gbm_model(X_train_final, y_train_aligned)

    X_train_cp, X_cal_cp, y_train_cp, y_cal_cp = engine_aware_split(
        df_train_windowed, X_train_final, y_train_aligned
    )

    point_results = evaluate_point_predictions(point_model, X_test_final_selected, y_test_final)

    # Phase 6: Conformalized Quantile Regression
    cqr = ConformalizedQuantileRegressor(alpha=Config.ALPHA)
    cqr.fit_and_calibrate(X_train_cp, y_train_cp, X_cal_cp, y_cal_cp)

    y_lower, y_upper = cqr.predict_intervals(X_test_final_selected)
    interval_results = evaluate_prediction_intervals(y_test_final, y_lower, y_upper)

    # Phase 7: Results Summary (save to results/<subset>/)
    results_df = pd.DataFrame({
        "Engine_ID": [int(idx.split("_")[0]) for idx in X_test_final_selected.index],
        "True_RUL": y_test_final.values,
        "Predicted_RUL": point_results["predictions"],
        "Lower_Bound": y_lower,
        "Upper_Bound": y_upper,
        "Interval_Width": (y_upper - y_lower),
        "Covered": ((y_test_final.values >= y_lower) & (y_test_final.values <= y_upper)).astype(int),
        "Absolute_Error": np.abs(y_test_final.values - point_results["predictions"]),
    }).round(3)

    n_engines = results_df["Engine_ID"].nunique()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = Config.SUBSET.lower()
    results_dir = Path("results") / prefix
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{prefix}_rul_predictions_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"FINAL EVALUATION METRICS ({Config.SUBSET})")
    print("=" * 80)
    print("Point Prediction Metrics:")
    print(f"  MAE: {point_results['mae']:.4f}")
    print(f"  RMSE: {point_results['rmse']:.4f}")
    print(f"  R²: {point_results['r2']:.4f}")
    print(f"  S-Score: {point_results['s_score']:.1f}")
    print("\nInterval Prediction Metrics:")
    print(f"  Coverage: {interval_results['coverage']:.3f}")
    print(f"  MPIW: {interval_results['mpiw']:.3f}")
    print(f"  NMPIW: {interval_results['nmpiw']:.3f}")
    print("=" * 80)
    # Save minimal eval metrics (one-row CSV)
    n_engines = results_df["Engine_ID"].nunique()

    metrics_row = {
        "subset": Config.SUBSET,
        "n_test_engines": int(n_engines),
        "mae": float(point_results["mae"]),
        "rmse": float(point_results["rmse"]),
        "r2": float(point_results["r2"]),
        "s_score": float(point_results["s_score"]),
        "coverage": float(interval_results["coverage"]),
        "mpiw": float(interval_results["mpiw"]),
        "nmpiw": float(interval_results["nmpiw"]),
    }

    metrics_path = results_dir / f"{prefix}_eval_metrics.csv"
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    print(f"Saved eval metrics CSV: {metrics_path}")

    # Save fitted artifacts to results/<subset>/
    joblib.dump(point_model, results_dir / f"{prefix}_gbm_point_model.pkl")
    joblib.dump(cqr, results_dir / f"{prefix}_cqr_interval_model.pkl")
    joblib.dump(preprocessing_objects, results_dir / f"preprocessing_pipeline_{prefix}.pkl")
    joblib.dump(kmeans_model, results_dir / f"kmeans_regimes_{prefix}.pkl")

    return {
        "subset": Config.SUBSET,
        "results_dir": results_dir,
        "point_results": point_results,
        "interval_results": interval_results,
        "results_table": results_df,
        "models": {"point_model": point_model, "cqr": cqr},
        "preprocessing_objects": preprocessing_objects,
        "rf_selector": rf_selector,
        "intermediate_data": {
            "df_test_windowed": df_test_windowed,
            "X_test": X_test,
            "df_test_final": df_test_final
        }
    }


if __name__ == "__main__":
    import argparse

    def _set_subset_in_config(subset: str) -> None:

        Config.SUBSET = subset

        Config.TRAIN_FILE = os.path.join(Config.DATA_DIR, f"train_{Config.SUBSET}_cleaned.csv")
        Config.TEST_FILE  = os.path.join(Config.DATA_DIR, f"test_{Config.SUBSET}_cleaned.csv")
        Config.RUL_FILE   = os.path.join(Config.DATA_DIR, f"rul_{Config.SUBSET}.csv")

        Config.FFT_COEFF_MAX = 10 if Config.SUBSET == "FD004" else 3

        # Asymmetric miscoverage ratio per your spec
        Config.CQR_OVER_TO_UNDER_RATIO = 3.0

        Config.validate()

    parser = argparse.ArgumentParser(description="Run CMAPSS RUL + CQR pipeline")
    parser.add_argument(
        "--subset",
        type=str,
        choices=["FD002", "FD004"],
        default=Config.SUBSET,
        help="CMAPSS subset to run",
    )
    args = parser.parse_args()

    _set_subset_in_config(args.subset)

    _ = run_complete_pipeline()




