import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_column_groups(X_train):
    case_cols = [col for col in X_train.columns if col.startswith('case::')]
    event_cols = [col for col in X_train.columns if col.startswith('event::')]
    activity_cols = [col for col in X_train.columns if col.startswith('activity::')]
    time_cols = [col for col in X_train.columns if col.startswith('time::')]
    system_cols = [col for col in X_train.columns if col.startswith('system::')]
    meta_cols = [col for col in X_train.columns if col.startswith('meta::')]
    return case_cols, event_cols, activity_cols, time_cols, system_cols, meta_cols


def get_ablation_variants(X_train):
    case_cols, event_cols, activity_cols, time_cols, system_cols, meta_cols = get_column_groups(X_train)

    activity_count_cols = [col for col in activity_cols if col.startswith('activity::num_')]
    activity_start_cols = [col for col in activity_cols if not col.startswith('activity::num_')]
    holiday_cols = [col for col in system_cols if 'holiday' in col]
    resource_cols = [col for col in system_cols if 'resource' in col]

    raw_sequence = activity_count_cols + [col for col in meta_cols if 'prefix_length' in col]
    case_level = raw_sequence + case_cols
    event_level = case_level + event_cols + activity_start_cols + time_cols
    system_level = event_level + system_cols

    high_level_variants = {
        'Raw sequence': raw_sequence,
        '+ Case-level features': case_level,
        '+ Event-level features': event_level,
        '+ System-level features': system_level,
        'Full pipeline': list(X_train.columns)
    }

    granular_variants = {
        'Raw sequence': raw_sequence,
        '+ Case attributes': case_level,
        '+ Event aggregations': case_level + event_cols,
        '+ Activity start times': case_level + event_cols + activity_start_cols,
        '+ Temporal features': event_level,
        '+ System features': system_level,
        'Full pipeline': list(X_train.columns)
    }

    return high_level_variants, granular_variants


def run_ablation(X_train, X_test, y_train, y_test, variants):
    results = []
    for name, cols in variants.items():
        available = list(dict.fromkeys([c for c in cols if c in X_train.columns]))
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[available], y_train)
        y_pred = model.predict(X_test[available])
        results.append({
            'Variant': name,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'N Features': len(available)
        })
    return pd.DataFrame(results).round(3)

def run_ablation_with_importance(X_train, X_test, y_train, y_test, variants):
    results = []
    importances = {}
    for name, cols in variants.items():
        available = list(dict.fromkeys([c for c in cols if c in X_train.columns]))
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train[available], y_train)
        y_pred = model.predict(X_test[available])
        results.append({
            'Variant': name,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred),
            'N Features': len(available)
        })
        importances[name] = pd.Series(
            model.feature_importances_, index=available
        ).sort_values(ascending=False).head(10)
    return pd.DataFrame(results).round(3), importances


def get_targeted_variants(X_train, primary_activity_col):
    all_cols = list(X_train.columns)
    return {
        'Full pipeline': all_cols,
        'Without relative_log_time': [c for c in all_cols if c != 'time::relative_log_time'],
        f'Without {primary_activity_col}': [c for c in all_cols if c != primary_activity_col],
        'Without both': [c for c in all_cols if c not in ['time::relative_log_time', primary_activity_col]],
    }