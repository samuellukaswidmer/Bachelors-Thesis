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
        available = sorted(set(cols) & set(X_train.columns))
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
        available = sorted(set(cols) & set(X_train.columns))
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

def run_group_only_analysis(X_train, X_test, y_train, y_test, group_prefix):
    
    group_cols = [
        col for col in X_train.columns 
        if col.startswith(f'{group_prefix}::')
    ]
    print(f"{group_prefix} columns:", group_cols)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[group_cols], y_train)
    
    importances = pd.Series(model.feature_importances_, index=group_cols)
    print(importances.sort_values(ascending=False))
    
    y_pred = model.predict(X_test[group_cols])
    print(f"MAE {group_prefix} only:", mean_absolute_error(y_test, y_pred))
    
    return model, importances

def run_single_group_removal(X_train, X_test, y_train, y_test, group_prefix, exclude=None):
    if exclude is None:
        exclude = []
    group_cols = [col for col in X_train.columns 
                  if col.startswith(f'{group_prefix}::') and col not in exclude]
    
    variants = {'Full pipeline': list(X_train.columns)}
    for col in group_cols:
        variants[f'Without {col}'] = [c for c in X_train.columns if c != col]
    
    results, importances = run_ablation_with_importance(X_train, X_test, y_train, y_test, variants)
    return results, importances

def create_case_log(event_log):
    cases = event_log.groupby("case:concept:name")
    case_log = cases.agg(
        start_time=("time:timestamp", 'first'),
        end_time=("time:timestamp", 'last'),
        no_of_events=("concept:name", 'count'))
    case_log['duration'] = case_log['end_time'] - case_log['start_time']
    return case_log

def case_add_activity_start_times(case_log, event_log, case_id_col='case:concept:name', activity_col='concept:name', time_col="time:timestamp"):
    cases = event_log.groupby([case_id_col,activity_col])
    activity_starts = cases[time_col].first()
    print(activity_starts)
    activities = event_log[activity_col].unique()
    for activity in activities:
        col_name = convert_name(activity) + '::start'
        case_log[col_name] = activity_starts.xs(activity, level=1, axis=0)
    return case_log 
    
def convert_name(name):
    return '_'.join(name.split(' '))   