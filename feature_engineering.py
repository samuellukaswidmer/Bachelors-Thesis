import pandas as pd
from pandas.io.stata import is_numeric_dtype

def detect_static_attributes(event_log, case_id_col='case:concept:name'):
    static_attributes = []
    dynamic_attributes = []
    for col in event_log.columns:
        case_value_counts = event_log.groupby(case_id_col)[col].apply(lambda x: x.dropna().nunique())
        if case_value_counts.max() <= 1:
            static_attributes.append(col)
        else:
            dynamic_attributes.append(col)
    return static_attributes, dynamic_attributes


def add_case_attributes(prefix, features, case_attributes):
    for attribute in case_attributes:
        if pd.api.types.is_numeric_dtype(prefix[attribute]):
            values = prefix[attribute].dropna()
            features[f"case::{attribute}"] = values.iloc[0] if not values.empty else 0
        else:
            values = prefix[attribute].dropna()
            if not values.empty:
                features[f"case::{attribute}"] = values.iloc[0]
    return features


def add_event_attributes(prefix, features, event_attributes):
    for attribute in event_attributes:
        if pd.api.types.is_numeric_dtype(prefix[attribute]):
            agg_values = prefix[attribute].agg(['min', 'max'])
            for agg, value in agg_values.items():
                features[f'event::{attribute}_{agg}'] = value
            features[f'event::{attribute}_last'] = prefix[attribute].dropna().iloc[-1] if not prefix[attribute].dropna().empty else 0
        else:
            features[f'event::{attribute}_last'] = prefix[attribute].dropna().iloc[-1] if not prefix[attribute].dropna().empty else 0
    return features


def add_attributes_from_last_event(prefix, features, activity_col, timestamp_col):
    last_event = prefix.iloc[-1]
    features['last_activity'] = last_event[activity_col]
    features['last_resource'] = last_event["org:resource"]


def add_activity_counts(prefix, features, all_activities, activity_col):
    for activity in all_activities:
        features[f'activity::num_{activity}'] = (prefix[activity_col] == activity).sum()


def add_activity_start_times(prefix, features, reference_date, all_activities, activity_col, timestamp_col):
    prefix_end_time = prefix.iloc[-1][timestamp_col]
    for activity in all_activities:
        if activity not in prefix[activity_col].values:
            features[f'activity::{activity}_abs_days::start'] = -1
            features[f'activity::{activity}_days_since'] = -1
            continue
        act_time = prefix[prefix[activity_col] == activity].iloc[0][timestamp_col]
        features[f'activity::{activity}_abs_days::start'] = (act_time - reference_date).days
        features[f'activity::{activity}_days_since'] = (prefix_end_time - act_time).days


def add_time_based_features(prefix, features, case_start_time, timestamp_col):
    features['time::start_day_of_week'] = prefix.iloc[0][timestamp_col].dayofweek
    features['time::start_month'] = prefix.iloc[0][timestamp_col].month
    features['time::case_start_year'] = case_start_time.year


def add_holiday_features(prefix, features, prefix_end_time, holidays, timestamp_col, timezone='Europe/Rome'):
    features["system::num_holidays_in_prefix"] = sum(
        1 for day in pd.date_range(
            prefix.iloc[0][timestamp_col].tz_convert(timezone).normalize(),
            prefix_end_time.tz_convert(timezone).normalize(),
            freq='D',
            ambiguous='NaT'
        ) if day.date() in holidays
    )
    features['system::days_to_next_holiday'] = next(
        (i for i in range(1, 366) if (prefix_end_time + pd.Timedelta(days=i)).date() in holidays), -1
    )
    features['system::days_since_last_holiday'] = next(
        (i for i in range(1, 366) if (prefix_end_time - pd.Timedelta(days=i)).date() in holidays), -1
    )


def add_resource_features(prefix, features):
    features["system::number_of_unique_resources"] = prefix["org:resource"].nunique()


def add_prefix_stats(prefix, features, case_id):
    features['meta::case_id'] = case_id
    features['meta::prefix_length'] = len(prefix)
    features['time::avg_time_between_events'] = prefix['time_since_prev'].mean()
    features['time::max_time_between_events'] = prefix['time_since_prev'].max()


def add_numeric_aggregates(prefix, features, timestamp_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = [timestamp_col]
    for col in prefix.select_dtypes(include='number').columns:
        if col not in exclude_cols:
            features[f'{col}_sum'] = prefix[col].sum()
            features[f'{col}_mean'] = prefix[col].mean()


def add_days_since_reference(prefix, features, reference_date, timestamp_col):
    features['days_since_reference'] = (prefix.iloc[-1][timestamp_col] - reference_date).days


def add_activity_flags(prefix, features, activities_to_flag, activity_col):
    for activity in activities_to_flag:
        features[f'has_{activity}'] = int(activity in prefix[activity_col].values)


def add_pairwise_delays(prefix, features, activity_pairs, activity_col, timestamp_col):
    for act1, act2, label in activity_pairs:
        if act1 in prefix[activity_col].values and act2 in prefix[activity_col].values:
            t1 = prefix[prefix[activity_col] == act1].iloc[0][timestamp_col]
            t2 = prefix[prefix[activity_col] == act2].iloc[0][timestamp_col]
            features[f'time::{label}_delay'] = (t2 - t1).days
        else:
            features[f'time::{label}_delay'] = -1


def add_time_features(prefix, features):
    features['time::relative_log_time'] = prefix['relative_log_time'].iloc[-1]
    features['time::relative_case_time'] = prefix['relative_case_time'].iloc[-1]
    features['time::time_since_prev'] = prefix['time_since_prev'].iloc[-1]

def add_duration_vs_period_avg(prefix_log_train, prefix_log_test):
    year_month_groups = prefix_log_train.groupby(['time::case_start_year', 'time::start_month'])['time::relative_case_time']
    prefix_log_train['system::duration_vs_period_avg'] = prefix_log_train['time::relative_case_time'] - year_month_groups.transform('mean')

    year_month_avg = prefix_log_train.groupby(['time::case_start_year', 'time::start_month'])['time::relative_case_time'].mean()
    prefix_log_test['system::duration_vs_period_avg'] = prefix_log_test.apply(
        lambda row: row['time::relative_case_time'] - year_month_avg.get(
            (row['time::case_start_year'], row['time::start_month']),
            row['time::relative_case_time']
        ),
        axis=1
    )
    return prefix_log_train, prefix_log_test