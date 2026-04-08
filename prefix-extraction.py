import pandas as pd
from feature_engineering import (
    detect_static_attributes, add_case_attributes, add_event_attributes,
    add_time_features, add_activity_counts, add_activity_start_times,
    add_time_based_features, add_prefix_stats, add_resource_features,
    add_holiday_features, add_pairwise_delays
)


def extract_prefixes_with_time_cutoff(
    event_log,
    cutoff_in_days,
    target_activity,
    holidays,
    case_id_col='case:concept:name',
    activity_col='concept:name',
    timestamp_col='time:timestamp',
    timezone='UTC'
):
    all_prefixes = []
    static_attributes, dynamic_attributes = detect_static_attributes(event_log, case_id_col)
    exclude = {timestamp_col, 'relative_case_time', 'relative_log_time', 'time_since_prev'}
    static_attributes = [a for a in static_attributes if a not in exclude]
    dynamic_attributes = [a for a in dynamic_attributes if a not in exclude]
    reference_date = event_log[timestamp_col].min()
    activities_to_flag = event_log[activity_col].unique()

    event_log = event_log.sort_values([case_id_col, timestamp_col])

    for case, group in event_log.groupby(case_id_col):
        prefix = group[group['relative_case_time'] < cutoff_in_days]
        if group['relative_case_time'].max() < cutoff_in_days:
            continue
        target_events = group[group[activity_col] == target_activity]
        if len(target_events) == 0:
            continue
        prefix_start_time = group.iloc[0][timestamp_col]
        prefix_end_time = group.iloc[-1][timestamp_col]
        target_value = target_events.iloc[0]['relative_log_time']

        features = {}
        add_case_attributes(prefix, features, static_attributes)
        add_event_attributes(prefix, features, dynamic_attributes)
        add_time_features(prefix, features)
        add_activity_counts(prefix, features, activities_to_flag, activity_col)
        add_activity_start_times(prefix, features, reference_date, activities_to_flag, activity_col, timestamp_col)
        add_time_based_features(prefix, features, prefix_start_time, timestamp_col)
        add_prefix_stats(prefix, features, case)
        add_resource_features(prefix, features)
        add_holiday_features(prefix, features, prefix_end_time, holidays, timestamp_col, timezone)
        features['target'] = target_value
        all_prefixes.append(features)

    return pd.DataFrame(all_prefixes)


def extract_prefixes_fixed_length(
    event_log,
    prefix_length,
    target_activity,
    holidays,
    case_id_col='case:concept:name',
    activity_col='concept:name',
    timestamp_col='time:timestamp',
    timezone='UTC',
    activity_pairs=None
):
    all_prefixes = []
    static_attributes, dynamic_attributes = detect_static_attributes(event_log, case_id_col)
    exclude = {timestamp_col, 'relative_case_time', 'relative_log_time', 'time_since_prev'}
    static_attributes = [a for a in static_attributes if a not in exclude]
    dynamic_attributes = [a for a in dynamic_attributes if a not in exclude]

    event_log = event_log.sort_values([case_id_col, timestamp_col])
    activities_to_flag = event_log[activity_col].unique()
    reference_date = event_log[timestamp_col].min()

    for case_id, group in event_log.groupby(case_id_col):
        case_start_time = group.iloc[0][timestamp_col]
        target_events = group[group[activity_col] == target_activity]
        if len(target_events) == 0:
            continue
        target_time = target_events.iloc[0][timestamp_col]

        for i in range(1, prefix_length + 1):
            if len(group) < i:
                continue
            prefix = group.iloc[:i].copy()
            prefix_end_time = prefix.iloc[-1][timestamp_col]
            if target_time <= prefix_end_time:
                continue

            features = {}
            add_case_attributes(prefix, features, static_attributes)
            add_event_attributes(prefix, features, dynamic_attributes)
            add_time_features(prefix, features)
            add_activity_counts(prefix, features, activities_to_flag, activity_col)
            add_activity_start_times(prefix, features, reference_date, activities_to_flag, activity_col, timestamp_col)
            add_time_based_features(prefix, features, case_start_time, timestamp_col)
            add_prefix_stats(prefix, features, case_id)
            add_resource_features(prefix, features)
            add_holiday_features(prefix, features, prefix_end_time, holidays, timestamp_col, timezone)
            if activity_pairs:
                add_pairwise_delays(prefix, features, activity_pairs, activity_col, timestamp_col)
            features['target'] = target_time
            all_prefixes.append(features)

    return pd.DataFrame(all_prefixes)