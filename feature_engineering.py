# functions to add to prefix
from pandas.io.stata import is_numeric_dtype
import pandas as pd

# Deprecated: Used before implementing event and case attributes
# def add_attributes(prefix, features, excluded_list=[timestamp_col, "relative_case_time", "time_since_prev"]): # might break because of raw values in ML model
#     for column in prefix.columns:
#         if column not in excluded_list:
#             features[column] = prefix[column].iloc[0] # currently only considers first value, mostly NaNs! --> fix by differentiating between case and event attributes 

def detect_static_attributes(event_log):
    static_attributes = []
    dynamic_attributes = []
    for col in event_log.columns:
        case_value_counts = event_log.groupby("case:concept:name")[col].apply(lambda x: x.dropna().nunique())
        if case_value_counts.max() <= 1:
            static_attributes.append(col)
        else:
            dynamic_attributes.append(col)
    return static_attributes, dynamic_attributes    


def add_case_attributes(prefix, features, case_attributes):
    for attribute in case_attributes:
        if pd.api.types.is_numeric_dtype(prefix[attribute]):
            values = prefix[attribute].dropna()
            features[attribute] = values.iloc[0] if not values.empty else 0
        else:
            values = prefix[attribute].dropna()
            if not values.empty:
                features[attribute] = values.iloc[0]
    return features

# TODO: Categoricals of numeric Time
def add_event_attributes(prefix, features, event_attributes):
    for attribute in event_attributes:
        # if it's a numeric data type
        if pd.api.types.is_numeric_dtype(prefix[attribute]):
            # aggregation encoding
            agg_values = prefix[attribute].agg(['min', 'max'])
            for agg, value in agg_values.items():
                features[f'{attribute}_{agg}'] = value
            # last state encoding
            features[f'{attribute}_last'] = prefix[attribute].dropna().iloc[-1] if not prefix[attribute].dropna().empty else 0
        # if it's a categorical data type
        else:
            # Last-State Encoding 
            features[f'{attribute}_last'] = prefix[attribute].dropna().iloc[-1] if not prefix[attribute].dropna().empty else 0
    return features


def add_attributes_from_last_event(prefix, features):
    last_event = prefix.iloc[-1]
    features['last_activity'] = last_event["concept:name"]
    features['last_resource'] = last_event["org:resource"]

def add_activity_counts(prefix, features, all_activites):
    for activity in all_activites:
        features[f'num_{activity}'] = (prefix["concept:name"] == activity).sum()

def add_activity_start_times(prefix, features, reference_date, all_activities):
    prefix_end_time = prefix.iloc[-1]["time:timestamp"]
    for activity in all_activities:
        if activity not in prefix["concept:name"].values:
            features[f'{activity}_abs_days::start'] = -1
            features[f'{activity}_days_since'] = -1
            continue
        act_time = prefix[prefix["concept:name"] == activity].iloc[0]["time:timestamp"]
        features[f'{activity}_abs_days::start'] = (act_time - reference_date).days
        features[f'{activity}_days_since'] = (prefix_end_time - act_time).days

def add_time_based_features(prefix, features, case_start_time):
    features['start_day_of_week'] = prefix.iloc[0]["time:timestamp"].dayofweek
    features['start_month'] = prefix.iloc[0]["time:timestamp"].month
    features['case_start_year'] = case_start_time.year
    
def add_holiday_features(prefix, features, prefix_end_time, holidays):
    features["num_holidays_in_prefix"] = sum(1 for day in pd.date_range(prefix.iloc[0]["time:timestamp"].tz_convert('Europe/Rome').normalize(),prefix_end_time.tz_convert('Europe/Rome').normalize(),freq='D',ambiguous='NaT')if day.date() in holidays)
    features['days_to_next_holiday'] = next((i for i in range(1, 366) if (prefix_end_time + pd.Timedelta(days=i)).date() in holidays), -1)
    features['days_since_last_holiday'] = next((i for i in range(1, 366) if (prefix_end_time - pd.Timedelta(days=i)).date() in holidays), -1)
    

def add_resource_features(prefix, features):
    features["number_of_unique_resources"] = prefix["org:resource"].nunique()

def add_prefix_stats(prefix, features, case_id):
    features['case_id'] = case_id
    features['prefix_length'] = len(prefix)
    features['avg_time_between_events'] = prefix['time_since_prev'].mean()
    features['max_time_between_events'] = prefix['time_since_prev'].max()

def add_numeric_aggregates(prefix, features, exclude_cols=["time:timestamp"]):
    for col in prefix.select_dtypes(include='number').columns:
        if col not in exclude_cols:
            features[f'{col}_sum'] = prefix[col].sum()
            features[f'{col}_mean'] = prefix[col].mean()

def add_days_since_reference(prefix, features, reference_date):
    features['days_since_reference'] = (prefix.iloc[-1]["time:timestamp"] - reference_date).days

def add_activity_flags(prefix, features, activities_to_flag):
    for activity in activities_to_flag:
        features[f'has_{activity}'] = int(activity in prefix["case:concept"].values)
        
def add_pairwise_delays(prefix, features, activity_pairs):
    for act1, act2, label in activity_pairs:
        if act1 in prefix["case:concept:name"].values and act2 in prefix["case:concept:name"].values:
            t1 = prefix[prefix["case:concept:name"] == act1].iloc[0]["time:timestamp"]
            t2 = prefix[prefix["case:concept:name"] == act2].iloc[0]["time:timestamp"]
            features[f'{label}_delay'] = (t2 - t1).days
        else:
            features[f'{label}_delay'] = -1

def add_time_features(prefix, features):
    features['relative_log_time'] = prefix['relative_log_time'].iloc[-1]
    features['relative_case_time'] = prefix['relative_case_time'].iloc[-1]
    features['time_since_prev'] = prefix['time_since_prev'].iloc[-1]

        