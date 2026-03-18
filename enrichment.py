# event log enrichment functions
def event_add_relative_log_time(event_log):
    event_log['relative_log_time'] = (event_log["time:timestamp"] - event_log["time:timestamp"].min()).dt.days
    return event_log

def add_inter_event_time(event_log, case_id_col='case:concept:name', time_col='time:timestamp'):
    """
    Adds time since previous event for each event in the log.
    First event of each case gets 0.
    """
    event_log['time_since_prev'] = (event_log.groupby(case_id_col)[time_col].transform(lambda x: x.diff())) # calculates time between events
    event_log['time_since_prev'] = event_log['time_since_prev'].dt.days # converts to days
    event_log['time_since_prev'] = event_log['time_since_prev'].fillna(0)  # first event of each case: 0
    return event_log

def event_add_relative_case_time(event_log):
    event_log['relative_case_time'] = event_log.groupby("case:concept:name")["time:timestamp"].transform(lambda x: (x - x.min()).dt.days)
    return event_log

def remove_empty_columns(event_log):
    columns = list(event_log.columns)
    keep = []
    for col in columns:
        if event_log[col].dropna().count() != 0:
            keep.append(col)
    return event_log[keep]