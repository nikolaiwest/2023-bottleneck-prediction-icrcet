import numpy as np
import pandas as pd
import csv


def write_new_row(path: str, row: list) -> None:
    with open(path, 'a+', newline='') as f:
        csv.writer(f).writerow(row)


def station_remains_active(events, num_station, num_job) -> bool:
    time_finish_last_job = events[
        (events['num_station'] == num_station) &
        (events['num_job'] == num_job) &
        (events['event_type'] == 'job finish')
    ]['t'].values
    time_start_next_job = events[
        (events['num_station'] == num_station) &
        (events['num_job'] == num_job + 1) &
        (events['event_type'] == 'job start')
    ]['t'].values
    return time_finish_last_job == time_start_next_job


def get_active_periods(path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def get_buffer(path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_events(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['status'] = 'init'

    for index, event in df.iterrows():
        # set 'status' to 'active' if a new job was started
        if event['event_type'] == 'job start' and event['num_job'] != 1:
            df.loc[index, 'status'] = 'active'
        elif event['event_type'] == 'job finish':
            _check_status = station_remains_active(
                df, event['num_station'], event['num_job'])
            if _check_status.size > 0 and _check_status:
                df.loc[index, 'status'] = 'active'
            else:
                df.loc[index, 'status'] = 'passive'
        else:
            # remain init for 'num_job' == 1
            pass
    return df


def _get_event_length(events: pd.DataFrame) -> int:
    t_max = max(events['t'])
    if t_max % 1 == 0:
        return int(t_max)
    else:
        return int(t_max) + 1


def _get_event_stations(events: pd.DataFrame, how: str) -> list:
    if how == 'name':
        return [f'S{n}' for n in events['num_station'].unique().tolist()]
    elif how == 'num':
        return [n for n in events['num_station'].unique().tolist()]
    else:
        raise ValueError(f'{how} not accepted')


def _get_t_of_first_active(events: pd.DataFrame, step: int, station: int) -> float:
    subset = events[(events['num_station'] == station) & (events['t'] <= step)]
    t_first_active = 0
    for _, step in subset[::-1].iterrows():
        if step['status'] == 'passive' or step['status'] == 'init':
            break
        else:
            t_first_active = step['t']
    return t_first_active


def calc_active_periods(events: pd.DataFrame) -> pd.DataFrame:
    events_length = _get_event_length(events)
    station_names = _get_event_stations(events, how='name')
    station_nums = _get_event_stations(events, how='num')
    # create new resut df
    active_periods = pd.DataFrame(
        np.nan,
        index=range(events_length),
        columns=station_names)
    # iter over results and the time of the first active state
    print(
        f'Calculating the active periods for all {len(station_names)} stations.')
    for index, _ in active_periods.iterrows():
        #print(f"- getting period resets: {index}/{events_length}", flush=True)
        for (station_name, station_num) in zip(station_names, station_nums):
            active_periods.loc[index, station_name] = _get_t_of_first_active(
                events, index, station_num)
            
    # repeat iteration to update active periods for all timesteps
    for index, _ in active_periods.iterrows():
        #print(f"- getting remaining time: {index}/{events_length}", flush=True)
        for (station_name, station_num) in zip(station_names, station_nums):
            if active_periods.loc[index, station_name] != 0:
                active_periods.loc[index, station_name] = index - \
                    active_periods.loc[index, station_name]
            elif active_periods.loc[index, station_name] == 0 and index > 0:
                active_periods.loc[index,
                                   station_name] = active_periods.loc[index-1, station_name] + 1
            else:
                active_periods.loc[index, station_name] = 0
    # determine bottleneck station
    active_periods['bottleneck'] = active_periods.idxmax(axis=1)
    return active_periods
#%%

