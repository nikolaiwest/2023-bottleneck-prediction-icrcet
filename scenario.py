import os
import datetime as dt

from simulation import run_simulation
from auxiliaries import (
    calc_active_periods,
    get_events,
    get_buffer,
    get_active_periods,
)


def run_scenario(
    process_times: 'list[float]',
    simulation_time: int,
    path_buffer: str,
    path_events: str,
    path_active_periods: str,
    save_results: bool,
    capa_init: int,
    capa_max: int,
    capa_inf: int,
) -> None:

    print(f'{dt.datetime.now().strftime("%H:%M:%S")}: Running: \n{scenario_1}')

    # check if simulation results exit
    if not os.path.exists(path_buffer) or not os.path.exists(path_events):
        # if not, run simulation and save data
        run_simulation(
            process_times=process_times,
            simulation_time=simulation_time,
            path_events=path_events,
            path_buffer=path_buffer,
            save_results=save_results,
            capa_init=capa_init,
            capa_max=capa_max,
            capa_inf=capa_inf
        )
    # get buffer and events
    events = get_events(path=path_events)
    buffer = get_buffer(path=path_buffer)

    # check if ap-file exits
    if not os.path.exists(path_active_periods):
        # if not, calculate and save the active periods
        active_periods = calc_active_periods(events)
        active_periods.to_csv(path_active_periods)
    else:
        # get ap
        active_periods = get_active_periods(path_active_periods)

    print(f'{dt.datetime.now().strftime("%H:%M:%S")}: Finished!')
    return events, buffer, active_periods



scenario_1 = {
    'process_times':  [2, 5, 2, 5, 2],
    'simulation_time': 10000,
    'path_buffer': 'buffer_10k.csv',
    'path_events': 'events_10k.csv',
    'path_active_periods': 'active_periods_10k.csv',
    'save_results': True,
    'capa_init': 1,
    'capa_max': 5,
    'capa_inf': 5,
}

events_s1, buffer_s1, active_periods_s1 = run_scenario(**scenario_1)
