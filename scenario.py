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
    scenario_name : str, 
    process_times : 'list[float]',
    simulation_time : int,
    path : str,
    save_results: bool,
    verbose: bool,
    capa_init: int,
    capa_max: int,
    capa_inf: int,
) -> None:
    if verbose:
        print(f'{dt.datetime.now().strftime("%H:%M:%S")}: Now running: \n{scenario_name}')

    path_events = f"{path}events{scenario_name}.csv"
    path_buffer = f"{path}buffer{scenario_name}.csv"
    path_active_periods = f"{path}active_periods{scenario_name}.csv"

    # check if simulation results exit already
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

    # check if ap-file exits already
    if not os.path.exists(path_active_periods):
        # if not, calculate and save the active periods
        active_periods = calc_active_periods(events)
        if save_results:
            active_periods.to_csv(path_active_periods)
    else:
        # get ap
        active_periods = get_active_periods(path_active_periods)

    if verbose:
        print(f'{dt.datetime.now().strftime("%H:%M:%S")}: Finished scenario {scenario_name}.')
    return events, buffer, active_periods


def _run(i): 
    if i < 10:
        num = f"00{i}"
    elif i <100:
        num = f"0{i}"
    else:
        num = str(i)
    scenario = {
        'scenario_name': f"_10k_S2-S4+25%_{num}",
        'process_times':  [2, 2.25, 2, 2.25, 2],
        'simulation_time': 10000,
        'path' : 'data/',
        'save_results': True,
        'verbose' : False,
        'capa_init': 1,
        'capa_max': 5,
        'capa_inf': 5,
    }
    events, buffer, active_periods = run_scenario(**scenario)


from joblib import Parallel, delayed
Parallel(n_jobs=12)(delayed(_run)(i) for i in range(100, 112))