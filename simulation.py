import os
import simpy
import auxiliaries as aux

from tqdm import tqdm
from scipy.stats import expon


class Station:

    def __init__(self, process_time: float, station_name: str, path_events: str):
        self.pt: float = process_time
        self.name = station_name
        self.path_events = path_events

        self.buffer_get: simpy.Container
        self.buffer_put: simpy.Container
        self.finished_jobs: int = 0
        self.break_down: bool = False
        self.state = 'init'

    def _change_state(self, new_state: str):
        state_dict = {
            'active': 0,
            'starved': 1,
            'blocked': 2,
            'breakdown': 3,
        }
        self.state = state_dict[new_state]

    def run_station(self, env: simpy.Environment):
        while True:
            # change state: waiting for material
            self._change_state('starved')
            # get material
            yield self.buffer_get.get(1)
            # change state: running job
            self._change_state('active')
            # time out while running
            aux.write_new_row(path=self.path_events, row=[
                              round(env.now, 3), self.name[1:], self.finished_jobs+1, 'job start'])
            yield env.timeout(self._apply_var(self.pt))
            aux.write_new_row(path=self.path_events, row=[
                              round(env.now, 3), self.name[1:], self.finished_jobs+1, 'job finish'])
            self.finished_jobs += 1
            # change state: returning material
            self._change_state('blocked')
            # return material
            yield self.buffer_put.put(1)

    def _apply_var(self, pt: float) -> float:
        # return max(0, random.gauss(pt, pt*STD_DEV))
        return max(0, expon.rvs(scale=pt, loc=pt, size=1)[0])


class Factory:

    def __init__(self,
                 process_times: 'list[int]',
                 path_events: str,
                 path_buffer: str,
                 save_results: bool,
                 capa_init: int,
                 capa_max: int,
                 capa_inf: int,
                 ):

        self.env = simpy.Environment()
        self.process_times = process_times
        self.path_events = path_events
        self.path_buffer = path_buffer
        self.save_results = save_results
        self.capa_init = capa_init
        self.capa_max = capa_max
        self.capa_inf = capa_inf
        self.num_stations = len(process_times)
        self.num_buffers = self.num_stations + 1

        self.station_names = [f'S{i}' for i in range(self.num_stations)]
        self.buffer_names = [f'B{i}' for i in range(self.num_buffers)]
        self.stations = self._get_stations()
        self.buffers = self._get_buffers()

        self._update_stations()

    def _update_stations(self):
        for n in range(self.num_stations):
            self.stations[self.station_names[n]
                          ].buffer_get = self.buffers[self.buffer_names[n]]
            self.stations[self.station_names[n]
                          ].buffer_put = self.buffers[self.buffer_names[n+1]]

    def _buffer_selector(self, n):
        if n == 0:
            return simpy.Container(  # first buffer
                env=self.env, capacity=float('inf'), init=self.capa_inf)
        elif n == self.num_stations:
            return simpy.Container(  # last buffer
                env=self.env, capacity=float('inf'), init=0)
        else:
            return simpy.Container(
                env=self.env, capacity=self.capa_max, init=self.capa_init)

    def _get_buffers(self) -> 'dict[str, simpy.Container]':
        return {buffer_name: self._buffer_selector(int(buffer_name[1:])) for buffer_name in self.buffer_names}

    def _get_stations(self) -> 'dict[str, Station]':
        return {
            station_name: Station(
                process_time=pt,
                station_name=station_name,
                path_events=self.path_events,
            ) for (station_name, pt) in zip(self.station_names, self.process_times)
        }

    def get_buffer_levels(self) -> 'list[int]':
        return [buffer.level for buffer in self.buffers.values()]

    def get_station_states(self) -> 'list[str]':
        return [station.state for station in self.stations.values()]

    def restock_customer(self, capa_inf: int) -> None:
        curr_stock = self.buffers[self.buffer_names[0]].level
        if curr_stock < capa_inf:
            self.buffers[self.buffer_names[0]].put(capa_inf-curr_stock)


def run_simulation(
    process_times: 'list[float]',
    simulation_time: int = 1000,
    path_buffer: str = 'buffer.csv',
    path_events: str = 'events.csv',
    save_results: bool = True,
    capa_init: int = 0,
    capa_max: int = 10,
    capa_inf: int = int(1e+2),
) -> None:
    '''
    Function to simulate a manufacturing line with fully connected stations. 

    The total number of stations is determined by the length of the provided 
    list of process times. Each station is preceded and followed by a buffer. 
    Hence, there will be only one more buffer than stations. The simulations'
    system boundaries are unlimited in supply and demand. 

    Parameters
    ----------
    process_time : list[float]
        Process times for the created stations. Will determine the total number
        of stations and buffers in the simulation. 
    simulation_time : int, default 1000
        Number of steps that the simulation will run for. The progression of 
        the simulation is displayed using a simple progress bar. 
    path_buffer : str, default "buffer.csv"
        File (and path) name to store the buffer values of the simulation run. 
    path_events : str, default "events.csv"
        File (and path) name to store every event of the simulation run. 
    save_results : bool, default True
        Parameter to execute the simulation without storing the buffer and 
        event results in a separate csv file. 
    capa_init : int, default 0
        Initial capacity of the simpy.Container that act as buffers in the 
        simulation. Does not interfere with the later AP calculation, but helps
        to reduce the time for the system to be swung in. 
    capa_max : int, default 10
        Maximum capacity of the simpy.Container that will cause stations to
        be blocked if they cannot put products/jobs into the buffer of the
        following station. Affects the bottleneck situation to a high degree.
    capa_inf : int, default 100
        Capacity level of the first station in the manufacturing line. Will 
        be refilled once per time step to simulate a virtually unlimited 
        supplier. This ensures that bottlenecks occur only due to throughput 
        restrictions between stations, and not due to insufficient supply. 
    '''

    # initialize factory
    factory = Factory(
        process_times=process_times,
        path_buffer=path_buffer,
        path_events=path_events,
        save_results=save_results,
        capa_init=capa_init,
        capa_max=capa_max,
        capa_inf=capa_inf
    )

    # run stations
    for station in factory.stations.values():
        factory.env.process(station.run_station(factory.env))

    # save?
    if save_results:
        # prevent overwriting
        if not os.path.exists(path_buffer):
            aux.write_new_row(
                path_buffer, ['t'] + factory.buffer_names)
        # prevent overwriting
        if not os.path.exists(path_events):
            aux.write_new_row(
                path_events, ['t', 'num_station', 'num_job', 'event_type'])

    # run simulation
    print(
        f'Running simulation with {len(process_times)} stations for {simulation_time} steps. ')
    for t in tqdm(range(1, simulation_time)):
        # iterate one simulation step
        factory.env.run(until=t)
        # save current buffer levels to file
        # note: events are written as they occur by each Station
        _new_row = [t] + factory.get_buffer_levels()
        aux.write_new_row(path_buffer, _new_row)
        # reset level of 'B0' to 'capa_inf'
        factory.restock_customer(capa_inf)
