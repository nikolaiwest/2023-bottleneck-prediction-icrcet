# simple function to display the simulation data 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

def visualize_sim_data( 
    scenario_name : str, 
    file_path : str, 
    t_start : int, 
    t_end : int,
    i ):

    # get active periods
    ap = pd.read_csv(f"{file_path}active_periods_{scenario_name}.csv", index_col=0)
    ap = ap[t_start:t_end]
    ap["bottleneck"] = [v[1:] for v in ap["bottleneck"]] # convert column to int
    ap = ap.astype(int)
    ap = ap.groupby(np.arange(len(ap))//i).mean() # aggregate
    ap = ap.astype({'bottleneck':'int'})

    # get buffer levels
    bl = pd.read_csv(f"{file_path}buffer_{scenario_name}.csv", index_col=0)
    bl = bl[t_start:t_end]
    bl = bl.groupby(np.arange(len(bl))//i).mean() # aggregate

    # new fig
    fig, axes = plt.subplots(3)
    fig.set_figheight(9)
    fig.set_figwidth(16)
    fig.suptitle(i)

    # buffer level 
    for buffer in bl.columns.tolist()[:-1]:
        axes[0].plot(range(len(bl[buffer])), bl[buffer], label=buffer)

    # sawtooth
    for station in ap.columns.tolist()[:-1]:
        axes[1].plot(range(len(ap[station])), ap[station], label=station)

    # bottlenecks
    axes[2].plot(ap.index, ap["bottleneck"])

    axes[0].legend()
    axes[1].legend()
    # axes[0].set_yscale("log")

    fig.savefig(fname=f"bottleneck_25%_{i}.png", dpi=300)
