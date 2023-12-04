from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from renewableopt.optimal_design import (
    DispatchData,
    MultiPeriodModel,
    visualize,
)
from renewableopt.peak_id import as_datetime, identify_worst_days, timedelta


def optimization_args(time, peak_data):
    load = peak_data.load
    scenario = next(iter(load.keys()))
    time_one_day = time[:len(load[scenario])]
    return time_one_day, load, peak_data.gen_pu

def run_scenario(generation, load, costs):
    time = np.array(generation.index)
    sources = list(generation.columns)
    generation_pu = np.array(generation)
    load_arr = np.array(load)


    model = MultiPeriodModel(
        initial_battery_charge=0.5,
        depth_of_discharge=0.1,
        cost_battery_energy=costs.loc["Battery energy cost (K/MWh)"],
        cost_battery_power=costs.loc["Battery power cost (K/MW)"],
        cost_generation=[
            costs.loc["Solar (K/MW)"],
            costs.loc["Wind cost (K/MW)"],
            costs.loc["Geothermal cost (K/MW)"]
        ])

    for method in ["manual_cluster", "kmeans_cluster"]:
        peak_data = identify_worst_days(time, load_arr, generation_pu,
                                      sources=sources, method=method)
        res = model.minimize_cost(*optimization_args(time, peak_data))
        dispatch = DispatchData.from_greedy(time, load_arr, generation_pu, sources, res, peak_data)
        if dispatch.feasible:
            return method, dispatch
    raise RuntimeError("No cluster method was feasible!!! This shouldn't be possible!")



def cluster_method_name(raw):
    return {
        "manual_cluster": "Solar only",
        "kmeans_cluster": "KMeans (Wind and solar)"
    }[raw]


def run_sensitivity(generation, loads, costs):
    results = {}
    for load_scenario, cost_scenario in product(loads.columns, costs.columns):
        print(load_scenario, cost_scenario)  # noqa
        cluster_method, dispatch = run_scenario(generation, loads[load_scenario], costs[cost_scenario])
        results[(load_scenario, cost_scenario)] = (cluster_method, dispatch)
    return results



def init_data(generation):
    sources = list(generation.columns)

    data = {
        "Load Scenario": [],
        "Cost Scenario": [],
        "Battery Energy Installed (MWh)": [],
        "Battery Power Installed (MW)": [],
    }
    for s in sources:
        data[f"{s.capitalize()} Installed (MW)"] = []
    data["Hours of storage"] = []
    data["Total cost of Capacity (Million USD)"] = []
    data["Load served over year (GWh)"] = []
    data["Energy cost ($/KWh)"] = []
    data["Peak Date"] = []
    data["Clustering Method"] = []
    return data

def extract_data(solutions, load_scen, cost_scen, generation, loads):
    sources = list(generation.columns)
    cluster_method, dispatch = solutions[(load_scen, cost_scen)]
    result = dispatch.result
    E_max = result.E_max
    P_batt = result.P_battery
    dt = timedelta(np.array(generation.index))
    load_served = loads[load_scen].sum() * dt
    peak_day = find_peak_day(dispatch)
    return {
        "Load Scenario": load_scen,
        "Cost Scenario": cost_scen,
        "Battery Energy Installed (MWh)": E_max,
        "Battery Power Installed (MW)": P_batt,
        **{
            f"{s.capitalize()} Installed (MW)": result.P_generation[i]
            for i, s in enumerate(sources)
        },
        "Hours of storage": E_max / P_batt,
        "Total cost of Capacity (Million USD)": result.total_cost / 1000,
        "Load served over year (GWh)": load_served / 1000,
        "Energy cost ($/KWh)": result.total_cost / load_served,
        "Peak Date": as_datetime(peak_day * 24),
        "Clustering Method": cluster_method_name(cluster_method),
    }


def find_peak_day(dispatch):
    soc = dispatch.per_day().soc
    return np.unravel_index(np.argmin(soc), soc.shape)[0]


def sensitivity_dataframe(results, loads, costs, generation):
    data = init_data(generation)

    for load_scenario, cost_scenario in product(loads.columns, costs.columns):
        for k, v in extract_data(
            results, load_scenario, cost_scenario, generation, loads
        ).items():
            data[k].append(v)
    return pd.DataFrame(data)


def plot_yearly_peak_dispatch(dispatch):
    peak_day = find_peak_day(dispatch)
    return visualize.plot_stack(dispatch.by_day(peak_day))


def plot_july(dispatch):
    return visualize.min_capacity_per_month(dispatch, ["July"])[0]


PLOTS = {
    "peak_dispatch": plot_yearly_peak_dispatch,
    "july_dispatch": lambda d: visualize.min_capacity_per_month(d, [7])[0],
    "december_dispatch": lambda d: visualize.min_capacity_per_month(d, [12])[0],
    "curtailment": visualize.daily_curtailment,
    "storage_stats": visualize.storage_capacity_statistics,
    "clustering": {
        "kmeans_cluster": visualize.plot_wind_solar_cluster,
        "manual_cluster": visualize.plot_cluster_1d
    }
}


def create_plotting_artifact(save_dir, results):
    save_dir.mkdir()
    plt.ioff()
    for i, (cluster_method, dispatch) in enumerate(results.values()):
        for plot_name, func in PLOTS.items():
            if plot_name == "clustering":
                fig = func[cluster_method](dispatch.peak_data)
            else:
                fig = func(dispatch)
            fig.set_size_inches([10, 10])
            fig.savefig(save_dir / f"{plot_name}_{i}.png")
            plt.close()
