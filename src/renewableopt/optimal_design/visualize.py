from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from renewableopt.optimal_design.dispatch import charge_periods, greedy_battery_control
from renewableopt.peak_id import as_datetime

###############################################################################
# Visualizations
###############################################################################

JAN1 = datetime(2012, 1, 1)
DECEMBER = 12

def initialize_day_ranges():
    day_ranges = {}
    for month in range(1, 13):
        start = datetime(2012, month, 1)
        if month != DECEMBER:
            end = datetime(2012, month + 1, 1)
        else:
            end = datetime(2013, 1, 1)
        day_ranges[month] = (
            (start - JAN1).days,
            (end - JAN1).days
        )
    return day_ranges

# Constants for month related manipulations.
DAY_RANGES = initialize_day_ranges()
MONTH_NAMES_DICT = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}
NAME_TO_MONTH_NUM = {
    name: num for num, name in MONTH_NAMES_DICT.items()
}
MONTH_NAMES = [MONTH_NAMES_DICT[i] for i in range(1, 13)]


def iter_months(data_per_day):
    for month in range(1, 13):
        start, end = DAY_RANGES[month]
        yield data_per_day[start:end]


def storage_capacity_statistics(dispatch, *, hack=False):
    result = dispatch.result
    soc_per_day = dispatch.per_day().soc

    E_max = result.E_max
    E_min = result.E_max * result.model.rho
    # eod_soc = [soc[:, -1] for soc in iter_months(soc_per_day)]
    min_soc = [np.min(soc, axis=1) for soc in iter_months(soc_per_day)]
    # HACK: remove bad data from first day
    if hack:
        min_soc[0] = min_soc[1]
    # max_soc = [np.max(soc, axis=1) for soc in iter_months(soc_per_day)]

    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig = plt.figure()
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    # plt.suptitle("Storage Capacity Statistics by Month")
    # plt.boxplot(eod_soc, vert=True, patch_artist=True, labels=MONTH_NAMES)
    # plt.set_title("Storage Remaining End of Day")
    # plt.set_ylabel("Energy (MWh)")
    plt.boxplot(min_soc, vert=True, patch_artist=True, labels=MONTH_NAMES)
    plt.title("Minimum Capacity Reached")
    plt.ylabel("Battery Charge (MWh)")
    # ax3.boxplot(max_soc, vert=True, patch_artist=True, labels=month_names)
    # ax3.set_title("Maximum Capacity Reached")
    # for ax in [ax1, ax2, ax3]:
    # for ax in [ax1, ax2]:
    plt.axhline(y=E_max, color="green", label="maximum")
    plt.axhline(y=E_min, color="orange", label="minimum")
    # plt.legend()
    fig.autofmt_xdate(rotation=80)
    return fig


def daily_curtailment(dispatch):
    curtailment_by_day = dispatch.per_day().total_curtailment()
    daily_curtailment = [np.sum(c, axis=1) * dispatch.dt / 1000
                        for c in iter_months(curtailment_by_day)]
    fig = plt.figure()
    plt.boxplot(daily_curtailment, vert=True, patch_artist=True, labels=MONTH_NAMES)
    fig.autofmt_xdate(rotation=80)
    plt.title("Daily Curtailment By Month")
    plt.ylabel("Energy curtailed (GWh/day)")
    return fig


def plot_battery_status(result, possible_controls, time, load, generation, title=None, plots=None):
    if plots is None:
        _, plots = plt.subplots(1, 2)

    title = f"Battery Status: {title}" if title is not None else "Battery Status"
    plt.suptitle(title)
    u_max, u_min = load, load - generation

    # Plot of battery power output
    plots[0].plot(time, u_min, label="Meets Load w/ 0% Curtailment")
    plots[0].plot(time, u_max, label="Meets Load w/ 100% Curtailment")
    max_u_batt = -np.inf
    for control_name, (u_batt, _) in possible_controls.items():
        label = f"Battery Power ({control_name})" if control_name is not None else "Battery Power"
        plots[0].plot(time, u_batt, label=label)
        max_u_batt = max(max_u_batt, np.max(u_batt))

    plots[0].set_xlabel("Time of Day (hr)")
    plots[0].set_ylabel("Battery Charge(-)/Discharge(+) rate (MW)")

    # Max and min lines based on maximum power output.
    plots[0].axhline(y=-result.P_battery, linestyle="--", color="green", label="Charge Limit")
    if np.max(max_u_batt) > 0.5 * result.P_battery:
        plots[0].axhline(y=result.P_battery, linestyle="--", color="orange", label="Discharge limit")
    plots[0].legend()

    for control_name, (_, soc) in possible_controls.items():
        label = f"SoC ({control_name})" if control_name is not None else "SoC"
        plots[1].plot(time, soc, label=label)
    plots[1].set_xlabel("Time of Day (hr)")
    plots[1].set_ylabel("Battery State of Charge (MWh)")
    # Capacity lines
    plots[1].axhline(y=result.E_max, linestyle="--", color="green", label="Maximum Capacity")
    plots[1].axhline(y=result.model.rho * result.E_max, linestyle="--", color="orange", label="Minimum Capacity")
    plots[1].legend()

def min_capacity_per_month(dispatch, months):
    pd = dispatch.per_day()
    figs = []
    for month in months:
        start, end = DAY_RANGES[month]
        # Day with minimum capacity
        day = start + np.argmin(
            # Minimum start across day
            np.min(pd.soc[start:end], axis=1)
        )
        figs.append(plot_stack(dispatch.by_day(day)))
        # date = datetime.strftime(JAN1 + dt_timedelta(days=int(day)), "%B %d")
        # plot_battery_status(dispatch.result, {
        #     "greedy": (pd.u_batt[day], pd.soc[day])
        # }, pd.time[day], pd.load[day], np.sum(pd.gen[day], axis=-1), date)
    return figs



def readable(s):
    return " ".join(map(lambda word: word.capitalize(), s.split("_")))


def lp_versus_greedy_comparison(result, time_one_day, worst_load, worst_generation, scenarios=None):
    if scenarios is None:
        # Show all scenarios by default!
        scenarios = result.scenarios

    for scenario in scenarios:
        possible_controls = {}
        generation = result.scale_generation(worst_generation[scenario])
        possible_controls["Greedy"] = greedy_battery_control(
            result, worst_load[scenario], generation)
        possible_controls["LP"] = (result.u_batt[scenario], result.x[scenario])
        plot_battery_status(
            result, possible_controls, time_one_day, worst_load[scenario], generation,
            title=readable(scenario))


def plot_stack(dispatch, curtailment=None):
    # time (T,)
    # load (T,)
    # gen (T, G)
    # u_batt (T,)
    time = as_datetime(dispatch.time)
    # time = dispatch.time
    load = dispatch.load
    curtail_kwargs = {"strategy": curtailment} if curtailment is not None else {}
    gen = dispatch.curtailed_generation(**curtail_kwargs)

    # Reorder generation so most utilized sources appear on bottom of stack
    gen_order = np.argsort(np.sum(dispatch.gen, axis=0))[::-1]
    gen = gen[:, gen_order]
    sources = [dispatch.sources[i] for i in gen_order]
    load_minus_batt = dispatch.load - dispatch.u_batt
    gen_cumsum = np.c_[
        np.zeros_like(dispatch.load), np.cumsum(gen, axis=1)
    ]

    fig, plots = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1])
    for i, source in enumerate(sources):
        # Fill in stack with generation cumulative sum.
        plots[0].fill_between(time, gen_cumsum[:, i],
                            gen_cumsum[:, i+1], label=source.capitalize())


    charge_plotted = False
    discharge_plotted = False
    for action, indices in charge_periods(load, load_minus_batt):
        # We plot charging and discharging differently. When charging,
        # we want to visualize the generation source that is filling charge.
        # We fill in the discharge area since this is contributing to meeting
        # load.
        if action == "Charge":
            # Append index so interval is closed.
            if indices[-1] + 1 < len(time):
                i = np.append(indices, indices[-1] + 1)
            else:
                i = indices
            plots[0].plot(time[i], load_minus_batt[i], "g",
                        linewidth=4,
                        label="Charge Load" if not charge_plotted else "")
            charge_plotted = True
        elif action == "Discharge":
            plots[0].fill_between(time[indices], load[indices],
                                load_minus_batt[indices], color="C3",
                                label="Battery Discharge" if not discharge_plotted else "")
            discharge_plotted = True
    plots[0].plot(time, load, "r", linewidth=4, label="Load")


    plots[0].legend(bbox_to_anchor=(1, 0.8))
    plots[0].set_ylabel("Power (MW)")
    plt.suptitle("Dispatch Stack")
    plots[1].plot(time, dispatch.soc)

    E_max = dispatch.result.E_max
    E_min = dispatch.result.model.rho * E_max
    plots[1].axhline(E_max, color="g", linestyle="--", label="Maximum Capacity",)
    plots[1].axhline(E_min, color="r", linestyle="--", label="Minimum Depth \nof Discharge")

    plots[1].set_ylabel("Battery SoC (MWh)")
    plots[1].legend(bbox_to_anchor=(1, 0.6))
    fig.autofmt_xdate(rotation=80)
    return fig


def plot_wind_solar_cluster(peak_data):
    problem_groups = peak_data.problem_groups
    load = peak_data.peak_loads
    solar = peak_data.daily_solar_capacity
    wind = peak_data.daily_wind_capacity
    fig, plots = plt.subplots(2, 2)
    first_run = True
    labels = np.sort(list(problem_groups.keys()))
    for x, y, i, j in [
        (load, solar, 0, 0),
        (load, wind, 1, 0),
        (wind, solar, 0, 1),
    ]:
        for label in labels:
            days = problem_groups[label]

    #         group = kmeans.labels_ == label
            plots[i][j].plot(x[days], y[days], "x", label=label)
            if first_run:
                plots[1][1].plot([0], [0], "x", label=label)
        first_run = False

    plots[0][0].set_xlabel("Peak Load (MW)")
    plots[0][0].set_ylabel("Solar Daily Capacity Factor (%)")
    plots[0][1].set_xlabel("Wind Daily Capacity Factor (%)")
    plots[0][1].set_ylabel("Solar Daily Capacity Factor (%)")
    plots[1][0].set_xlabel("Wind Daily Capacity Factor (%)")
    plots[1][0].set_xlabel("Peak Load (MW)")
    plots[1][1].legend()
    plots[1][1].axis("off")
    return fig


def plot_cluster_1d(peak_data):
    fig = plt.figure()
    solar = peak_data.daily_solar_capacity
    for name, days in peak_data.problem_groups.items():
        plt.plot(peak_data.peak_loads[days], solar[days], "x", label=name)
    plt.xlabel("Peak Load (MW)")
    plt.ylabel("Daily Capacity Factor (%)")
    plt.legend()
    return fig
