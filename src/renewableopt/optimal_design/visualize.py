from datetime import datetime
from datetime import timedelta as dt_timedelta

import matplotlib.pyplot as plt
import numpy as np

from renewableopt.optimal_design.multi_period import greedy_battery_control

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


def storage_capacity_statistics(result, soc_per_day):
    E_max = result.E_max
    E_min = result.E_max * result.model.rho
    eod_soc = [soc[:, -1] for soc in iter_months(soc_per_day)]
    min_soc = [np.min(soc, axis=1) for soc in iter_months(soc_per_day)]
    # max_soc = [np.max(soc, axis=1) for soc in iter_months(soc_per_day)]

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    plt.suptitle("Storage Capacity Statistics by Month")
    ax1.boxplot(eod_soc, vert=True, patch_artist=True, labels=MONTH_NAMES)
    ax1.set_title("Storage Remaining End of Day")
    ax1.set_ylabel("Energy (MWh)")
    ax2.boxplot(min_soc, vert=True, patch_artist=True, labels=MONTH_NAMES)
    ax2.set_title("Minimum Capacity Reached")
    # ax3.boxplot(max_soc, vert=True, patch_artist=True, labels=month_names)
    # ax3.set_title("Maximum Capacity Reached")
    # for ax in [ax1, ax2, ax3]:
    for ax in [ax1, ax2]:
        ax.axhline(y=E_max, color="green", label="maximum")
        ax.axhline(y=E_min, color="orange", label="minimum")
    fig.autofmt_xdate(rotation=80)


def daily_curtailment(dt, curtailment_by_day):
    daily_curtailment = [np.sum(c, axis=1) * dt
                        for c in iter_months(curtailment_by_day)]
    fig = plt.figure()
    plt.boxplot(daily_curtailment, vert=True, patch_artist=True, labels=MONTH_NAMES)
    fig.autofmt_xdate(rotation=90)
    plt.title("Daily Curtailment By Month")
    plt.ylabel("Energy curtailed (MWh/day)")


def plot_battery_status(result, possible_controls, time, load, generation, title=None):
    fig, plots = plt.subplots(1, 2)

    title = f"Battery Status: {title}" if title is not None else "Battery Status"
    plt.suptitle(title)
    u_max, u_min = load, load - generation

    # Plot of battery power output
    plots[0].plot(time, u_min, label="Load - Generation")
    plots[0].plot(time, u_max, label="Load")
    max_u_batt = -np.inf
    for control_name, (u_batt, _) in possible_controls.items():
        plots[0].plot(time, u_batt, label=f"Battery Power ({control_name})")
        max_u_batt = max(max_u_batt, np.max(u_batt))

    plots[0].set_xlabel("Time of Day (hr)")
    plots[0].set_ylabel("Battery Charge(-)/Discharge(+) rate (MW)")

    # Max and min lines based on maximum power output.
    plots[0].axhline(y=-result.P_battery, linestyle="--", color="green", label="Charge Limit")
    if np.max(max_u_batt) > 0.5 * result.P_battery:
        plots[0].axhline(y=-result.P_battery, linestyle="--", color="orange", label="Discharge limit")
    plots[0].legend()

    for control_name, (_, soc) in possible_controls.items():
        plots[1].plot(time, soc, label=f"SoC ({control_name})")
    plots[1].set_xlabel("Time of Day (hr)")
    plots[1].set_ylabel("Battery State of Charge (MWh)")
    # Capacity lines
    plots[1].axhline(y=result.E_max, linestyle="--", color="green", label="Maximum Capacity")
    plots[1].axhline(y=result.model.rho * result.E_max, linestyle="--", color="orange", label="Minimum Capacity")
    plots[1].legend()

def min_capacity_per_month(result, soc_per_day, u_batt_per_day, time_one_day, load_per_day, generation_per_day):
    for month in range(1, 13):
        start, end = DAY_RANGES[month]
        # Day with minimum capacity
        day = start + np.argmin(
            # Minimum start across day
            np.min(soc_per_day[start:end], axis=1)
        )
        date = datetime.strftime(JAN1 + dt_timedelta(days=int(day)), "%B %d")
        plot_battery_status(result, {
            "greedy": (u_batt_per_day[day], soc_per_day[day])
        }, time_one_day, load_per_day[day], generation_per_day[day], date)


def lp_versus_greedy_comparison(result, time_one_day, worst_load, worst_generation):
    # Generation
    for scenario in result.scenarios:
        possible_controls = {}
        generation = result.scale_generation(worst_generation[scenario])
        possible_controls["Greedy"] = greedy_battery_control(
            result, worst_load[scenario], generation)
        possible_controls["LP"] = (result.u_batt[scenario], result.x[scenario])
        plot_battery_status(result, possible_controls, time_one_day, worst_load[scenario], generation)
