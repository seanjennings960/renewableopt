"""Identification of days which produce peak stress on power system."""
from functools import reduce

import numpy as np


def assert_partition(parts, full_set):
    # Check that problem groups form a partition
    full_union = reduce(
        lambda a, b: a.union(b),
        parts,
        set()
    )
    assert full_union == full_set
    intersection = reduce(
        lambda a, b: a.intersection(b),
        parts,
        full_union
    )
    assert intersection == set()

def assert_int(a):
    assert a.is_integer()
    return np.int32(a)


def timedelta(time):
    """Return the delta between each time[i+1] and time[i], asserting it constant."""
    dt = time[1] - time[0]
    assert np.all(np.isclose(np.diff(time), dt))
    return dt

def reshape_by_day(time, load, generation_pu):
    dt = timedelta(time)
    timesteps_per_day = assert_int(24 / dt)
    days = assert_int(time.shape[0] / timesteps_per_day)
    # time_per_day = time.reshape(days, timesteps_per_day)
    # load_per_day = load.reshape(days, timesteps_per_day)
    # solar_pu_per_day = solar_pu.reshape(days, timesteps_per_day)
    return tuple(
        arr.reshape(days, timesteps_per_day, -1).squeeze()
        for arr in [time, load, generation_pu]
    )


def topological_sort(arr: np.ndarray, order=None, axis=0):
    r"""
    Return a graph (adjacency matrix) of a (partial) order along given axis.

    Let M be a metric space. A slice of arr[axis] returns an
    ndarray[M].

    Arguments:
        arr: array to be sorted
        order: function taking (a: M, b: ndarray[M]) and returning
            a <= b: ndarray[bool]. Note: rules of a (partial) order should hold:
                1. if a <= b and b <= c, then a <= c.
                2. a <= a is true for all a \in M
                (for partial only) 3. there exists a,b with the property both a <= b and
                    a >= b is false.
    """
    arr = arr.swapaxes(0, axis)
    n = arr.shape[0]
    g = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        comp_indices = np.arange(0, n) != i
        g[i, comp_indices] = order(arr[i], arr[comp_indices])
    return g


def partial_order(a, b):
    return np.all(a <= b, axis=tuple(i for i in range(1, b.ndim)))


def gamma_matrix(n, dt):
    return np.tril(np.full((n, n), -dt))


def topo_argmax(arr):
    # There's some hard-coded arguments to topological_sort and other axis arguments.
    # Make this a bit more general?
    g = topological_sort(arr, partial_order)
    num_less_than = np.sum(g, axis=-1)
    return np.argwhere(num_less_than == 0)[:, 0]


def manual_clustering(problem_days, peak_loads, daily_solar):
    problem_groups = {}
    problem_groups["low_load_low_solar"] = problem_days[
        peak_loads[problem_days] < 70  # noqa
    ]
    problem_groups["medium_load_low_solar"] = problem_days[np.logical_and(
        peak_loads[problem_days] > 70,  # noqa
        daily_solar[problem_days] > -2  # noqa
    )]
    problem_groups["medium_load_medium_solar"] = problem_days[np.logical_and(
        daily_solar[problem_days] > -4,  # noqa
        daily_solar[problem_days] < -2  # noqa
    )]
    problem_groups["high_load_cloudy"] = problem_days[np.logical_and(
        daily_solar[problem_days] > -8,  # noqa
        daily_solar[problem_days] < -6  # noqa
    )]
    problem_groups["high_load_sunny"] = problem_days[
        daily_solar[problem_days] < -9,  # noqa
    ]
    return problem_groups


def worst_case_by_group(problem_groups, load_per_day, energy_solar_per_day):
    worst_load = {
        name: np.max(load_per_day[group], axis=0)
        for name, group in problem_groups.items()
    }
    worst_solar = {
        name: np.max(energy_solar_per_day[group], axis=0)
        for name, group in problem_groups.items()
    }
    return worst_load, worst_solar


SUPPORTED_METHODS = ["toposort", "cap_ratio", "peak_load"]


def identify_worst_days(time, load, gen_pu, *, method="toposort"):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Method {method} not supported. Select from: {SUPPORTED_METHODS}")
    dt = timedelta(time)
    time_per_day, load_per_day, gen_pu_per_day = reshape_by_day(
        time, load, gen_pu
    )
    timesteps_per_day = time_per_day.shape[1]
    # Integrate solar output over the course of the day.
    gamma = gamma_matrix(timesteps_per_day, dt)
    # Multiply over timesteps_per_day axis
    # gen_pu_per_day has shape (days, timesteps, num_sources)
    if gen_pu_per_day.ndim < 3:  # noqa
        gen_pu_per_day = gen_pu_per_day[:, :, np.newaxis]
    energy_solar_per_day = np.einsum(
        "ijk,ja->ijk", gen_pu_per_day, gamma.T
    )

    # Order by peak load and minimum daily solar generation.
    # Days with low generation could cause problems just as easily
    # as days with high load, so find maximums of partial ordering!
    peak_loads = np.max(load_per_day, axis=-1)
    daily_gen = np.min(energy_solar_per_day, axis=1)

    if method=="cap_ratio":
        # HACK: just use first generation source to determine ratio.
        cap_ratio = peak_loads / daily_gen[:, 0]
        worst_index = np.argmin(cap_ratio)
        return ({
            "worst_cap_ratio": load_per_day[worst_index]
        }, {
            "worst_cap_ratio": gen_pu_per_day[worst_index]
        })
    elif method=="peak_load":
        worst_indices = np.argsort(peak_loads)
        return {
            f"peak_{i}": load_per_day[i]
            for i in worst_indices[-5:]
        }, {
            f"peak_{i}": gen_pu_per_day[i]
            for i in worst_indices[-5:]
        }

    # TODO: consider minimum capacity factor than load peaks individually??
    problem_days = topo_argmax(np.c_[
        peak_loads, daily_gen
    ])

    # Group together days which are close together on the (peak_load, daily_solar)
    # plane and find the worst case within each group, to reduce computational burden,
    # at relatively small optimality cost.
    problem_groups = manual_clustering(problem_days, peak_loads, daily_gen)
    assert_partition(problem_groups.values(), set(problem_days))
    # For worst case generation, look at the integral, since any
    worst_load, worst_solar_energy = worst_case_by_group(
        problem_groups, load_per_day, energy_solar_per_day)
    worst_solar_pu = {
        name: np.linalg.inv(gamma) @ solar_energy
        for name, solar_energy in worst_solar_energy.items()
    }
    return worst_load, worst_solar_pu
