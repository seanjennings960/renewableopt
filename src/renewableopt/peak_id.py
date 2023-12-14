"""Identification of days which produce peak stress on power system."""
from functools import reduce
from itertools import groupby

import numpy as np
from sklearn.cluster import KMeans


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


def as_datetime(time_rel_hr):
    # Convert to numpy datetime....
    jan1 = np.datetime64("2012-01-01 00:00")
    dts_min = (time_rel_hr * 60).astype("timedelta64[m]")
    return jan1 + dts_min


def timedelta(time):
    """Return the delta between each time[i+1] and time[i], asserting it constant."""
    dt = time[1] - time[0]
    assert np.all(np.isclose(np.diff(time), dt))
    return dt

def shape_by_day(time_hr):
    dt = timedelta(time_hr)
    timesteps_per_day = assert_int(24 / dt)
    days = assert_int(time_hr.shape[0] / timesteps_per_day)
    return (days, timesteps_per_day)

def reshape_by_day(time, *arrs):
    days, timesteps_per_day = shape_by_day(time)
    # time_per_day = time.reshape(days, timesteps_per_day)
    # load_per_day = load.reshape(days, timesteps_per_day)
    # solar_pu_per_day = solar_pu.reshape(days, timesteps_per_day)
    return tuple(
        arr.reshape(days, timesteps_per_day, -1).squeeze()
        for arr in [time, *arrs]
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


def dedup(a):
    a_arr = np.copy(a)
    out = a_arr.tolist()
    dups = find_duplicates(a)
    for dup in dups:
        for i, dup_i in enumerate(np.argwhere(a_arr == dup)):
            out[dup_i.squeeze()] += f"_{i}"
    return out

def find_duplicates(a):
    return [x for x, g in groupby(np.sort(a)) if len(list(g)) > 1]


def interval_namer(x):
    x_range = np.linspace(np.min(x), np.max(x), 4)
    intervals = [
        ("low", x_range[0], x_range[1]),
        ("medium", x_range[1], x_range[2]),
        ("high", x_range[2], x_range[3]),
    ]
    def decide_name(y):
        for name, low, high in intervals:
            if low <= y <= high:
                return name
        return "out of bounds"
    return decide_name

def cluster_names(kmeans, peak_loads, daily_gen, sources):
    load_namer = interval_namer(peak_loads)
    gen_namers = {s: interval_namer(daily_gen[:, i])
                 for i, s in enumerate(sources)}
    names = []
    for center in kmeans.cluster_centers_:
        name = f"{load_namer(center[0])}_load"
        for i, source in enumerate(sources):
            name += f"_{gen_namers[source](center[i + 1])}_{source}"
        names.append(name)
    return dedup(names)



def kmeans_cluster(problem_days, load, gen, sources, n_clusters=5):
#     load = peak_loads[problem_days]
#     gen = daily_gen[problem_days]
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(np.c_[
        # HACK: gen is in unit MWh/day/MW_installed, so a value of 24,
        # corresponds to capacity factor 1. Multiply by 100 so to get
        # a percentage-based capacity factor 10-40%. peak loads are typically
        # 70-100 MW, so this roughly normalizes them...
        load, gen * 100/24
    ])
    names = cluster_names(kmeans, load, gen, sources)
    return {
        name: problem_days[
            kmeans.labels_ == i
        ]
        for i, name in enumerate(names)
    }


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


SUPPORTED_METHODS = ["manual_cluster", "kmeans_cluster", "cap_ratio", "peak_load"]


def identify_worst_days(time, load, gen_pu, *, sources=None, method="kmeans_cluster"):
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
    energy_per_day = np.einsum(
        "ijk,ja->iak", gen_pu_per_day, gamma.T
    )

    # Order by peak load and minimum daily solar generation.
    # Days with low generation could cause problems just as easily
    # as days with high load, so find maximums of partial ordering!
    peak_loads = np.max(load_per_day, axis=-1)
    daily_gen = np.min(energy_per_day, axis=1)

    if method=="cap_ratio":
        # HACK: just use first generation source to determine ratio.
        cap_ratio = peak_loads / daily_gen[:, 0]
        worst_index = np.argmin(cap_ratio)
        return PeakData({
            "worst_cap_ratio": [worst_index]
        }, {
            "worst_cap_ratio": load_per_day[worst_index]
        }, {
            "worst_cap_ratio": gen_pu_per_day[worst_index]
        }, peak_loads, daily_gen)
    elif method=="peak_load":
        worst_indices = np.argsort(peak_loads)
        return PeakData(
            {f"peak_{i}": [i] for i in worst_indices},
            {
            f"peak_{i}": load_per_day[i]
            for i in worst_indices[-5:]
        }, {
            f"peak_{i}": gen_pu_per_day[i]
            for i in worst_indices[-5:]
        }, peak_loads, daily_gen)

    elif method in ["manual_cluster", "kmeans_cluster"]:
        if method == "kmeans_cluster" and sources is None:
            raise ValueError("Sources must be specified with kmeans cluster method.")
        # if method == "manual_cluster" and daily_gen.shape[1] > 1:
        #     raise ValueError("Manual clustering only supported with solar data. "
        #                      "Got multiple generation sources.")
        if method == "manual_cluster":
            daily_gen = daily_gen[:, sources.index("solar")]
        else:
            # Remove geothermal to make names less verbose.
            nongeothermal = [i for i, s in enumerate(sources) if s != "geothermal"]
            sources = [s for s in sources if s != "geothermal"]
            daily_gen = daily_gen[:, nongeothermal]

        # TODO: consider minimum capacity factor than load peaks individually??
        problem_days = topo_argmax(np.c_[
            peak_loads, daily_gen
        ])

        # Group together days which are close together on the (peak_load, daily_solar)
        # plane and find the worst case within each group, to reduce computational burden,
        # at relatively small optimality cost.
        if method == "manual_cluster":
            problem_groups = manual_clustering(problem_days, peak_loads, daily_gen)
        else:
            problem_groups = kmeans_cluster(
                problem_days,
                peak_loads[problem_days],
                # Pass negative sign so naming works properly
                -daily_gen[problem_days],
                sources,
                n_clusters=15
            )
        assert_partition(problem_groups.values(), set(problem_days))
        # For worst case generation, look at the integral, since any
        worst_load, worst_energy = worst_case_by_group(
            problem_groups, load_per_day, energy_per_day)
        worst_gen_pu = {
            name: np.linalg.inv(gamma) @ energy
            for name, energy in worst_energy.items()
        }
        return PeakData(
            problem_groups, worst_load, worst_gen_pu, peak_loads, daily_gen)


class PeakData:
    def __init__(self, problem_groups, load, gen_pu, peak_loads, daily_gen):
        # Problem groups: Dict[group] -> List of days
        # load and gen_pu: Dict[group] -> array of load/generation
        self.problem_groups = problem_groups
        self.load = load
        self.gen_pu = gen_pu
        self.peak_loads = peak_loads
        self.daily_gen = daily_gen
        if self.daily_gen.ndim == 1:
            # HACK: Add new axis for when we do manual clustering and clobber
            # the other ones...
            self.daily_gen = daily_gen[:, np.newaxis]

    @property
    def daily_solar_capacity(self):
        return  - self.daily_gen[:, 0] / 24 * 100

    @property
    def daily_wind_capacity(self):
        return  - self.daily_gen[:, 1] / 24 * 100
