from itertools import groupby

import numpy as np

from renewableopt.optimal_design.multi_period import MockOptimizationResult
from renewableopt.peak_id import as_datetime, reshape_by_day, timedelta


def groupby_index(it):
    with_index = list(zip(it, np.arange(len(it))))
    for val, group in groupby(with_index, key=lambda x: x[0]):
        # Extract indices from group
        _, indices = zip(*group)
        yield (val, np.array(indices))


def charge_periods(load, load_minus_batt, eps=1e-5):
    discharge_i = load > load_minus_batt + eps
    charge_i = load + eps < load_minus_batt
    charge_action = np.choose(discharge_i.astype(int) - charge_i.astype(int) + 1,
                              ["Charge", "Full", "Discharge"])
    return list(groupby_index(charge_action))


def day_indices(time, day):
    dt = as_datetime(time)
    return np.logical_and(
        dt >= day,
        dt < day + np.timedelta64(24, "h")
    )


class InfeasibleControlError(Exception):
    def __init__(self, msg, u, x):
        self.u = u
        self.x = x
        super().__init__(msg)

def greedy_battery_control(result, load, generation):
    dt = result.dt
    E_max = result.E_max
    x0 = E_max * result.model.eta
    P_max = result.P_battery
    E_min = E_max * result.model.rho
    # Various other controls are feasible and quite likely superior in terms
    # of limiting degradation. But this should give a better sense of how much
    # we are running up on the limits, than the control given by LP output?
    u_out = np.zeros_like(load)
    x_out = np.zeros_like(load)
    for i, (load_t, gen_t) in enumerate(zip(load, generation)):
        x_curr = x0 if i == 0 else x_out[i - 1]
        # Pull maximum amount of generation (infinite curtailment)
        u_curr = max(-P_max, load_t - gen_t)
        if u_curr > P_max:
            raise InfeasibleControlError(
                "Load exceeds battery discharge limits.", u_out[:i], x_out[:i])
        x_next = x_curr - dt * u_curr
        if x_next < E_min:
            raise InfeasibleControlError(
                "Out of battery capacity.", u_out[:i], x_out[:i])
        elif x_next > E_max:
            # Battery is full!
            x_next = E_max
            u_curr = (x_curr - E_max) / dt
        x_out[i] = x_next
        u_out[i] = u_curr
    return u_out, x_out


class DispatchData:
    def __init__(self, time, load, gen, u_batt, soc, sources, result, feasible, peak_data):
        self.time = time
        self.load = load
        self.gen = gen
        self.u_batt = u_batt
        self.soc = soc
        for arr in [load, u_batt, soc]:
            assert arr.shape == time.shape
        assert gen.shape == (*time.shape, len(sources))
        self.sources = sources
        self.result = result
        self.feasible = feasible
        self.dt = timedelta(time.flatten())
        self.peak_data = peak_data

    def __getitem__(self, index):
        return DispatchData(
            self.time[index],
            self.load[index],
            self.gen[index],
            self.u_batt[index],
            self.soc[index],
            self.sources,
            self.result,
            self.feasible,
            self.peak_data
        )

    def by_day(self, day):
        day = as_datetime(np.array(day * 24))
        return self[day_indices(self.time, day)]


    def per_day(self):
        return DispatchData(
            *reshape_by_day(self.time, self.load, self.gen, self.u_batt, self.soc),
            self.sources, self.result, self.feasible, self.peak_data
        )

    def __len__(self):
        return len(self.time)

    @classmethod
    def from_greedy(cls, time, load, gen_pu, sources, result, peak_data):
        gen = result.scale_generation(gen_pu)
        gen_full = result.scale_generation(gen_pu, sum_sources=False)
        try:
            u_batt, soc = greedy_battery_control(result, load, gen)
            feasible = True
        except InfeasibleControlError as err:
            u_batt, soc = err.u, err.x
            err_i = len(u_batt)
            time = time[:err_i]
            load = load[:err_i]
            gen_full = gen_full[:err_i]
            feasible = False

        return cls(
            time, load, gen_full, u_batt, soc, sources, result, feasible, peak_data
        )

    def worst_cases(self):
        per_day = self.per_day()
        peaks = self.peak_data
        time_one_day = per_day.time[0]
        return {
            grp: DispatchData.from_greedy(
                time_one_day, peaks.load[grp],
                peaks.gen_pu[grp], self.sources, self.result, self.peak_data)
            for grp in peaks.load.keys()
        }


    def increment_capacity(self, delta_P_batt, delta_E_batt, delta_P_gen, gen_pu):
        E_max = self.result.E_max + delta_E_batt
        P_generation = self.result.P_generation + delta_P_gen
        P_batt = self.result.P_battery + delta_P_batt
        new_result = MockOptimizationResult(
            E_max, P_generation, P_batt, self.result.dt, self.result.model)
        return DispatchData.from_greedy(
            self.time, self.load, gen_pu, self.sources, new_result, self.peak_data)

    def curtailed_generation(self, strategy="even"):
        sources = self.sources
        gen = self.gen.copy()
        total_gen_needed = self.load - self.u_batt
        if strategy in sources:
            curtail_i = sources.index(strategy)
            no_curtail_i = [i for i in range(len(sources)) if i != curtail_i]
            other_gen = np.sum(gen[..., no_curtail_i], axis=1)
            total_gen_needed -= other_gen
            if np.any(total_gen_needed < 0):
                # Specifying a curtailment order would really be better than
                # erroring here.
                raise ValueError("Load/generation cannot be balanced by "
                                 f"curtailing only single source: {strategy}")
            # Limit only the source requested.
            gen[..., curtail_i] = np.clip(gen[..., curtail_i], None,
                                        total_gen_needed)
        elif strategy=="even":
            total_gen = np.sum(gen, axis=-1)
            ratio = total_gen / total_gen_needed
            gen /= ratio[..., np.newaxis]
        else:
            raise ValueError(f"Given strategy ({strategy}) "
                             f"does not match name of a source. "
                             f"Sources: {sources}")
        return gen

    def total_curtailment(self):
        return np.sum(self.gen - self.curtailed_generation("even"), axis=-1)
