import numpy as np
from scipy.optimize import linprog

from renewableopt.peak_id import assert_int, gamma_matrix, timedelta

STATUS_MESSAGES = {
    0 : "Optimization terminated successfully.",
    1 : "Iteration limit reached.",
    2 : "Problem appears to be infeasible.",
    3 : "Problem appears to be unbounded.",
    4 : "Numerical difficulties encountered."
}

GEN_PROFILE_MAX_DIM = 2

class MultiPeriodResult:
    def __init__(self, result, model, dt, scenarios, num_generation, C, b0):
        if not result.success:
            err = (f"Linear program failed with status code {result.status}:\n"
                   f"{STATUS_MESSAGES[result.status]}")
            raise RuntimeError(err)

        self.result = result
        self.model = model
        self.scenarios = scenarios

        num_design_params = num_generation + 2
        design_params = result.x[-num_design_params:]
        self.E_max = design_params[0]
        self.P_battery = design_params[-1]
        self.P_generation = design_params[1:-1]
        self.dt = dt
        self.num_timesteps = assert_int((result.x.shape[0] - num_design_params) / len(scenarios))
        self.u_batt = {
            name: self.battery_control(k)
            for k, name in enumerate(scenarios)
        }
        self.x = {
            name: self.battery_charge(u_batt)
            for name, u_batt in self.u_batt.items()
        }
        self.C = C
        self.b0 = b0
        self.num_generation = num_generation

    def battery_control(self, k):
        # Find the battery control for a given scenario
        start = k * self.num_timesteps
        end = (k + 1) * self.num_timesteps
        return self.result.x[start:end]

    def battery_charge(self, u_batt):
        # Calculate state of charge based on the feasible
        # u_batt control.
        gamma = gamma_matrix(self.num_timesteps, self.dt)
        x0 = self.model.eta * self.E_max
        x = gamma @ u_batt + x0
        return np.r_[x0, x[:-1]]

    def scale_generation(self, generation_pu):
        # Accepts pu generation of shape (T, G) (or shape (T,) if G=1).
        # and returns a shape (T,) array of the total available generation
        # at each time.
        if generation_pu.ndim <= 1:
            assert self.num_generation == 1, \
                f"Got 1D generation_pu, but the number of generation sources is {self.num_generation}"
        elif generation_pu.ndim != GEN_PROFILE_MAX_DIM:
             raise ValueError("Only up to 2 dimensional generation profiles supported.")
        else:
            # 2D input array.
            assert generation_pu.shape[1] == self.num_generation, \
                f"Input has {generation_pu.shape[1]} generation sources, but result contains {self.num_generation}"

        generation = generation_pu * self.P_generation
        if generation.ndim == GEN_PROFILE_MAX_DIM:
            generation = np.sum(generation, axis=-1)
        return generation

class MultiPeriodModel:
    def __init__(self, initial_battery_charge, depth_of_discharge,
                 cost_battery_energy, cost_battery_power, cost_generation):
        self.eta = initial_battery_charge
        self.rho = depth_of_discharge
        self.cost_battery_energy = cost_battery_energy
        self.cost_battery_power = cost_battery_power
        self.cost_generation = cost_generation
        self.num_generation = len(cost_generation)


    def minimize_cost(self, time, load, generation):
        """Minimize cost over multiple periods, ensuring battery is full at EoD."""
        T = time.shape[0]
        dt = timedelta(time)
        # Load (str -> array(T))
        # Generation (str -> array(T, G))
        scenarios = list(load.keys())
        assert  set(scenarios) == set(generation.keys()), \
            "Load and generation dictionaries must have same keys."
        for l_profile in load.values():
            #
            assert l_profile.shape[0] == T, "Each load profile must have same number of timesteps."

        generation = generation.copy()
        for name, gen in generation.items():
            assert gen.shape[0] == T, "Each generation profile must have same number of timesteps."
            if gen.ndim == 1:
                generation[name] = gen[:, np.newaxis]
                assert self.num_generation == 1, \
                    f"Generation array is 1D, but cost given for {self.num_generation} generation sources."
            elif gen.ndim > GEN_PROFILE_MAX_DIM:
                raise ValueError("Got generation profile with dimension >2.")
            elif generation[name].shape[1] != self.num_generation:
                raise ValueError(
                    "Not all generation scenarios have the same number of generation sources as given "
                    f"number of generation costs ({self.num_generation}!")

        # Constraint sensitivity w.r.t. battery control
        gamma = gamma_matrix(T, dt)
        A = np.vstack([
            gamma,
            -gamma,
            np.eye(T),
            -np.eye(T),
            np.eye(T),
            -np.eye(T)
        ])

        zero = np.zeros(T)
        one = np.ones(T)

        # Constraint sensitivity w.r.t. design parameters
        B_list = [
            np.c_[
                np.r_[one * (1 - self.eta), one[:-1] * (self.eta - self.rho), 0,
                      zero, zero, zero, zero],
                np.kron(np.array([0, 0, 0, 1, 0, 0])[:, np.newaxis], generation[k]),
                np.kron([0, 0, 0, 0, 1, 1], one)
            ]
            for k in scenarios
        ]


        # Full constraint matrix for concatenated decision variables [u_k, d]
        C = np.c_[
            np.kron(np.eye(len(scenarios)), A),
            -np.vstack(B_list)
        ]
        # Constant constraints from load
        b_list = [
            np.kron([0, 0, 1, -1, 0, 0], load[k])
            for k in scenarios
        ]
        b0 = np.concatenate(b_list)
        # Costs associated with battery control (0) and system design
        c = np.r_[np.zeros(T * len(scenarios)),
                  self.cost_battery_energy,
                  self.cost_generation,
                 self.cost_battery_power]


        return MultiPeriodResult(
            linprog(c, A_ub=C, b_ub=b0, bounds=(None, None)),
            self,
            dt,
            scenarios,
            self.num_generation,
            C, b0
        )


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
            raise ValueError("Infeasible control: Load exceeds battery discharge limits.")
        x_next = x_curr - dt * u_curr
        if x_next < E_min:
            raise ValueError("Infeasible control: Out of battery capacity.")
        elif x_next > E_max:
            # Battery is full!
            x_next = E_max
            u_curr = (x_curr - E_max) / dt
        x_out[i] = x_next
        u_out[i] = u_curr
    return u_out, x_out

