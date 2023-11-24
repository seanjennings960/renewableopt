import numpy as np
from scipy.optimize import linprog

from renewableopt.peak_id import gamma_matrix


class SinglePeriodResult:
    def __init__(self, result, model, dt):
        self.result = result
        self.model = model
        self.E_max = result.x[-2]
        self.P_solar = result.x[-1]
        self.u_batt = result.x[:-2]
        self.dt = dt

        # Calculate state of charge based on the feasible
        # u_batt control.
        gamma = gamma_matrix(self.u_batt.shape[0], dt)
        x0 = self.model.eta * self.E_max
        x = gamma @ self.u_batt + x0
        self.x = np.r_[x0, x[:-1]]

class SinglePeriodModel:
    def __init__(self, initial_battery_charge, depth_of_discharge,
                 cost_battery_energy, cost_solar):
        self.eta = initial_battery_charge
        self.rho = depth_of_discharge
        self.cost_battery = cost_battery_energy
        self.cost_solar = cost_solar


    def minimize_cost(self, time, load, solar_pu):
        T = time.shape[0]
        dt = time[1] - time[0]
        gamma = np.tril(np.full((T, T), -dt))
        A = np.vstack([
            gamma,
            -gamma,
            np.eye(T),
            -np.eye(T)
        ])

        zero = np.zeros(T)
        one = np.ones(T)

        b0 = np.r_[
            zero,
            zero,
            load,
            -load
        ]
        B = np.r_[
            np.c_[(1 - self.eta) * one, zero],
            np.c_[(self.eta - self.rho) * one, zero],
            np.c_[zero, zero],
            np.c_[zero, solar_pu]
        ]
        # Full constraint matrix for concatenated decision variables [u, d]
        C = np.c_[A, -B]
        # Cost: zero for battery usage,
        c = np.r_[zero, self.cost_battery, self.cost_solar]
        return SinglePeriodResult(
            linprog(c, A_ub=C, b_ub=b0, bounds=(None, None)),
            self,
            dt)
