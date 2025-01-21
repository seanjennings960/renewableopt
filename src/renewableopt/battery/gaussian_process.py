import numpy as np
import pandas as pd

from renewableopt.battery.feature_extraction import CapacityData

PREPROCESS_DTYPE = np.dtype([
    ("uid", "U11"),  # unique identifier of cell
    ("T", float),  # Ambient temperature during cycling
    ("T_m", float),  # Temperature during measurement
    ("delta", float),  # Depth of discharge
    ("Q", float),  # Max SoC
    ("gamma_ch", float),  # Charge rate
    ("gamma_dch", float),  # discharge rate
    ("k", float),  # Number of equivalent full cycles
    ("c_ch", float),  # Capacity measured during charge
    ("c_dch", float)  # Capacity measured during discharge
])

def row_from_uid(uid, exp):
    serial = uid[:9]
    stage = int(uid[-1])
    row = exp[
        (exp["serial"] == serial) &
        (exp["stage"] == stage)]
    if len(row) == 0:
        raise ValueError(f"UID {uid} not found")
    elif len(row) > 1:
        raise ValueError("Multiple UIDs found!")
    return row.iloc[0]

def preprocess(c: CapacityData, exp: pd.DataFrame):
    uids = sorted(c.keys())
    # Filter out cell where the data was missing or corrupted.
    uids = [uid for uid in uids if c[uid] is not None]

    X = []  # Features to fit, tuples of (T_a, delta, Q, gamma+, gamma-, n_F)

    for uid in uids:
        row = row_from_uid(uid, exp)
        T_a = row["amb_temp_tp"]
        delta = row["dod_tp"]
        Q = row["soc_max_tp"]
        gamma_ch = row["c_ch_tp"]
        gamma_dch = row["c_dch_tp"]

        checkups = list(c[uid].iter_both())
        N = len(checkups)
        for i, (cap_ch, cap_dch, n_f) in enumerate(checkups):
            # T_m = Temperature at which capacity is measured
            if i == 0 or i == N - 3:
                T_m = 10
            elif i == 1 or i == N - 2:
                T_m = 23
            elif i == 2 or i == N - 1:  #noqa
                T_m = 45
            else:
                T_m = T_a

            # if np.any(np.isnan(x_i)) or np.isnan(cap):
            #     continue
            x_i = (uid, T_a, T_m, delta, Q, gamma_ch, gamma_dch, n_f, cap_ch, cap_dch)
            X.append(x_i)
    return np.array(X, dtype=PREPROCESS_DTYPE)
