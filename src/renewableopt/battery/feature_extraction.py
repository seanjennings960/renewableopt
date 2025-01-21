from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, griddata
from scipy.signal import savgol_filter

from renewableopt.battery.data_import import import_datafile


############################################################################################################
# CAPACITY EXTRACTION
############################################################################################################
def df_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting the capacity measurement from larger DataFrame

    :param df:
    :return df_capa: pd.DataFrame: DataFrame containing capacity measurement only
    """
    df_capa_meas = df[((df.step_type == 21) & (df.c_cur > 0)) | ((df.step_type == 22) & (df.c_cur < 0))]  # noqa

    return df_capa_meas


def capacity(df: pd.DataFrame) -> list:
    """
    Evaluate capacity of measurement in df with step_type = [21,22]

    :param df: pd.DataFrame: df with step_type = [21,22]
    :return: {'Q_mean': float, 'Q_ch': float, 'Q_dch': float, 'q_ch': np.ndarray, 'q_dch': np.ndarray}
    """
    q_kapa_ch = q_calc(df[(df.step_type == 21) & (df.c_cur > 0)])  # noqa
    q_kapa_dch = q_calc(df[(df.step_type == 22) & (df.c_cur < 0)])  # noqa

    capa_ch = q_kapa_ch[-1] - q_kapa_ch[0]
    capa_dch = q_kapa_dch[0] - q_kapa_dch[-1]

    capa_mean = (capa_ch + capa_dch) / 2 / 3600
    capa_ch = capa_ch / 3600
    capa_dch = capa_dch / 3600

    return {"Q_mean": capa_mean, "Q_ch": capa_ch, "Q_dch": capa_dch, "q_ch": q_kapa_ch, "q_dch": q_kapa_dch}


def q_calc(df: pd.DataFrame) -> np.ndarray:
    """
    Integrate current over time to get charge throughput

    :param df:
    :return q: np.array:
    """
    try:
        q_val = (df.run_time.diff() * df.c_cur).fillna(0).values
        q = np.cumsum(q_val)

        return q

    except Exception as e:
        print(f"Exception: q_calc: {e}")  # noqa
        q = np.array([np.nan])
        return q


############################################################################################################
# OPEN CIRCUIT VOLTAGE (OCV) EXTRACTION
############################################################################################################
def df_ocv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracting the ocv measurement from larger DataFrame

    :param df: pd.DataFrame: larger dataframe containing ocv measurement
    :return df_ocv_meas: pd.DataFrame: DataFrame containing ocv measurement only
    """

    df_ocv_meas = df[(df.step_type == 31) | (df.step_type == 32)]  # noqa

    return df_ocv_meas


def ocv_curve(df: pd.DataFrame, enable_filter: bool = False, polyorder: int = 3) -> dict:  # noqa
    """
    Extracting the open circuit voltage curve (OCV) contained in the dataframe with step_type = [31,32].
    Here, only the constant current CC-OCV measurement can be evaluated.

    :param polyorder: Order of the polynomial filter for smoothing the OCV curve
    :type polyorder: int
    :param enable_filter: Enable filter for smoothing the OCV curve
    :type enable_filter: bool
    :param df:
    :return: {'SoC': np.ndarray, 'OCV': np.ndarray, 'OCV_ch': np.ndarray, 'OCV_dch': np.ndarray}
    """
    # Select the charging (CC) / discharging areas and store as vectors
    c_cur_dch = df.c_cur[(df.step_type == 32)].values  # noqa
    c_vol_dch = df.c_vol[(df.step_type == 32)].values  # noqa

    c_cur_ch = df.c_cur[(df.step_type == 31)].values  # noqa
    c_vol_ch = df.c_vol[(df.step_type == 31)].values  # noqa

    # Time section division into soc-interval
    soc_dch = np.transpose(np.linspace(100, 0, (len(c_cur_dch))))
    soc_ch = np.transpose(np.linspace(0, 100, (len(c_cur_ch))))

    # Interpolation of the charging and discharging curve
    soc = np.linspace(0, 100, 1001)
    ocv_dch = griddata(soc_dch, c_vol_dch, soc, method="nearest")
    ocv_ch = griddata(soc_ch, c_vol_ch, soc, method="nearest")

    if enable_filter:
        window_size = int(len(ocv_dch) * 0.001)
        ocv_dch = savgol_filter(ocv_dch, window_size, polyorder)
        ocv_ch = savgol_filter(ocv_ch, window_size, polyorder)

    # Formation of the open circuit voltage curve (average)
    ocv = (ocv_dch + ocv_ch) / 2

    return {"SOC": soc.round(2), "OCV": ocv.round(4), "OCV_ch": ocv_ch.round(4), "OCV_dch": ocv_dch.round(4)}


def dva_curve(ocv: np.ndarray, soc: np.ndarray, average_window: int = 10, enable_spline: bool = False) -> np.ndarray:  # noqa
    """
    Differentiate the OCV-curve to get the DVA-curve

    :param ocv: Array of OCV values
    :type ocv: np.ndarray
    :param soc: Array of SoC values
    :type soc: np.ndarray
    :param average_window: int = 10: window for moving average
    :param enable_spline: Enable spline interpolation
    :type enable_spline: bool
    :return:
    """
    dva = moving_average(data=np.diff(ocv), window=average_window)

    if enable_spline:
        soc_vec = np.linspace(min(soc), max(soc), 1000)
        ocv_spline = CubicSpline(soc, ocv)
        ocv_vec = ocv_spline(soc_vec)
        dva = savgol_filter(np.diff(ocv_vec), 51, 3)

    return dva.round(8)


############################################################################################################
# DC INNER RESISTANCE (DCIR) EXTRACTION
############################################################################################################

def df_single_pulse(df: pd.DataFrame,
                    step_type: int = 5032,
                    extend_pulse: list = None,  # noqa
                    ) -> pd.DataFrame:
    """
    Extract single pulse from pulse test contained in df with extended time window before and
    after pulse start and stop.

    :param df: pd.DataFrame: df containing pulse measurements
    :type df: pd.DataFrame
    :param step_type: int = 5032: define the step_type to extract. Default: 50% SoC, -1C (discharge)
    :type step_type: int
    :param extend_pulse: list = None: extend pulse by [start_t_s, stop_t_s] seconds before and after step_type
    :type extend_pulse: list
    :return: df: pd.Dataframe: Dataframe containing just the wanted pulse
    """

    if extend_pulse is None:
        extend_pulse = [0, 0]

    # extract additional time before and after pulse
    t_before, t_after = extend_pulse

    # Limit the values to the range [0, 540] for t_before and [0, 30] for t_after
    t_before = max(0, min(t_before, 540))
    t_after = max(1, min(t_after, 30))

    # find start and stop index of pulse in df
    start_idx = df[df.step_type == step_type].index[0]
    stop_idx = df[df.step_type == step_type].index[-1]

    # find time of pulse start and stop in df
    pulse_start_time = df.loc[start_idx, "run_time"]
    pulse_end_time = df.loc[stop_idx, "run_time"]
    # shift start and stop index accordingly to extend_pulse values
    start_idx = df[df["run_time"] >= (pulse_start_time - t_before - 1)].index[0]
    stop_idx = df[df["run_time"] <= (pulse_end_time + t_after)].index[-1]

    # extract pulse with extended time window from df and reset index
    df = df.copy()[start_idx:stop_idx].reset_index(drop=True)

    # identify state changes by current changes
    current_changes_idx = abs(df.c_cur.diff()) > 1
    current_changes_idx = current_changes_idx.index[current_changes_idx == True]  # noqa

    if len(current_changes_idx) < 2:  # noqa
        print("Warning: Pulse contains less than 2 current changes. Pulse might be corrupted.")  # noqa

    return df


def rdc_extract(df: pd.DataFrame, ocv_fcns: dict, t: float = 10) -> list:
    """
    Extract RDC after t seconds from pulse in Ohm.
    OCV functions are needed for OCV correction. The functions are stored in a dictionary:
    ocv_fcns = {'f_ocv(capacity)': interpolate.interp1d(self.df_ocv_ref['capacity'].values,
                                                                 self.df_ocv_ref['mean'].values,
                                                                 kind='linear'),
                         'f_capacity(ocv)': interpolate.interp1d(self.df_ocv_ref['mean'].values,
                                                                 self.df_ocv_ref['capacity'].values,
                                                                 kind='linear')
                         }

    :param df: pd.DataFrame: results from df_single_pulse
    :type df: pd.DataFrame
    :param ocv_fcns: dict: dictionary containing the functions f_ocv(capacity) and f_capacity(ocv)
    :type ocv_fcns: dict
    :param t: int = 10: reference time  in seconds for RDC
    :type t: float
    :return: dict: {'RDC': float, 'I_pulse': float}: RDC[Ohm], I_pulse[A]
    """

    # find index triggered by a voltage change > abs(0.005V)
    c_vol0 = df.c_vol[0]
    c_vol_threshold = 0.005
    c_vol_trigger_index = ((df["c_vol"] - c_vol0).abs() >= c_vol_threshold).idxmax()

    # extract voltage before the pulse by taking the median of the values before voltage change
    c_vol0 = np.median(df.c_vol[:c_vol_trigger_index - 1])
    c_capa0 = ocv_fcns["f_capacity(ocv)"](c_vol0)

    # Extract the pulse only by identifying state changes by current changes
    current_changes_idx = abs(df.c_cur.diff()) > 1
    current_changes_idx = current_changes_idx.index[current_changes_idx == True]  # noqa

    pulse_idx_0 = current_changes_idx[0]
    pulse_idx_1 = current_changes_idx[-1]

    df = df.copy()[pulse_idx_0 - 1:pulse_idx_1].reset_index(drop=True)

    # reset time base
    df.run_time = df.run_time.copy() - df.run_time.copy()[0]

    # Check the pulse duration
    if df.iloc[-1]["run_time"] < t:
        return {"RDC": np.nan, "I_pulse": np.nan}

    # calculate charge throughput in Ah during pulse
    df["q"] = q_calc(df) / 3600

    # find index of value at t seconds
    time_end_index = np.where(df.run_time <= t)[0][-1]

    # find voltage at t seconds after pulse start including ocv correction
    c_capa1 = df.q[time_end_index] + c_capa0
    c_vol1_ocv_delta = ocv_fcns["f_ocv(capacity)"](c_capa1) - c_vol0
    c_vol1 = df.c_vol[time_end_index] - c_vol1_ocv_delta

    # calculate pulse current
    c_cur = np.median(df.c_cur[:time_end_index])

    # Check for CV phase at the end of the pulse
    if abs(df.iloc[time_end_index-10]["c_cur"]) < abs(0.9*c_cur):
        return {"RDC": np.nan, "I_pulse": np.nan}

    # calculate rdc
    rdc = abs((c_vol1 - c_vol0) / c_cur)

    return {"RDC": rdc, "I_pulse": c_cur}


############################################################################################################
# CALENDAR TIME EXTRACTION
############################################################################################################
def calendar_time(meta0: Path,
                  meta1: Path,
                  df_cu0: pd.DataFrame
                  ) -> float:
    """
    Find the calendar time of the measurement in the experimental campaign.
    This is done by comparing the date of the measurement with the date of the previous measurement.
    1. Find the date of the measurement.
    2. Find the date of the previous measurement.
    3. Extract the calendar time of previous CU and exCU measurements.
    4. Calculate the calendar time of the measurement
    5. Correct the calendar time by the number of days, when CUs and exCUs were performed.

    :param meta0: str: metadata of the previous measurement
    :param meta1: str: metadata of the current measurement
    :param df_cu0: pd.DataFrame: dataframe of the previous measurement

    :return: cal_time: float: calendar time of the aging phase before the CU and after the previous CU
    """
    # total time in days between the starts of the measurements
    total_time = (date_extract(meta1) - date_extract(meta0)).days
    # CU time in seconds to days
    cu_time = df_cu0.run_time.iloc[-1]/3600/24
    # calendar time in days
    cal_time: float = total_time - cu_time

    return cal_time


def date_extract(file: Path) -> datetime:
    # extract the date of measurement of the dataframe from the metadata
    with open(file) as f:
        content = f.readlines()
    for line in content:
        if "Measurement start date" in line:
            date = line.split(":")[1].strip()

    try:
        date = datetime.strptime(date, "%d.%m.%y").strftime("%d.%m.%y")  # noqa
    except ValueError:
        date = datetime.strptime(date, "%d.%m.%Y").strftime("%d.%m.%y")  # noqa

    return datetime.strptime(date, "%d.%m.%y").date()  # noqa


def cycle_time(df: pd.DataFrame) -> float:
    """
    Compute the time of cycling

    :param df: pd.DataFrame: cycling data
    :type df: pd.DataFrame
    :return: t: float: time in days
    """
    try:
        cyc_time = np.sum(df.run_time_diff.values) / (3600 * 24)  # t in days

    except Exception as e:
        print(f"Exception: cycle_time: {e}")  # noqa
        cyc_time = np.nan

    return cyc_time


############################################################################################################
# FULL EQUIVALENT CYCLE (FEC) EXTRACTION#
############################################################################################################
def df_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding df[run_time_diff] and extracting the cycles from cycle measurement DataFrame

    :param df: pd.DataFrame
    :return df_cycles: pd.DataFrame: DataFrame containing cycle data only
    """
    df["run_time_diff"] = df.run_time.copy().diff()
    df_cycles = df[((df.step_type == 41) | (df.step_type == 42)) & (abs(df.c_cur) > 0.05)].reset_index(drop=True)  # noqa

    return df_cycles


def fec_extract(df: pd.DataFrame, capa_ref: float = 4.9) -> float:
    """
    Calculate Full Equivalent Cycles (FEC) from cycling data.

    :param df: pd.DataFrame: cycling data
    :type df: pd.DataFrame
    :param capa_ref: float: capacity, the FECs are referenced to
    :type capa_ref: float
    :return: fec
    """
    fec: float

    df = df_cycle(df)

    try:
        q_pos_half = df[df.step_type == 41]  # noqa
        q_neg_half = df[df.step_type == 42]  # noqa

        q_val_pos = (q_pos_half.run_time_diff * q_pos_half.c_cur).fillna(0).values
        q_val_neg = (q_neg_half.run_time_diff * q_neg_half.c_cur).fillna(0).values

        fec_pos = abs(sum(q_val_pos)).round(2)
        fec_neg = abs(sum(q_val_neg)).round(2)

        fec_temp = fec_pos + fec_neg

        fec = fec_temp / (2 * capa_ref * 3600)

        return round(fec, 4)

    except Exception as e:
        print(f"Exception: fec_extract: {e}")  # noqa
        fec = np.nan
        return fec


############################################################################################################
# MOVING AVERAGE FILTER
############################################################################################################
def moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    smooth measurements by applying moving average filter

    :param window: int
    :param data: np.ndarray
    :return: np.ndarray

    """
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


############################################################################################################
# CELL CAPACITY OVER LIFETIME
############################################################################################################

class ValidationError(Exception):
    pass

INITIAL_AND_FINAL_CHECKUPS = {
    "ET_T10", "ET_T23", "ET_T45", "AT_T10", "AT_T23", "AT_T45"
}

def capacity_index(file_end, N, i):
    map = {  # noqa
        "ET_T10": 0,
        "ET_T23": 1,
        "ET_T45": 2,
        "AT_T10": N + 3,
        "AT_T23": N + 4,
        "AT_T45": N + 5
    }
    if file_end not in map:
        return i + 2
    return map[file_end]


def state_transition(file_end, N, i):
    # State transitions of file end:
    # ET_T10 -> ET_T23 -> ET_T45 -> ZYK <-> CU
    #                                |
    #                                > AT_T10 -> AT_T23 -> AT_T45
    map = {  # noqa
        "ET_T10": "ET_T23",
        "ET_T23": "ET_T45",
        "ET_T45": "ZYK",
        "AT_T10": "AT_T23",
        "AT_T23": "AT_T45",
        "AT_T45": None,  # finished
        "CU": "ZYK",
    }
    if file_end == "ZYK":
        next_i = i + 1
    else:
        next_i = i

    if file_end not in map:
        # file_end == "ZYK"
        if i == N:
            next_file_end = "AT_T10"
        elif i < N:
            next_file_end = "CU"
        else:
            raise ValidationError("i out of range!!")
        next_i = i + 1
    else:
        next_file_end = map[file_end]

    return next_file_end, next_i


class CyclingCapacity:
    def __init__(self, c_ch, c_dch, n_fe):
        self.c_ch = c_ch
        self.c_dch = c_dch
        self.n_fe = n_fe
        self.N = len(self.c_ch) - 6

    def checkup_index(self, file_end=None, i=None):
        num_None = sum([file_end is None, i is None])
        if num_None == 2:  #noqa
            raise ValueError("Either file_end or i must be specified. passed neither")
        if num_None == 0:
            raise ValueError("Either file_end or i must be specified. passed both")
        if file_end not in INITIAL_AND_FINAL_CHECKUPS:
            raise ValueError("file_end must be one of either from initial or final checkups i.e.:\n"
                             f"{INITIAL_AND_FINAL_CHECKUPS}")

        return capacity_index(file_end, self.N, i)

    # def c_ch_from_temp(self, T):
    #     if T not in TEMP_ORDER:
    #         return self.c_ch[3:-3]
    #     i = TEMP_ORDER.index(T)
    #     return np.r_[self.c_ch[i], self.c_ch[3:-3], self.c_ch[i-3]]

    # def c_dch_from_temp(self, T):
    #     if T not in TEMP_ORDER:
    #         return self.c_dch[3:-3]
    #     i = TEMP_ORDER.index(T)
    #     return np.r_[self.c_dch[i], self.c_dch[3:-3], self.c_dch[i-3]]

    def iter(self):
        # c_ch = self.c_ch_from_temp(T)
        # c_dch = self.c_dch_from_temp(T)
        # Just estimate capacity with mean... There must be a better way to utilize
        # this to reduce the variance?
        c_est = (self.c_ch + self.c_dch) / 2

        yield from zip(c_est, self.cumulative_cycles())

    def cumulative_cycles(self):
        # n_fe is the number of cycles between each checkup, take cumulative
        # sum to count total cycles.
        cumsum = np.cumsum(self.n_fe)
        # First 3 cycles are initial checkup. Need to add 2 more at the end for
        # final checkup.
        cum_cycles = np.r_[np.zeros(3, dtype=np.float32), cumsum,
                           cumsum[-1], cumsum[-1]]
        return cum_cycles

    def iter_both(self):

        yield from zip(self.c_ch, self.c_dch, self.cumulative_cycles())


def extract_cycling_capacity(cell_dir: Path, skip_files=None) -> np.ndarray:
    r"""
    Return the capacity and full-equivalent-cycles of the cell over test.

    Output:
        y = 1D ndarray of length (3N + 13) where N is the number of checkups
            As a block array:
            y = [
                C_ch
                C_dch
                n_FE
            ]
            where C_ch and C_dch are charge and discharge capacities at various
            checkups, and have format:
            C_[ch, dch] = [
                C_{0,T0},
                C_{0,T1},
                C_{0,T2},
                C_1,
                C_2,
                ...,
                C_N,
                C_{N+1,T0}
                C_{N+1,T1}
                C_{N+1,T2}
            ] \in \R^{N+6}
            where C_{[0, N+1], Ti} is the capacity at various temperatures during
            initial and final checkups, w/ T0 = 10, T1=23, T2=45.

            N_FE has length N + 1 and with N_{FE,i} measuring the number of full-equivalent
            cycles after checkup i, for i=0,...,N.

    Raises:
        ValidationError: if files are missing
        FeatureError: if features fail to extract
    """
    checkups = [file for file in cell_dir.glob("*CU.csv")
                if skip_files is None or file.name not in skip_files]

    N = len(checkups)
    csvs = [file for file in cell_dir.glob("*.csv")
            if skip_files is None or file.name not in skip_files]
    if len(csvs) != 2 * N + 7:
        raise ValidationError(f"Found incorrect number of files in directory {cell_dir}\n"
                              f"number checkups: {N} | num csvs: {len(csvs)}")

    c_ch = np.full(N + 6, np.nan)
    c_dch = np.full(N + 6, np.nan)
    n_fe = np.full(N + 1, np.nan)
    i = 0
    print(f"Num checkups: {N}")  # noqa

    file_end = "ET_T10"
    for file in sorted(csvs, key=lambda f: f.name):
        print(f"{file_end} | {i} | {file.name}")  # noqa
        if file_end is None:
            raise ValidationError("Extra CSVs????")

        if not str(file).endswith(file_end + ".csv"):
            raise ValidationError(f"Expected next file to end in {file_end}. Got {file!s}")
        df = import_datafile(file)
        if file_end == "ZYK":
            n_fe[i] = fec_extract(df)
        else:
            try:
                cap = capacity(df)
                j = capacity_index(file_end, N, i)
                c_ch[j] = cap["Q_ch"]
                c_dch[j] = cap["Q_dch"]
            except Exception as exc:
                print("Error extracting capacity:" + exc)  # noqa

        file_end, i = state_transition(file_end, N, i)

    return CyclingCapacity(c_ch, c_dch, n_fe)


TEMP_ORDER = [10, 23, 45]


class CapacityData(dict):

    def to_array(self):
        for cap in self.values():
            if cap is None:
                continue
            cap.N = len(cap.n_fe) - 1
        N_max = max([cap.N for cap in self.values() if cap is not None])
        print([cap.N for cap in self.values() if cap is not None])  # noqa

        dtype = np.dtype([
            ("uid", "U11"),
            ("N", np.int16),
            ("c_charge", np.float32, N_max + 6),
            ("c_discharge", np.float32, N_max + 6),
            ("n_fe", np.float32, N_max + 1),
        ])
        arr = np.full(len(self), np.nan, dtype=dtype)
        for i, (uid, cap) in enumerate(self.items()):
            arr[i]["uid"] = uid
            if cap is not None:
                arr[i]["N"] = cap.N
                arr[i]["c_charge"][:cap.N + 6] = cap.c_ch
                arr[i]["c_discharge"][:cap.N + 6] = cap.c_dch
                arr[i]["n_fe"][:cap.N + 1] = cap.n_fe
            else:
                arr[i]["N"] = -1
        return arr

    @classmethod
    def from_array(cls, array):
        out = {}
        for row in array:
            N = row["N"]
            if N == -1:
                out[row["uid"]] = None
            else:
                out[row["uid"]] = CyclingCapacity(
                    row["c_charge"][:N + 6],
                    row["c_discharge"][:N + 6],
                    row["n_fe"][:N + 1],
                )
        return cls(**out)


    @classmethod
    def load(cls, file):
        f = np.load(file)
        return cls.from_array(f["cycle_capacities"])

    def save(self, file):
        np.savez(file, cycle_capacities=self.to_array())
