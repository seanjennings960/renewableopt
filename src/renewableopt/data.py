from pathlib import Path
import numpy as np

# Remove these hard-coded directories?
DATASET_DIR = Path(
    "/home/sean/data/studies/data_current_classes/renewables_course/Project2/datasets")
EXCEL_FILENAME = DATASET_DIR / "project2_load_profile.csv"
NPZ_LOAD_FILE = DATASET_DIR / "project_2_load_profile.npz"
GEOJSON_FILE = DATASET_DIR / "gis" / "SD_county_jurisdictions.geojson"
NPZ_DNI_FILE = DATASET_DIR / 'dni.npz'


def load_load_data():
    return np.load(NPZ_LOAD_FILE)


def load_dni_data():
    return np.load(NPZ_DNI_FILE)


def match_times(load, dni):
    # Hackily convert to min...
    load_t = total_seconds(load['time'] - load['time'][0]) / 60  # min
    dni_t = total_seconds(dni['time'] - dni['time'][0]) / 60  # min
    # Time-zone correction.
    dni_t -= 7 * 60  # UTC -> PDT

    # perform interpolation and then pull data from dict to individual
    # arrays. Should we return a Dict/Dataframe obj again at the end??
    dnis = np.interp(load_t, dni_t, np.mean(dni['dnis'], axis=1))
    load_power = load["power"]
    # Convert DNI to solar per unit of installed generation capactiy,
    # assuming 100% efficiency and rated at 1000 W/m^2.
    solar_pu = dnis / 1000
    return load_t, load_power, solar_pu


def total_seconds(timedelta):
    return timedelta.astype("timedelta64[s]").astype(float)


def filter_days(start_day, num_days, load_t, load, solar_pu):
    # Well this could be cleaner, but works...
    min_per_day = 60 * 24
    t_start = start_day * min_per_day
    t_end = t_start + num_days * min_per_day  # 1day * 60 min/hr * 24hr/day
    times = np.logical_and(load_t <= t_end, load_t >= t_start)
    load_t = load_t[times]
    load = load[times]
    solar_pu = solar_pu[times]
    return load_t, load, solar_pu


def load_by_day(start_day, num_days):
    """
    Returns time, load, and generation for given day.

    Performs 1d linear interpolation to pair solar DNI data
    to load aat each given time.

    Arguments:
        start_day (int): number of day within year to start from
        num_days (int): number of days of year to load.

    Return
    """
    load = load_load_data()
    dni = load_dni_data()
    load_t, load, solar_pu = match_times(load, dni)
    return filter_days(start_day, num_days, load_t, load, solar_pu)