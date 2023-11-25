from pathlib import Path

import numpy as np

# Remove these hard-coded directories?
DATASET_DIR = Path(
    "/home/sean/data/studies/data_current_classes/renewables_course/Project2/datasets")
EXCEL_FILENAME = DATASET_DIR / "project2_load_profile.csv"
NPZ_LOAD_FILE = DATASET_DIR / "project_2_load_profile.npz"
GEOJSON_FILE = DATASET_DIR / "gis" / "SD_county_jurisdictions.geojson"
NPZ_DNI_FILE = DATASET_DIR / "dni.npz"


def load_load_data():
    return np.load(NPZ_LOAD_FILE)


def load_solar_data(load_t):
    dnis = np.load(NPZ_DNI_FILE)
        # Take mean over DNI locations, then
        # convert DNI to solar per unit of installed generation capactiy,
        # assuming 100% efficiency and rated at 1000 W/m^2.
    solar_pu = np.mean(dnis["dnis"], axis=1) / 1000
    return match_solar_times(load_t, dnis["time"], solar_pu)


GENERATION_LOADER = {
    "solar": load_solar_data,
}


def match_solar_times(load_t, time, solar_pu):
    # Expects generation is Indexable with keys:
    #    "time": np.datetime64,
    #    "pu": float
    # Hackily convert to min...
    gen_t = total_seconds(time - time[0]) / 60  # min
    # Time-zone correction.
    gen_t -= 7 * 60  # UTC -> PDT

    # perform interpolation and then pull data from dict to individual
    # arrays. Should we return a Dict/Dataframe obj again at the end??
    gen_out = np.interp(load_t, gen_t, solar_pu)
    return gen_out


def total_seconds(timedelta):
    return timedelta.astype("timedelta64[s]").astype(float)


def filter_days(start_day, num_days, load_t, load, generation_pu):
    # Well this could be cleaner, but works...
    min_per_day = 60 * 24
    t_start = start_day * min_per_day
    t_end = t_start + num_days * min_per_day  # 1day * 60 min/hr * 24hr/day
    times = np.logical_and(load_t <= t_end, load_t >= t_start)
    load_t = load_t[times]
    load = load[times]
    generation_pu = generation_pu[times]
    return load_t, load, generation_pu


def load_by_day(start_day, num_days, sources=None):
    """
    Returns time, load, and generation for given day.

    Performs 1d linear interpolation to pair solar DNI data
    to load aat each given time.

    Arguments:
        start_day (int): number of day within year to start from
        num_days (int): number of days of year to load.

    Return:
        Tuple of
            time (shape (T,))
            load (shape (T,))
            generation (shape (T, G) or shape (T,) if G=1)
    """
    load = load_load_data()
    if sources is None:
        # Default to just solar for backwards compatibility.
        sources = ["solar"]
    known_sources = set(GENERATION_LOADER.keys())
    if not set(sources).issubset(known_sources):
        raise ValueError(f"Found unexpected generation sources: {set(sources) - known_sources}. "
                         f"Supported sources: {known_sources}")
    load_t = total_seconds(load["time"] - load["time"][0]) / 60  # min
    generation_pu = np.zeros((load_t.shape[0], len(sources)), np.float64)
    for i, source in enumerate(sources):
        generation_pu[:, i] = GENERATION_LOADER[source](load_t)
    return filter_days(
        start_day, num_days,
        load_t, load["power"],
        generation_pu.squeeze())
