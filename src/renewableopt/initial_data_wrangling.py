from shapely.geometry import Point
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from data import total_seconds

# DATASET_DIR = Path(
#     "/home/sean/data/power_systems/renewables_course/Project2/datasets")
DATASET_DIR = Path(
    "/home/sean/data/studies/data_current_classes/renewables_course/Project2/datasets")
EXCEL_FILENAME = DATASET_DIR / "project2_load_profile.csv"
NPZ_FILE = DATASET_DIR / "project_2_load_profile.npz"
GEOJSON_FILE = DATASET_DIR / "gis" / "SD_county_jurisdictions.geojson"
NPZ_DNI_FILE = DATASET_DIR / 'dni.npz'
CSV_GHI_FILE = DATASET_DIR / 'ghi.csv'

def subtract_10_year(dt: str):
    mdy = dt.split(" ")[0]
    m, d, y = mdy.split("/")
    # Subtract 10 years to make it 2012.... a year which has a leap day
    new_y = int(y) - 10
    new_mdy = "/".join([m, d, str(new_y)])

    rest = dt.split(" ")[1:]
    new_str = " ".join([new_mdy] + rest)

    formats = ["%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p"]
    errs = []
    for format in formats:
        try:
            return datetime.strptime(new_str, format)
        except ValueError as err:
            errs.append(err)
            # We ignore any par
            pass
    else:
        raise ValueError(f"Non of expected formats worked. errors: {errs}")

def to_npdatetime(dt):
    return pd.Timestamp(dt).to_datetime64()

def parse_data(data):
    """
    Parse data from messy excel "Project 2 - Load Profile.xlsx" file.

    Returns 2 numpy arrays of datetime [datetime[us] dtype] and power [float dtype]

    In excel file dates are reported in 2022. However, since the data is "coerced"
    into the year 2022 from a 2012 dataset, it does not play nice with datetime
    objects: 2012 is a leap year while 2022 is not. To work around this, we
    subtract 10 years from each datetime so that the year is returned to 2012.
    """
    n = len(data) - 1
    times = []
    power = np.full(n, np.nan, dtype=float)

    for i, line in enumerate(data[1:]):
        dt, p = line.split(",")
        dt = subtract_10_year(dt)
        dt = pd.Timestamp(dt).to_datetime64()

        times.append(dt)
        power[i] = p
    return np.array(times), power


def convert_excel_to_npz(excel_filename, npz_filename):
    with open(excel_filename, 'r') as f:
        data = f.readlines()
    time, power = parse_data(data)
    # Remove dates from 2021 that were for some reason included in load dataset
    start_date = datetime(2012, 1, 1) - timedelta(seconds=1)
    valid_indices = np.where(time > start_date)
    time = time[valid_indices]
    power = power[valid_indices]
    print('saving to npz')
    np.savez_compressed(NPZ_FILE, time=time, power=power)
    print('Successfully saved npz')


def check_datetimes(time):
    deltas = np.diff(time)
    sorted_deltas = np.sort(deltas)
    sorted_deltas_min = sorted_deltas / 1e6 / 60
    # assert np.all(sorted_deltas_min == 30)
    print(f'5 max deltas: {sorted_deltas_min[-5:]}')
    print(f'5 min deltas: {sorted_deltas_min[:5]}')

def plot_load(time, power):
    plt.figure()
    plt.plot(time, power)
    plt.show()

def load_geojson():
    geo = geopandas.read_file(GEOJSON_FILE)
    print(len(geo))
    print(geo.columns)
    # geo.explore()
    # geo['centroid'] = geo.centroid
    # print(geo['centroid'])
    # plt.show()
    return geo


def load_solar_data():
    dni_path = DATASET_DIR / 'ghi_data'
    time = None
    dnis = []
    coords = []
    for path in dni_path.iterdir():
        if not path.suffix == '.csv':
            continue
        path_components = path.name.split("_")
        lat = float(path_components[1])
        lon = float(path_components[2])
        print(f"Parsing {path.name}")
        data = pd.read_csv(path, header=2)
        if time is None:
            time = np.array([to_npdatetime(datetime(
                    data["Year"].loc[i],
                    data["Month"].loc[i],
                    data["Day"].loc[i],
                    data["Hour"].loc[i],
                    data["Minute"].loc[i],
                ))
                for i in range(0, len(data))]
            )
        dnis.append(np.array(data["GHI"]))
        coords.append((lat, lon))
    # Create multidim array
    print('Time dtype:', time.dtype)
    dnis = np.hstack([dni[:, np.newaxis] for dni in dnis])
    return time, dnis, coords
    # print('first path:', first_path.name)
    # print(lat)
    # print(lon)
    # print(data)
    # print(data.columns)

def convert_dni_to_npz():
    time, dnis, coords = load_solar_data()    
    print("Saving to NPZ")
    np.savez_compressed(NPZ_DNI_FILE, time=time, dnis=dnis, coords=coords)
    print("Successfully saved")

def convert_solar_to_csv():
    time, ghi, _ = load_solar_data()
    print("Saving to txt")
    time_min = total_seconds(time - time[0]) / 60
    np.savetxt(CSV_GHI_FILE, np.c_[time_min, np.mean(ghi, axis=-1)], delimiter=',')
    print("Successfully saved")



def plot_load_vs_solar():
    data = np.load(NPZ_FILE)
    time, power = data['time'], data['power']

    data = np.load(NPZ_DNI_FILE, allow_pickle=False)
    time_dni = data['time']
    print(data['coords'].shape)
    print(data['dnis'].shape)
    print(data['time'].dtype)
    mean_dni = np.mean(data['dnis'], axis=1)
    plt.figure()
    plt.plot(time, power)
    plt.plot([time[0], time[-1]], [np.mean(power), np.mean(power)])
    # plt.plot(time_dni, mean_dni)
    # plt.legend()
    plt.show()


def coords_to_geo():
    dni_data = np.load(NPZ_DNI_FILE)
    dni = np.array(dni_data["dnis"])
    coords = dni_data['coords']
    geo = load_geojson()
    print(len(geo))
    dni_by_area = np.full(len(geo), np.nan)
    print(dni.shape)
    for coord in coords:
        lat_long = Point(*coord[::-1])
        containing_region = np.where(geo.contains(lat_long))
        if len(containing_region) == 0:
            continue
        if len(containing_region) != 1:
            raise ValueError("Multiple regions??")
        i = containing_region[0]
        dni_by_area[containing_region] = np.mean(dni[:, i])
        # np.where(geo)
    geo.plot(dni_by_area, legend=True)
    # plt.plot(coords[:, 1], coords[:, 0], 'ro')
    plt.legend()


    plt.show()


def total_load():
    loads = np.load(NPZ_FILE)
    total_load_energy = np.sum(loads['power']) * 5/60
    dni_data = np.load(NPZ_DNI_FILE)
    dni_data = np.array(dni_data['dnis'])
    geo = load_geojson()
    # total_area = geo.to_crs('crs').area
    total_area = 1.1e10
    average_over_total_county = np.mean(dni_data, axis=1)
    print(average_over_total_county.shape)
    print('total area:', total_area)
    energy_per_area = np.sum(average_over_total_county) * 30/60
    total_solar_energy = total_area * energy_per_area / 1e6
    print("total load (MWh): ", total_load_energy)
    print("total load (MWh): ", total_load_energy)
    print("Total solar_energy (MWh): ", total_solar_energy)
    geo.plot(geo.area, legend=True)
    plt.show()



def main():
    # coords_to_geo()
    # total_load()
    # print("DATASET_DIR exists:", DATASET_DIR.exists())
    convert_solar_to_csv()
    # convert_excel_to_npz(EXCEL_FILENAME , NPZ_FILE)
    # check_datetimes(data['time'])
    # plot_load(time, power)
    # load_geojson()
    # check_datetimes(time)
    # plot_load_vs_solar()


    



if __name__ == "__main__":
    main()