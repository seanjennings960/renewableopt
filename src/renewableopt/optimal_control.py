from initial_data_wrangling import load_geojson


def main():
    geo = load_geojson()
    print(geo)


if __name__ == '__main__':
    main()
