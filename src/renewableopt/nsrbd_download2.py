import requests
import pandas as pd
import urllib.parse
import time

with open("API_KEY", "r") as f:
    API_KEY = f.readline().strip()
EMAIL = "seje7952@colorado.edu"
BASE_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-2-2-download.json?"
POINTS = [
'217101,217102,217103,217104,217105,217106,217107,217779,217780,217781,217782,217783,217784,217785,217786,217787,217788,217789,217790,217791,217792,218465,218466,218467,218468,218469,218470,218471,218472,218473,218474,218475,218476,218477,218478,219153,219154,219155,219156,219157,219158,219159,219160,219161,219162,219163,219164,219165,219166,219841,219842,219843,219844,219845,219846,219847,219848,219849,219850,219851,219852,219853,219854,220530,220531,220532,220533,220534,220535,220536,220537,220538,220539,220540,220541,220542,220543,221224,221225,221226,221227,221228,221229,221230,221231,221232,221233,221234,221235,221236,221237,221920,221921,221922,221923,221924,221925,221926,221927,221928,221929,221930,221931,221932,221933,222618,222619,222620,222621,222622,222623,222624,222625,222626,222627,222628,222629,222630,222631,223316,223317,223318,223319,223320,223321,223322,223323,223324,223325,223326,223327,223328,223329,224015,224016,224017,224018,224019,224020,224021,224022,224023,224024,224025,224026,224027,224028,224716,224717,224718,224719,224720,224721,224722,224723,224724,224725,224726,224727,224728,224729,225421,225422,225423,225424,225425,225426,225427,225428,225429,225430,225431,225432,225433,225434,226126,226127,226128,226129,226130,226131,226132,226133,226134,226135,226136,226137,226138,226139,226831,226832,226833,226834,226835,226836,226837,226838,226839,226840,226841',
'226842,226843,226844,227538,227539,227540,227541,227542,227543,227544,227545,227546,227547,227548,227549,227550,227551,228250,228251,228252,228253,228254,228255,228256,228257,228258,228259,228260,228261,228262,228263,228966,228967,228968,228969,228970,228971,228972,228973,228974,228975,228976,228977,228978,228979,229683,229684,229685,229686,229687,229688,229689,229690,229691,229692,229693,229694,229695,229696,230401,230402,230403,230404,230405,230406,230407,230408,230409,230410,230411,230412,230413,230414,231120,231121,231122,231123,231124,231125,231126,231127,231128,231129,231130,231131,231132,231133,231840,231841,231842,231843,231844,231845,231846,231847,231848,231849,231850,231851,231852,231853,232561,232562,232563,232564,232565,232566,232567,232568,232569,232570,232571,232572,232573,232574,233283,233284,233285,233286,233287,233288,233289,233290,233291,233292,233293,233294,233295,233296,234006,234007,234008,234009,234010,234011,234012,234013,234014,234015,234016,234017,234018,234019,234730,234731,234732,234733,234734,234735,234736,234737,234738,234739,234740,234741,234742,234743'
]

def main():
    input_data = {
        'attributes': 'ghi',
        'interval': '30',
        'include_leap_day': 'true',
        'api_key': API_KEY,
        'email': EMAIL,
    }
    for name in ['2012']:
        print(f"Processing name: {name}")
        for id, location_ids in enumerate(POINTS):
            input_data['names'] = [name]
            input_data['location_ids'] = location_ids
            print(f'Making request for point group {id + 1} of {len(POINTS)}...')

            if '.csv' in BASE_URL:
                url = BASE_URL + urllib.parse.urlencode(input_data, True)
                # Note: CSV format is only supported for single point requests
                # Suggest that you might append to a larger data frame
                data = pd.read_csv(url)
                print(f'Response data (you should replace this print statement with your processing): {data}')
                # You can use the following code to write it to a file
                # data.to_csv('SingleBigDataPoint.csv')
            else:
                headers = {
                  'x-api-key': API_KEY
                }
                data = get_response_json_and_handle_errors(requests.post(BASE_URL, input_data, headers=headers))
                download_url = data['outputs']['downloadUrl']
                # You can do with what you will the download url
                print(data['outputs']['message'])
                print(f"Data can be downloaded from this url when ready: {download_url}")

                # Delay for 1 second to prevent rate limiting
                time.sleep(1)
            print(f'Processed')


def get_response_json_and_handle_errors(response: requests.Response) -> dict:
    """Takes the given response and handles any errors, along with providing
    the resulting json

    Parameters
    ----------
    response : requests.Response
        The response object

    Returns
    -------
    dict
        The resulting json
    """
    if response.status_code != 200:
        print(f"An error has occurred with the server or the request. The request response code/status: {response.status_code} {response.reason}")
        print(f"The response body: {response.text}")
        exit(1)

    try:
        response_json = response.json()
    except:
        print(f"The response couldn't be parsed as JSON, likely an issue with the server, here is the text: {response.text}")
        exit(1)

    if len(response_json['errors']) > 0:
        errors = '\n'.join(response_json['errors'])
        print(f"The request errored out, here are the errors: {errors}")
        exit(1)
    return response_json

if __name__ == "__main__":
    print('hello for now:')
    print(API_KEY)
    print(f"We have: {len(POINTS)} points")
    main()