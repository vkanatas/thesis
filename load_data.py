import pandas as pd
import numpy as np
import os
from config import appliances, houses_refit, houses_ukdale, arguments



# For REFIT dataset
def load_raw_refit_data(appliance, house_no, appliance_col) -> pd.DataFrame:

    # Loading the house we need

    raw_house = pd.read_csv(f'./data/RAW_House{house_no}_Part1.csv')

    # Drop columns we don't need and resample every 8 seconds.

    raw_house.drop(columns=['Time'], inplace=True)
    raw_house['Time'] = pd.to_datetime(raw_house['Unix'], unit ='s')
    raw_house.set_index('Time', inplace=True)
    raw_house = raw_house.rename(columns={appliance_col: appliance}).loc[:, ['Aggregate', appliance]]
    raw_house = raw_house.resample('8s').mean().fillna(method='ffill', limit=15)  # 15 * 8 sec = 2min ffil
    raw_house = raw_house.fillna(value=0)  # The rest NaN values ( not in 2 min interval from last measurement)

    return raw_house


def load_raw_ukdale_data(appliance, house_no, channel):

    # channel 1 is for aggregate data
    df_main = pd.read_table('data/ukdale/house_' + str(house_no) + '/' + 'channel_' +
                            str(1) + '.dat',
                            sep="\s+",
                            usecols=[0, 1],
                            names=['Time','Aggregate'],
                            dtype={'time': str},
                            )
    df_main['Time'] = pd.to_datetime(df_main['Time'], unit='s')
    df_main.set_index('Time', inplace=True)

    df_app = pd.read_table('data/ukdale/house_' + str(house_no) + '/' + 'channel_' +
                           str(channel) + '.dat',
                           sep="\s+",
                           usecols=[0, 1],
                           names=['Time', appliance], #TODO change it with variable applinace
                           dtype={'time': str},
                           )

    df_app['Time'] = pd.to_datetime(df_app['Time'], unit='s')
    df_app.set_index('Time', inplace=True)

    raw_house = df_main.join(df_app, how='outer')
    # Not aligned
    raw_house = raw_house.resample('6s').mean().fillna(method='ffill', limit=20)  # 20 * 6 sec = 2min ffil
    raw_house = raw_house.fillna(value=0)  # The rest NaN values ( not in 2 min interval from last measurement)

    print(raw_house)

    return raw_house


def outlier_power(appliance, house_outlier) -> pd.DataFrame:

    house_new = pd.DataFrame()
    house_new['Aggregate'] = house_outlier['Aggregate']

    house_new[appliance] = house_outlier[appliance].where(house_outlier[appliance] < appliances[appliance]['max_threshold']).fillna(method='ffill')

    return house_new


def appliance_state(appliance, house_to_state) -> pd.DataFrame:

    app_state = pd.DataFrame()
    app_state['Time'] = house_to_state.index

    app_state[appliance] = np.where(house_to_state[appliance] > appliances[appliance]['on_power'], 1, 0)

    app_state.set_index('Time', inplace=True)
    return app_state


def process_refit_data(appliance):

    for house_no, appliance_col in houses_refit[appliance]["houses"].items():
        resampled_house = load_raw_refit_data(appliance, house_no, appliance_col)
        out_house = outlier_power(appliance, resampled_house)

        # Extract appliance states of the house
        states = appliance_state(appliance, out_house)

        # Create directories if they do not exist
        if not os.path.isdir(f"./processed_refit_{appliance}"):
            os.makedirs(f"./processed_refit_{appliance}")
        if not os.path.isdir(f"./states_refit_{appliance}"):
            os.makedirs(f"./states_refit_{appliance}")
        # Save house and state

        out_house.to_csv(f'./processed_refit_{appliance}/house_{house_no}.csv', index=True)
        states.to_csv(f'./states_refit_{appliance}/house_{house_no}.csv', index=True)


def process_ukdale_data(appliance):

    for house_no, channel in houses_ukdale[appliance]["houses"].items():
        resampled_house = load_raw_ukdale_data(appliance, house_no, channel)
        out_house = outlier_power(appliance, resampled_house)

        # Extract appliance states of the house
        states = appliance_state(appliance, out_house)

        # Create directories if they do not exist
        if not os.path.isdir(f"./processed_ukdale_{appliance}"):
            os.makedirs(f"./processed_ukdale_{appliance}")
        if not os.path.isdir(f"./states_ukdale_{appliance}"):
            os.makedirs(f"./states_ukdale_{appliance}")
        # Save house and state

        out_house.to_csv(f'./processed_ukdale_{appliance}/house_{house_no}.csv', index=True)
        states.to_csv(f'./states_ukdale_{appliance}/house_{house_no}.csv', index=True)



if __name__ == '__main__':
    appliance = arguments["appliance"]
    dataset = arguments["dataset"]
    if dataset == "ukdale":
        process_ukdale_data(appliance)
    elif dataset == "refit":
        process_refit_data(appliance)
    else:
        print("Not supported dataset, must be ukdale or refit")






