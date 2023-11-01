
import pandas as pd
import argparse
import numpy as np
from models.utils import normalize_data
import os
from config import appliances, arguments, houses_refit, houses_ukdale


def delete_rows_with_zeros(aggregate, appliance, state, dop, threshold=None):
    if dop == 0:
        aggregate_drop = aggregate
        appliance_drop = appliance
        state_drop = state
    else:
        rp = np.random.RandomState(42)
        if threshold is not None:
            appliance = np.where(appliance > threshold, appliance, 0)
        idx = np.where(~appliance.any(axis=1))[0]
        print('****************************')
        print(idx.shape)
        print(appliance.shape)
        idx_to_drop = rp.choice(idx, size=round(dop*len(idx)), replace=False)
        print('****************************')
        print(idx_to_drop)
        aggregate_drop = np.delete(aggregate, idx_to_drop, axis=0)
        appliance_drop = np.delete(appliance, idx_to_drop, axis=0)
        state_drop = np.delete(state, idx_to_drop, axis=0)

    return aggregate_drop, appliance_drop, state_drop


def load_train_set(dop, house_dict, type_dis="seq2seq", **appliance):
    """
    Loads house 2, 7, 9, as the training set
    :param type_dis:
    :param dop: drop out percentage
    :param appliance: Info about name, window_size and stride
    :return: X_train, y_train and array with min_max for scaling
    """

    train = np.empty((0, 2))  # 2 columns, aggregate and appliance
    states = np.empty(0)
    # TODO check Sequence-to-Sequence Load Disaggregation Using Multiscale Residual Neural Network
    #  how they filter training samples page 5

    for house_no in appliance["houses"]:

        start_date = house_dict[appliance["name"]]['dates'][f'house{house_no}'][0]
        end_date = house_dict[appliance["name"]]['dates'][f'house{house_no}'][1]

        train_df = pd.read_csv(f'./processed_{appliance["dataset"]}_{appliance["name"]}/house_{house_no}.csv').set_index('Time', inplace=False).loc[start_date: end_date]# Just to reduce size
        states_df = pd.read_csv(f'./states_{appliance["dataset"]}_{appliance["name"]}/house_{house_no}.csv').set_index('Time', inplace=False).loc[start_date: end_date]

        train = np.append(train, train_df[['Aggregate', appliance["name"]]].to_numpy(), axis=0)
        states = np.append(states, states_df[appliance["name"]].to_numpy(), axis=0)

        # Adding some zeros, to not mix data between houses
        train = np.append(train, np.zeros((appliance["window_size"] + 1, 2)), axis=0)
        states = np.append(states, np.zeros((appliance["window_size"] + 1, )), axis=0)

    # Scale data based on test set
    min_aggregate = np.min(train[:, 0])
    max_aggregate = np.max(train[:, 0])
    min_appliance = np.min(train[:, 1])
    max_appliance = np.max(train[:, 1])

    normalize_data(train[:, 0], min_aggregate, max_aggregate)
    normalize_data(train[:, 1], min_appliance, max_appliance)

    if appliance["name"] == "Dishwasher" and appliance["dataset"] == "ukdale":
        threshold = 5 / max_appliance  # 5 = on power threshold / 2

    else:
        threshold = None
    # Make the rolling window with a stride
    print(threshold)
    if type_dis == "seq2seq":
        x_train_all = np.lib.stride_tricks.sliding_window_view(train[:, 0], appliance["window_size"])[
                      ::2, :]

        y_train_all = np.lib.stride_tricks.sliding_window_view(train[:, 1], appliance["window_size"])[::2, :]
        states_all = np.lib.stride_tricks.sliding_window_view(states, appliance["window_size"])[::2, :]

        x_train, y_train, states_train = delete_rows_with_zeros(x_train_all, y_train_all,states_all ,dop=dop,
                                                                threshold=threshold)

    elif type_dis == "seq2point":  # stride == 1
        x_train_all = np.lib.stride_tricks.sliding_window_view(train[:, 0], appliance["window_size"])[::, :]
        y_train_all= train[window_size//2: window_size//2 + x_train_all.shape[0], 1]
        states_all = states[window_size//2: window_size//2 + x_train_all.shape[0]]
        x_train, y_train, states_train = delete_rows_with_zeros(x_train_all, y_train_all, states_all, dop=dop)

    return x_train, y_train, states_train, np.array([[min_aggregate, max_aggregate], [min_appliance, max_appliance]])


def load_validation_set(min_max, house_dict, type="seq2seq", **appliance):

    start_date = house_dict[appliance["name"]]['dates'][f'house{appliance["houses"]}'][0]
    end_date = house_dict[appliance["name"]]['dates'][f'house{appliance["houses"]}'][1]
    val_df = pd.read_csv(f'./processed_{appliance["dataset"]}_{appliance["name"]}/house_{appliance["houses"]}.csv').set_index('Time', inplace=False).loc[start_date: end_date]
    state_df = pd.read_csv(f'./states_{appliance["dataset"]}_{appliance["name"]}/house_{appliance["houses"]}.csv').set_index('Time', inplace=False).loc[start_date: end_date]
    val = val_df[['Aggregate', appliance["name"]]].to_numpy()
    state = state_df[appliance["name"]].to_numpy()

    normalize_data(val[:, 0], min_max[0, 0], min_max[0, 1])
    normalize_data(val[:, 1], min_max[1, 0], min_max[1, 1])

    if type == "seq2seq":
        x_val = np.lib.stride_tricks.sliding_window_view(val[:, 0], appliance["window_size"])[::appliance["stride"], :]
        y_val = np.lib.stride_tricks.sliding_window_view(val[:, 1], appliance["window_size"])[::appliance["stride"], :]
        states_val = np.lib.stride_tricks.sliding_window_view(state , appliance["window_size"])[::appliance["stride"], :]
    elif type == "seq2point":
        x_val = np.lib.stride_tricks.sliding_window_view(val[:, 0], appliance["window_size"])[::, :]
        y_val = val[window_size//2: window_size//2 + x_val.shape[0]:, 1]
        states_val = state[window_size//2: window_size//2 + x_val.shape[0]]

    return x_val, y_val, states_val


def load_test_set(min_max, house_dict, type="seq2seq",**appliance):

    start_date = house_dict[appliance["name"]]['dates'][f'house{appliance["houses"]}'][0]
    end_date = house_dict[appliance["name"]]['dates'][f'house{appliance["houses"]}'][1]
    test_df = pd.read_csv(f'./processed_{appliance["dataset"]}_{appliance["name"]}/house_{appliance["houses"]}.csv').set_index('Time', inplace=False).loc[start_date: end_date]
    state_df = pd.read_csv(f'./states_{appliance["dataset"]}_{appliance["name"]}/house_{appliance["houses"]}.csv').set_index('Time', inplace=False).loc[start_date: end_date]

    test = test_df[['Aggregate', appliance["name"]]].to_numpy()
    states_test = state_df[appliance["name"]].to_numpy()

    normalize_data(test[:, 0], min_max[0, 0], min_max[0, 1])
    if type == "seq2seq":

        x_test = test[:, 0]
        y_test = test[:, 1]  # Target as sequence

    elif type == "seq2point":
        x_test = np.lib.stride_tricks.sliding_window_view(test[:, 0], appliance["window_size"])[::, :]
        y_test = test[window_size//2: window_size//2 + x_test.shape[0], 1]  # Target as sequence
        states_test = states_test[window_size//2: window_size//2 + x_test.shape[0]]


    return x_test, y_test, states_test


def percentage_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


if __name__ == '__main__':

    type_dis = arguments["type"]
    print(type_dis)
    appliance_name = arguments["appliance"]
    dataset = arguments["dataset"]
    dop = arguments["dop"]

    if dataset == "ukdale":
        house_dict = houses_ukdale
    elif dataset == "refit":
        house_dict = houses_refit
    else:
        print("Dataset must be ukdale or refit")
        exit(-1)

    window_size = appliances[appliance_name]["window_size"]
    stride = appliances[appliance_name]["stride"]
    house_list = np.array(list(house_dict[appliance_name]["houses"].keys()))
    print(house_list)

    # Profuce train/validation/test set
    X_train, Y_train, state_train, min_max_array = load_train_set(dop, house_dict, type_dis,dataset=dataset, name=appliance_name, window_size=window_size, stride=stride, houses= house_list[:-2])
    X_val, Y_val, state_val = load_validation_set(min_max_array, house_dict, type_dis,dataset=dataset, name=appliance_name, window_size=window_size, stride=stride, houses= house_list[-2])
    X_test, Y_test, state_test = load_test_set(min_max_array, house_dict,type_dis,dataset=dataset, name=appliance_name, window_size=window_size, stride=stride, houses= house_list[-1])

    # Save .npy arrays for training-test
    if not os.path.isdir(f"./{dataset}_{appliance_name}"):
        os.makedirs(f"./{dataset}_{appliance_name}")
    np.save(f'./{dataset}_{appliance_name}/x_train.npy', X_train)
    np.save(f'./{dataset}_{appliance_name}/y_train.npy', Y_train)
    np.save(f'./{dataset}_{appliance_name}/state_train.npy', state_train)
    np.save(f'./{dataset}_{appliance_name}/x_val.npy', X_val)
    np.save(f'./{dataset}_{appliance_name}/y_val.npy', Y_val)
    np.save(f'./{dataset}_{appliance_name}/state_val.npy', state_val)
    np.save(f'./{dataset}_{appliance_name}/x_test.npy', X_test)
    np.save(f'./{dataset}_{appliance_name}/y_test.npy', Y_test)
    np.save(f'./{dataset}_{appliance_name}/state_test.npy', state_test)
    np.save(f'./{dataset}_{appliance_name}/minmax.npy', min_max_array)


