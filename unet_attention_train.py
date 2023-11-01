import os
from models.unet_attention import *
from config import arguments


if __name__ == '__main__':
    appliance = arguments["appliance"]
    dataset = arguments["dataset"]

    if not os.path.isdir(f"./{dataset}_{appliance}"):
        print(f'Directory {dataset}_{appliance} does not exist, firstly run data_train_val_test.py')
    else:

        # Load data
        X_train = np.load(f'./{dataset}_{appliance}/x_train.npy')
        Y_train = np.load(f'./{dataset}_{appliance}/y_train.npy')
        state_train = np.load(f'./{dataset}_{appliance}/state_train.npy')
        X_val = np.load(f'./{dataset}_{appliance}/x_val.npy')
        Y_val = np.load(f'./{dataset}_{appliance}/y_val.npy')
        state_val = np.load(f'./{dataset}_{appliance}/state_val.npy')
        X_test = np.load(f'./{dataset}_{appliance}/x_test.npy')
        Y_test = np.load(f'./{dataset}_{appliance}/y_test.npy')
        state_test = np.load(f'./{dataset}_{appliance}/state_test.npy')
        min_max_scaling = np.load(f'./{dataset}_{appliance}/minmax.npy')

        dissagregator = UnetAttentionDissagregator(appliance, dataset, 50, 256)

        dissagregator.train(X_train, [Y_train, state_train], X_val, [Y_val, state_val])
        print('*************************')
        print('Training finished')
        print(X_test.shape)
        predicted_output_sequence, predicted_output_sequence_onoff = dissagregator.predict(X_test, Y_test, state_test)
        np.save(f'./{dataset}/prediction_{dissagregator.model_name}_whole_{appliance}.npy',
                predicted_output_sequence)
        np.save(f'./{dataset}/prediction_onoff_{dissagregator.model_name}_whole_{appliance}.npy',
                predicted_output_sequence_onoff)
        predicted_output_sequence *= min_max_scaling[1, 1] - min_max_scaling[1, 0]  # max_appliance - min_appliance
        # Clip negative values to zero
        predicted_output_sequence[predicted_output_sequence < 0] = 0.0

        dissagregator.compute_metrics(predicted_output_sequence, Y_test, predicted_output_sequence_onoff, state_test)
        print('*************************')
        print(dissagregator.model_name)
        print('Prediction finished')

