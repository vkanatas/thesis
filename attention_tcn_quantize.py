import numpy as np
from models.attention_tcn import *
from config import arguments
import argparse
from codecarbon import EmissionsTracker

if __name__ == '__main__':

    appliance = arguments["appliance"]
    dataset = arguments["dataset"]
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--percentage', type=float, required=True,
                        help='sparsity percentage')
    args = parser.parse_args()
    percentage = args.percentage

    if not os.path.isdir(f"./{dataset}_{appliance}"):
        print(f'Directory {dataset}_{appliance} does not exist, firstly run data_train_val_test.py')
    else:

        X_test = np.load(f'./{dataset}_{appliance}/x_test.npy')
        print(len(X_test))
        Y_test = np.load(f'./{dataset}_{appliance}/y_test.npy')
        state_test = np.load(f'./{dataset}_{appliance}/state_test.npy')
        min_max_scaling = np.load(f'./{dataset}_{appliance}/minmax.npy')

        dissagregator = AttentionTCNDissagregator(appliance, dataset, 1, 256)

        pruned_model = tf.keras.models.load_model(
            f'./{dataset}/saved_pruned_model/attention_tcn/model_appliance_{appliance}_percentage_{percentage}')

        dissagregator.set_pruned_model(pruned_model)
        dissagregator.quantize(dissagregator.pruned_model)

        tracker = EmissionsTracker(output_file=f"./{dataset}/{dissagregator.model_name}_{appliance}_emission.csv")
        tracker.start()

        predicted_output_sequence, predicted_output_sequence_onoff = dissagregator.predict_quantized(X_test)

        predicted_output_sequence *= min_max_scaling[1, 1] - min_max_scaling[1, 0]  # max_appliance - min_appliance

        # Clip negative values to zero
        predicted_output_sequence[predicted_output_sequence < 0] = 0.0
        predicted_output_sequence = np.minimum(predicted_output_sequence,
                                               X_test[:predicted_output_sequence.shape[0]] * min_max_scaling[0][1])
        emissions: float = tracker.stop()
        print(f"Emissions: {emissions} kg")

        dissagregator.compute_metrics(predicted_output_sequence, Y_test[:len(predicted_output_sequence)], predicted_output_sequence_onoff, state_test[:len(predicted_output_sequence)])
        print('*************************')
        print(dissagregator.model_name)
        print('Prediction finished')

        print('Quantization finished')
