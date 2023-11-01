
import tempfile
import numpy as np
import time


def mae(prediction, true):
    MAE = abs(true - prediction)
    MAE = np.sum(MAE)
    MAE = MAE / len(prediction)
    return MAE


def sae(prediction, true, N):
    T = len(prediction)
    K = int(T / N)
    SAE = 0
    for k in range(1, N):
        pred_r = np.sum(prediction[k * N: (k + 1) * N])
        true_r = np.sum(true[k * N: (k + 1) * N])
        SAE += abs(true_r - pred_r)
    SAE = SAE / (K * N)
    return SAE


def classification_metrics(prediction, true):
    epsilon = 1e-8
    TP = epsilon
    FN = epsilon
    FP = epsilon
    TN = epsilon
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            prediction_binary = 1
        else:
            prediction_binary = 0
        if prediction_binary == 1 and true[i] == 1:
            TP += 1
        elif prediction_binary == 0 and true[i] == 1:
            FN += 1
        elif prediction_binary == 1 and true[i] == 0:
            FP += 1
        elif prediction_binary == 0 and true[i] == 0:
            TN += 1
    print(f"TP is: {TP}")
    print(f"TN is: {TN}")
    print(f"FN is: {FN}")
    print(f"FP is: {FP}")

    R = TP / (TP + FN)
    P = TP / (TP + FP)
    acc = (TP+TN) / (TP+TN+FP+FN)
    f1 = (2 * P * R) / (P + R)
    return f1, R, P, acc


def standardize_data(data, mu=0.0, sigma=1.0):
    data -= mu
    data /= sigma
    return data


def normalize_data(data, min_value=0.0, max_value=1.0):
    data -= min_value
    data /= max_value - min_value


def build_sequence(array, stride):
    """
    Takes as input a 2D array and produces a sequence using median on the overlapping values
    :param array: The 2D array
    :param stride:
    :return: The final sequence
    """

    start = time.time()

    unique_sequence = []
    samples, window_size = array.shape

    # Find positions of every instance and calculate the median, add it to the sequence
    for i in range(stride * (array.shape[0] - 1) + window_size):
        # from which row to start looking, make negative values 0
        n_rows = (i-window_size) // stride + 1
        n_rows = n_rows if n_rows > 0 else 0
        # till which row to look for, maximum value is window_size//stride but for small values we need less rows
        row_offset = i//stride + 1 if i < window_size else window_size // stride

        # Calcualte in which (rows,columns) there are instances, something like diagonal
        # take care of the last diagonals
        rows = [j for j in range(n_rows, min(n_rows+row_offset, samples))]
        columns = [j for j in range(i-n_rows*stride, -1, -stride)]

        unique_sequence.append(np.median(array[rows, columns[:len(rows)]]))

    unique_sequence = np.array(unique_sequence)

    end = time.time()
    print('*********************')
    print(f'Time spend: {end - start} sec')
    return unique_sequence

def get_gzipped_model_size(file):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)
