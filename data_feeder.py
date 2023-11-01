"""
Based on https://github.com/kunnao/LightNILM
"""

import numpy as np

class TestSlidingWindowGenerator(object):

    def __init__(self, number_of_windows, inputs, targets, offset, stride,qo=1):
        self.__number_of_windows = number_of_windows
        self.__offset = offset
        self.__inputs = inputs
        self.__stride = stride
        self.__targets = targets

        if targets.ndim == 1:
            self.n_outputs = 1
        else:
            self.n_outputs = targets.shape[1]
        self.total_size = len(inputs)
        self.qo = qo

    def load_dataset(self):

        self.__inputs = self.__inputs.flatten()
        max_number_of_windows = self.__inputs.size - 2 * self.__offset

        if self.__number_of_windows < 0:
            self.__number_of_windows = max_number_of_windows

        indicies = np.arange(0, max_number_of_windows, self.__stride, dtype=int)
        print(indicies)
        for start_index in range(0, len(indicies), self.__number_of_windows):

            splice = indicies[start_index: start_index + self.__number_of_windows]
            input_data = np.array([self.__inputs[index: index + 2 * self.__offset + self.qo] for index in splice])
            target_data = self.__targets[splice].reshape(-1, self.n_outputs)

            yield input_data, target_data
