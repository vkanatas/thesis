import zipfile
import tensorflow as tf
import os
import numpy as np
import tensorflow_model_optimization as tfmot
import tempfile
from models.utils import *
from config import *
from data_feeder import TestSlidingWindowGenerator
from abc import ABC, abstractmethod
import pickle


class Seq2Seq(ABC):

    def __init__(self, model_name, appliance, dataset, epochs=100, batch_size=64):
        self.model_name = model_name
        self.appliance = appliance
        self.dataset = dataset
        self.stride = appliances[self.appliance]['stride']
        self.window_size = appliances[self.appliance]['window_size']
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()
        self.pruned_model = None  # Initially empty

        if not os.path.isdir(f"./{self.dataset}"):
            os.makedirs(f"./{self.dataset}")


    def train(self, x_train, y_train, x_val, y_val):
        # Building and training model

        self.model.summary()

        history = self.model.fit(x=x_train, y=y_train, epochs=self.epochs, validation_data=(x_val, y_val),
                                 batch_size=self.batch_size, verbose=1)
        self.model.save(f'{self.dataset}/saved_model/{self.model_name}/model_appliance_{self.appliance}')
        self.model.save_weights(f'./{self.dataset}/saved_model/{self.model_name}/model_appliance_{self.appliance}_model_weights', save_format='tf')

        # Plotting the results of training
        history_dict = history.history
        print(history_dict)
        if not os.path.isdir(f"./{self.dataset}/trainHistoryDict"):
            os.makedirs(f"./{self.dataset}/trainHistoryDict")
        filename = f"./{self.dataset}/trainHistoryDict/{self.model_name}_{self.appliance}_train_history.pkl"
        pickle.dump(history.history, open(filename, 'wb'))

    def prune(self, x_train, y_train, x_val, y_val, prune_percentage):
        end_step = np.ceil(x_train.shape[0] / self.batch_size).astype(np.int32) * self.epochs

        def not_prune_attention(layer):
            if "attention" in layer.name or "output" in layer.name:
                return layer
            return tfmot.sparsity.keras.prune_low_magnitude(layer,
                                                            pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
                                                                initial_sparsity=0.05,
                                                                final_sparsity=prune_percentage, begin_step=100,
                                                                end_step=end_step))

        model_to_prune = tf.keras.models.clone_model(
            self.model,
            clone_function=not_prune_attention,

        )

        model_to_prune.summary()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model_to_prune.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())

        logdir = tempfile.mkdtemp()

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        model_to_prune.fit(x=x_train, y=y_train, epochs=self.epochs, validation_data=(x_val, y_val),
                           verbose=1, callbacks=callbacks)
        self.pruned_model = tfmot.sparsity.keras.strip_pruning(model_to_prune)

        print("final model")
        self.pruned_model.summary()
        self.pruned_model.save(f'./{self.dataset}/saved_pruned_model/{self.model_name}/model_appliance_{self.appliance}_percentage_{prune_percentage}')

    def quantize(self, model):

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_and_pruned_tflite_model = converter.convert()
        if not os.path.isdir(f"./{self.dataset}/saved_quantized_model/{self.model_name}"):
            os.makedirs(f"./{self.dataset}/saved_quantized_model/{self.model_name}")

        tflite_file = f"./{self.dataset}/saved_quantized_model/{self.model_name}/model_appliance_{self.appliance}.tflite"
        with open(tflite_file, 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)

        print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (
            get_gzipped_model_size(tflite_file)))

    def predict(self, x_test, y_test, type_model='whole'):

        if type_model == 'whole':
            model = self.model
        elif type_model == 'pruned':
            model = self.pruned_model
        else:

            print('Not valid type_model to predict must be whole(default) or pruned')
            return -1

        window_offset = int(self.window_size // 2)

        test_gen = TestSlidingWindowGenerator(number_of_windows=self.batch_size, inputs=x_test, targets=y_test,
                                              offset=window_offset, stride=self.stride,
                                              qo=self.window_size % 2)

        predicted_output = model.predict(x=test_gen.load_dataset(), verbose=1)

        predicted_output_sequence = build_sequence(predicted_output, self.stride)

        return predicted_output_sequence


    def predict_quantized(self, X_test):
        interpreter = tf.lite.Interpreter(
            model_path=f"./{self.dataset}/saved_quantized_model/{self.model_name}/model_appliance_{self.appliance}.tflite")
        interpreter.allocate_tensors()

        # Arrays for the predictions
        reg_pred = np.zeros(((len(X_test) - self.window_size) // self.stride, self.window_size), dtype=np.float32)

        input_index = interpreter.get_input_details()[0]["index"]

        output_index_0 = interpreter.get_output_details()[0]["index"]

        print(f"max range {(len(X_test) - self.window_size) // self.stride}")
        for i in range(0, (len(X_test) - self.window_size) // self.stride):
            if i % 100 == 0:
                print(f" {i} samples done", flush=True)
            test_sample = np.reshape(X_test[i * self.stride:i * self.stride + self.window_size], (1, -1, 1)).astype(
                np.float32)
            interpreter.set_tensor(input_index, test_sample)
            interpreter.invoke()
            reg_pred[i] = interpreter.get_tensor(output_index_0)

        print("prediction done")
        predicted_output_sequence = build_sequence(reg_pred, self.stride)
        print("regression seq done")
        np.save(f'./{self.dataset}/prediction_{self.model_name}_quantized_{self.appliance}.npy',
                predicted_output_sequence)

        return predicted_output_sequence

    @abstractmethod
    def _build_model(self):
        pass

    def compute_metrics(self, y_predict, y_test, states_test):
        f_mae = mae(y_predict, y_test[:y_predict.shape[0]])
        print(f'MAE is: {f_mae}')
        states_predict = np.where(y_predict > appliances[self.appliance]['on_power'], 1, 0)
        f1, recall, precision, acc = classification_metrics(states_predict, states_test[:states_predict.shape[0]])
        print(f"recall is: {recall}")
        print(f"precision is: {precision}")
        print(f'f1 score is: {f1}')
        print(f"accuracy is: {acc}")

        return f_mae, f1

    def export_pruned_model(self):
        model_for_export = tfmot.sparsity.keras.strip_pruning(self.pruned_model)

        _, pruned_keras_file = tempfile.mkstemp('.h5')
        tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
        print('Saved pruned Keras model to:', pruned_keras_file)

        converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
        pruned_tflite_model = converter.convert()

        _, pruned_tflite_file = tempfile.mkstemp('.tflite')

        with open(pruned_tflite_file, 'wb') as f:
            f.write(pruned_tflite_model)

        print('Saved pruned TFLite model to:', pruned_tflite_file)
        print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))

    def set_model(self, model):
        self.model = model

    def set_pruned_model(self, model):
        self.pruned_model = model
