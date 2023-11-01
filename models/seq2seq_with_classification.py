"""
Base model for seq2seq with regression and classification output
"""
from abc import ABC
import tensorflow as tf
from models.seq2seq_model import *


class Seq2SeqClassification(Seq2Seq, ABC):
    @abstractmethod
    def _build_model(self):
        # Output layers must be named regression_output, classification_output
        pass

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
        model_to_prune.compile(optimizer=optimizer, loss={
            "regression_output": tf.keras.losses.MeanSquaredError(),
            "classification_output": tf.keras.losses.BinaryCrossentropy()
        })

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



    def predict(self, x_test, y_test, state_test,type_model='whole'):
        if type_model == 'whole':
            model = self.model
        elif type_model == 'pruned':
            model = self.pruned_model
            print(model.summary)
        else:

            print('Not valid type_model must be whole(default) or pruned')
            return -1

        window_offset = int(self.window_size // 2)
        targets = np.column_stack([y_test, state_test])

        test_gen = TestSlidingWindowGenerator(number_of_windows=self.batch_size, inputs=x_test, targets=targets,
                                              offset=window_offset, stride= self.stride,
                                              qo=self.window_size % 2)

        predicted_output, predicted_output_onoff = model.predict(x=test_gen.load_dataset(), verbose=1)

        predicted_output_sequence = build_sequence(np.array(predicted_output), self.stride)
        predicted_output_sequence_onoff = build_sequence(np.array(predicted_output_onoff), self.stride)

        return predicted_output_sequence, predicted_output_sequence_onoff

    def predict_quantized(self, X_test):
        interpreter = tf.lite.Interpreter(
            model_path=f"./{self.dataset}/saved_quantized_model/{self.model_name}/model_appliance_{self.appliance}.tflite")
        interpreter.allocate_tensors()

        # Arrays for the predictions
        reg_pred = np.zeros(((len(X_test) - self.window_size) // self.stride, self.window_size), dtype=np.float32)
        class_pred = np.zeros(((len(X_test) - self.window_size)//self.stride, self.window_size), dtype=np.float32)

        input_index = interpreter.get_input_details()[0]["index"]

        output_index_0 = interpreter.get_output_details()[0]["index"]
        output_index_1 = interpreter.get_output_details()[1]["index"]
        print(f"max range {(len(X_test) - self.window_size)//self.stride}")
        for i in range(0, (len(X_test) - self.window_size)//self.stride):
            if i % 100 == 0:
                print(f" {i} samples done", flush=True)
            test_sample = np.reshape(X_test[i*self.stride:i*self.stride+self.window_size], (1, -1, 1)).astype(np.float32)
            interpreter.set_tensor(input_index, test_sample)
            interpreter.invoke()
            reg_pred[i] = interpreter.get_tensor(output_index_0)
            class_pred[i] = interpreter.get_tensor(output_index_1)

        predicted_output_sequence = build_sequence(reg_pred, self.stride)
        predicted_output_sequence_onoff = build_sequence(class_pred, self.stride)

        return predicted_output_sequence, predicted_output_sequence_onoff


    def compute_metrics(self, y_predict, y_test, states_predict ,states_test):
        f_mae = mae(y_predict, y_test[:y_predict.shape[0]])
        print(f'MAE is: {f_mae}')
        f1, recall, precision, acc = classification_metrics(states_predict, states_test[:states_predict.shape[0]])
        print(f"recall is: {recall}")
        print(f"precision is: {precision}")
        print(f'f1 score is: {f1}')
        print(f"accuracy is: {acc}")

        return f_mae, f1
