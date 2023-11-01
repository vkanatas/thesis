"""
From Improving Non-Intrusive Load Disaggregation through an Attention-Based Deep Neural Network
V. Piccialli and Antonio M. Sudoso
"""

from models.seq2seq_with_classification import *


class PrunableBidirectionalLSTM(tf.keras.layers.Bidirectional, tfmot.sparsity.keras.PrunableLayer):
    def get_prunable_weights(self):
        # Return a list of prunable weights in the layer
        return self.forward_layer._trainable_weights + self.backward_layer._trainable_weights


class LDwADissagregator(Seq2SeqClassification):
    def __init__(self, appliance, dataset, epochs=100, batch_size=64, filters=32, kernel_size=4, units=128):
        self.filters = filters
        self.kernel_size = kernel_size
        self.units = units
        super().__init__("attention with classification", appliance, dataset, epochs, batch_size)

        print(self.units)

    @staticmethod
    def reduce_sum(x):
        return tf.reduce_sum(x, axis=1)

    def _build_model(self):
        input_data = tf.keras.Input(shape=(self.window_size, 1))
        print(self.model_name)

        # CLASSIFICATION SUBNETWORK
        x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
        x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
        classification_output = tf.keras.layers.Dense(units=self.window_size, activation='sigmoid',
                                                      name="classification_output")(x)

        # REGRESSION SUBNETWORK
        y = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(input_data)
        y = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(y)
        y = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(y)
        y = tf.keras.layers.Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu')(y)
        print(y.shape)

        y = PrunableBidirectionalLSTM(tf.keras.layers.LSTM(self.units, return_sequences=True), merge_mode='concat')(y)

        w = tf.keras.layers.Dense(units=self.units, activation='tanh', name="attention_W")(y)
        attention_weights = tf.keras.layers.Dense(units=1, activation='softmax', name="attention_V")(w)
        context_vector = tf.keras.layers.Multiply(name="attention_multiply")([y, attention_weights])
        context_vector = tf.keras.layers.Lambda(self.reduce_sum, name="attention_sum")(context_vector)

        y = tf.keras.layers.Dense(self.units, activation='relu')(context_vector)
        output = tf.keras.layers.Dense(self.window_size, activation='relu', name="output")(y)

        regression_output = tf.keras.layers.Multiply(name="regression_output")([output, classification_output])

        full_model = tf.keras.Model(inputs=input_data, outputs=[regression_output, classification_output], name="LDwA")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        full_model.compile(optimizer=optimizer, loss={
            "regression_output": tf.keras.losses.MeanSquaredError(),
            "classification_output": tf.keras.losses.BinaryCrossentropy()
        })
        full_model.summary()

        return full_model

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

        # Make an empty model and load weights, pruning does not work with just load_model
        model = self._build_model()
        model.load_weights(f'./{self.dataset}/saved_model/{self.model_name}/model_appliance_{self.appliance}_model_weights')
        model_to_prune = tf.keras.models.clone_model(
            model,
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

    def quantize(self, model):

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        quantized_and_pruned_tflite_model = converter.convert()
        if not os.path.isdir(f"./{self.dataset}/saved_quantized_model/{self.model_name}"):
            os.makedirs(f"./{self.dataset}/saved_quantized_model/{self.model_name}")

        tflite_file = f"./{self.dataset}/saved_quantized_model/{self.model_name}/model_appliance_{self.appliance}.tflite"
        with open(tflite_file, 'wb') as f:
            f.write(quantized_and_pruned_tflite_model)

        print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (
            get_gzipped_model_size(tflite_file)))

