from models.seq2seq_with_classification import *
from models.blocks import tcn


class AttentionTCNDissagregator(Seq2SeqClassification):
    def __init__(self, appliance, dataset,epochs=100, batch_size=64, filters=32, kernel_size=4, units=128):
        self.filters = filters
        self.kernel_size = kernel_size
        self.units = units
        super().__init__("attention_tcn", appliance, dataset, epochs, batch_size)

        print(self.units)

    @staticmethod
    def reduce_sum(x):
        return tf.reduce_sum(x, axis=1)

    def _build_model(self):

        input_data = tf.keras.Input(shape=(self.window_size,1))
        y = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size, activation='relu')(input_data)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)
        y = tf.keras.layers.MaxPooling1D()(y)
        y = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size, activation='relu')(y)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)
        y = tf.keras.layers.MaxPooling1D()(y)
        y = tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size, activation='relu')(y)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)
        y = tf.keras.layers.MaxPooling1D()(y)
        y = tf.keras.layers.Conv1D(filters=128, kernel_size=self.kernel_size, activation='relu')(y)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)

        y = tcn(y, self.units, self.kernel_size, num_of_layers=4)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tcn(y, self.units, self.kernel_size, num_of_layers=4)
        y = tf.keras.layers.BatchNormalization(scale=False)(y)

        w = tf.keras.layers.Dense(units=self.units, activation='tanh', name="attention_W")(y)
        attention_weights = tf.keras.layers.Dense(units=1, activation='softmax', name="attention_V")(w)
        context_vector = tf.keras.layers.Multiply(name="attention_multiply")([y, attention_weights])
        context_vector = tf.keras.layers.Lambda(self.reduce_sum, name="attention_sum")(context_vector)

        y = tf.keras.layers.Dense(self.units, activation='relu')(context_vector)
        output = tf.keras.layers.Dense(self.window_size, activation='relu', name="output")(y)
        classification_output = tf.keras.layers.Dense(units=self.window_size, activation='sigmoid',
                                                      name="classification_output")(y)
        regression_output = tf.keras.layers.Multiply(name="regression_output")([output, classification_output])
        full_model = tf.keras.Model(inputs=input_data, outputs=[regression_output, classification_output], name="attention_tcn")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        full_model.compile(optimizer=optimizer, loss={
            "regression_output": tf.keras.losses.MeanSquaredError(),
            "classification_output": tf.keras.losses.BinaryCrossentropy()
        })

        return full_model






