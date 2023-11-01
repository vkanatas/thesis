from models.seq2seq_model import *


class CNNDissagregator(Seq2Seq):
    def __init__(self, appliance, dataset, epochs=100, batch_size=64):
        super().__init__("cnn_seq2seq", appliance, dataset, epochs, batch_size)

    def _build_model(self):
        input_data = tf.keras.Input(shape=(self.window_size, 1))

        x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        regression_output = tf.keras.layers.Dense(self.window_size, activation='linear', name="regression_output")(x)

        full_model = tf.keras.Model(inputs=input_data, outputs=regression_output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        full_model.compile(optimizer=optimizer, loss={
            "regression_output": tf.keras.losses.MeanSquaredError(),
        })

        return full_model
