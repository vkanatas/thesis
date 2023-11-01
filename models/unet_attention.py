from models.seq2seq_with_classification import *
from models.blocks import tcn, glu_conv, attention_unet_block


class UnetAttentionDissagregator(Seq2SeqClassification):
    def __init__(self, appliance, dataset, epochs=100, batch_size=64, filters=32, kernel_size=4, units=128):
        self.filters = filters
        self.kernel_size = kernel_size
        self.units = units
        super().__init__("attention_unet_maxpooling", appliance, dataset, epochs, batch_size)

        print(self.units)

    def _build_model(self):
        input_data = tf.keras.Input(shape=(self.window_size, 1))

        print(input_data.dtype)
        print(self.model_name)
        # CLASSIFICATION SUBNETWORK
        x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu', name="first")(input_data)
        print(x.dtype)
        x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
        classification_output = tf.keras.layers.Dense(units=self.window_size, activation='sigmoid',
                                                      name="classification_output")(x)
        class_out_res = tf.keras.layers.Reshape((self.window_size, 1))(classification_output)

        y = tf.keras.layers.Concatenate()([input_data, class_out_res])

        # REGRESSION SUBNETWORK
        enc16 = glu_conv(y, 16, self.kernel_size)

        enc16 = tf.keras.layers.BatchNormalization(scale=False)(enc16)
        enc161 = tf.keras.layers.MaxPooling1D()(enc16)

        enc32 = glu_conv(enc161, 32, self.kernel_size)
        enc32 = tf.keras.layers.BatchNormalization(scale=False)(enc32)
        enc321 = tf.keras.layers.MaxPooling1D()(enc32)

        enc64 = glu_conv(enc321, 64, self.kernel_size)
        enc64 = tf.keras.layers.BatchNormalization(scale=False)(enc64)
        enc641 = tf.keras.layers.MaxPooling1D()(enc64)

        enc128 = glu_conv(enc641, 128, self.kernel_size)
        enc128 = tf.keras.layers.BatchNormalization(scale=False)(enc128)

        base = tcn(enc128, self.units, self.kernel_size, num_of_layers=7)
        base = tf.keras.layers.BatchNormalization(scale=False)(base)

        base = tf.keras.layers.Dropout(0.2)(base)
        base = tcn(base, self.units, self.kernel_size, num_of_layers=7)
        base = tf.keras.layers.BatchNormalization(scale=False)(base)

        dec64 = tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size, padding="same", activation='relu',
                                       name="dec64")(
            base)
        att64 = attention_unet_block(enc64, dec64, 64)
        up64 = tf.keras.layers.UpSampling1D(size=2)(base)
        up64 = tf.keras.layers.Concatenate()([up64, att64])
        up64 = tf.keras.layers.BatchNormalization(scale=False)(up64)
        up_conv64 = tf.keras.layers.Conv1D(filters=64, kernel_size=self.kernel_size, padding="same", activation='relu')(
            up64)
        up_conv64 = tf.keras.layers.BatchNormalization(scale=False)(up_conv64)

        dec32 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size, padding="same", activation='relu',
                                       name="dec32")(
            up_conv64)
        att32 = attention_unet_block(enc32, dec32, 32)
        up32 = tf.keras.layers.UpSampling1D(size=2)(up_conv64)
        up32 = tf.keras.layers.Concatenate()([up32, att32])
        up_conv32 = tf.keras.layers.Conv1D(filters=32, kernel_size=self.kernel_size, padding="same", activation='relu')(
            up32)
        up_conv32 = tf.keras.layers.BatchNormalization(scale=False)(up_conv32)

        dec16 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size, padding="same", activation='relu',
                                       name="dec16")(up_conv32)
        att16 = attention_unet_block(enc16, dec16, 16)
        up16 = tf.keras.layers.UpSampling1D(size=2)(up_conv32)
        up16 = tf.keras.layers.Concatenate()([up16, att16])
        up16 = tf.keras.layers.BatchNormalization(scale=False)(up16)
        up_conv16 = tf.keras.layers.Conv1D(filters=16, kernel_size=self.kernel_size, padding="same", activation='relu')(
            up16)

        y = tf.keras.layers.Dropout(0.2)(up_conv16)
        y = tf.keras.layers.Flatten()(y)
        regression_output = tf.keras.layers.Dense(self.window_size, activation='relu', name="regression_output")(y)

        full_model = tf.keras.Model(inputs=input_data, outputs=[regression_output, classification_output],
                                    name="AttentionUnet")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        full_model.compile(optimizer=optimizer, loss={
            "regression_output": tf.keras.losses.MeanSquaredError(),
            "classification_output": tf.keras.losses.BinaryCrossentropy()},
                           )
        print(full_model.summary())
        return full_model





