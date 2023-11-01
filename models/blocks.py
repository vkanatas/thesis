
import tensorflow as tf


def glu_conv(input_data, filters, kernel_size):
    """
    Based on Convolutional Sequence to Sequence Non-intrusive Load Monitoring Kunjin Chen
    :param input_data:
    :param filters:
    :param kernel_size:
    :return:
    """
    main_path = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(input_data)
    additional_path = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="sigmoid"
                                              ,padding="same")(input_data)

    out = tf.keras.layers.Multiply()([main_path, additional_path])

    return out


def attention_unet_block(x, gating, inter_shape):

    theta_x = tf.keras.layers.Conv1D(inter_shape, kernel_size=1, strides=1, padding='same',name=f"attention_1_{inter_shape}")(x)  # 16
    theta_x = tf.keras.layers.BatchNormalization()(theta_x)
    print(f"theta x {theta_x.shape}")

    phi_g = tf.keras.layers.Conv1D(inter_shape, kernel_size=1, strides=1, padding='same',name=f"attention_2_{inter_shape}")(gating)
    phi_g = tf.keras.layers.BatchNormalization()(phi_g)

    upsample_g = tf.keras.layers.Conv1DTranspose(inter_shape, kernel_size=3,
                                                 strides=(theta_x.shape[1] // gating.shape[1]),
                                                 padding='same',name=f"attention_3_{inter_shape}")(phi_g)  # 16

    concat_xg = tf.keras.layers.Add()([upsample_g, theta_x])
    act_xg = tf.keras.layers.Activation('relu')(concat_xg)
    psi = tf.keras.layers.Conv1D(1, 1, padding='same',name=f"attention_4_{inter_shape}")(act_xg)

    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    y = tf.keras.layers.Multiply()([sigmoid_xg, x])

    result = tf.keras.layers.Conv1D(x.shape[2], 1, padding='same',name=f"attention_5_{inter_shape}")(y)
    result_bn = tf.keras.layers.BatchNormalization()(result)

    return result_bn


def tcn(input_data, filters, kernel_size, num_of_layers):
    """
    Based on Conv-NILM-Net, a causal and multi-appliance
    model for energy source separation?
    :param input_data:
    :param filters:
    :param kernel_size:
    :param num_of_layers:
    :return:
    """

    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=1)(input_data)

    for i in range(num_of_layers-1):
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', dilation_rate=2**(i+1))(x)

    x = tf.keras.layers.Activation('relu')(x)
    res = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(input_data)

    out = tf.keras.layers.Add()([x, res])

    return out

