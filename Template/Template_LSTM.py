import keras

from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, concatenate
from keras.layers import multiply, add
from keras.layers import Input

def LSTM_TEMPLATE (inputs, nb_filtre, nb_layer, dropout_flag, dropout_value):
    nb_filtre = int(nb_filtre)
    nb_filtre_b = nb_filtre*2

    print(nb_filtre)
    print(nb_filtre_b)

    x = Conv2D(nb_filtre,
               kernel_size=3,
               strides=1,
               padding='same')(inputs)
    x = BatchNormalization()(x)
    for i in range(int(nb_layer)):
        # Layer 1
        C = Conv2D(nb_filtre_b,
                   kernel_size=3,
                   strides=1,
                   padding='same')(x)
        C = BatchNormalization()(C)

        H = Conv2D(nb_filtre,
                   kernel_size=3,
                   strides=1,
                   padding='same')(x)
        H = BatchNormalization()(H)

        Hx = concatenate([H, x])
        F = Activation('sigmoid')(Hx)

        I = Activation('sigmoid')(Hx)
        L = Activation('tanh')(Hx)
        IL = multiply([I, L])

        C = multiply([C, F])
        C = add([C, IL])

        C = Activation('tanh')(C)
        O = Activation('sigmoid')(Hx)
        x = multiply([C, O])

        nb_filtre_b = nb_filtre_b + nb_filtre

    outputs = Conv2D(nb_filtre,
                      kernel_size=3,
                      strides=1,
                      padding='same')(x)
    x = BatchNormalization()(outputs)

    if dropout_flag == 1 or dropout_flag == 4 or dropout_flag == 5 or dropout_flag == 7:
        x = Dropout(dropout_value)(x)

    x = keras.layers.add([x, outputs])

    y = Flatten()(x)

    if dropout_flag == 2 or dropout_flag == 4 or dropout_flag == 6 or dropout_flag == 7:
        y = Dropout(dropout_value)(y)

    outputs = Dense(2, activation='softmax')(y)

    if dropout_flag == 3 or dropout_flag == 5 or dropout_flag == 6 or dropout_flag == 7:
        outputs = Dropout(dropout_value)(outputs)

    return outputs
