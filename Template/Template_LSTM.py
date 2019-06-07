import keras

from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, concatenate
from keras.layers import multiply, add
from keras.layers import Input

def LSTM_TEMPLATE (inputs, nb_filtre, nb_layer):
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

    x = keras.layers.add([x, outputs])

    y = Flatten()(x)

    outputs = Dense(2, activation='softmax')(y)
    return outputs
