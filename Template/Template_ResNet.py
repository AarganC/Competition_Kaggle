import keras

from keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

# -----------------
#           |
# Model     |  n
#           |v1(v2)
# -----------------
# ResNet20  | 3 (2)
# ResNet32  | 5(NA)
# ResNet44  | 7(NA)
# ResNet56  | 9 (6)
# ResNet110 |18(12)
# ResNet164 |27(18)
# ResNet1001| (111)
# -----------------

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def ResNet_TEMPLATE(n, inputs, activation, dropout_flag, dropout_value):
    num_filters_in = 16
    num_classes = 2
    # Start model definition.

    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(int(n)):
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_flag == 1 or dropout_flag == 4 or dropout_flag == 5 or dropout_flag == 7:
        x = Dropout(dropout_value)(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    if dropout_flag == 2 or dropout_flag == 4 or dropout_flag == 6 or dropout_flag == 7:
        y = Dropout(dropout_value)(y)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    if dropout_flag == 3 or dropout_flag == 5 or dropout_flag == 6 or dropout_flag == 7:
        outputs = Dropout(dropout_value)(outputs)
    return outputs
