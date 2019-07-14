from keras.layers import Dense

def MLP(inputs, act,nb_layer,n):
    x = Dense(int(n), activation=act)(inputs)
    for i in range(int(nb_layer)):
        x = Dense(int(n), activation=act)(x)
    outputs = Dense(2, activation='sigmoid')(x)

    return outputs