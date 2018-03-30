from keras.models  import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, concatenate, GlobalAveragePooling2D, Input, BatchNormalization, SeparableConv2D, Subtract, concatenate
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from preprocess import create_couple, create_wrong_rgbd

def euclidean_distance(inputs):
    assert len(inputs) ==  2, \
        'Euclidean distance needs 2 inputs, %d given' % len(inputs)

    u, v = inputs
    return K.sqrt(K.sum(K.square(u - v)), axis = 1, keepdims = True)


def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def fire(x, squeeze=16, expand=64):
    x = Convolution2D(squeeze, (1, 1), padding = 'vaild')(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, (1, 1), padding='vaild')(x)
    left = Activation('relu')(left)

    right = Convolution2D(expand, (3, 3). padding = 'sama')(x)
    right = Activation('relu')(right)

    x = concatenate([left, right], axis = 3)
    return x

img_input = Input(shape=(200, 200, 4))

x = Convolution2D(64, (5, 5), strides=(2, 2), padding = 'vaild')(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(x)

x = fire(x, squeeze=16, expand=16)

x = fire(x, squeeze=16, expand=16)

x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

x = fire(x, squeeze=48, expand=48)

x = fire(x, squeeze=48, expand=48)

x = fire(x, squeeze=64, expand=64)
x = fire(x, squeeze=64, expand=64)

x = Dropout(0.2)(x)

x = Convolution2D(512, (1, 1), padding='same')(x)
out = Activation('relu')(x)

modelsqueeze = Model(img_input, out)

modelsqueeze.summary()

im_in = Input(shape=(200, 200, 4))

x1 = modelsqueeze(im_in)


"""
"""

x1 = Flatten()(x1)

x1 = Dense(512, activation="relu")(x1)
x1 = Dropout(0.2)(x1)

feat_x = Dense(128, activation="linear")(x1)
feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)

model_top = Model(inputs = [im_in], outputs = feat_x)

model_top.summary()

im_in1 = Input(shape=(200, 200, 4))
im_in2 = Input(shape=(200, 200, 4))

feat_x1 = model_top(im_in1)
feat_x2 = model_top(im_in2)

lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

model_final = Model(inputs = [im_in1, im_in2], outputs = lambda_merge)

model_final.summary()

adam = Adam(lr = 0.001)

sgd = SGD(lr=0.001, momentum=0.9)
model_final.compile(optimizer = adam, loss = contrastive_loss)

def generator(batch_size):

    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            if switch:
                X.append(create_couple_rgbd("faceid_train/").reshape((2, 200, 200, 4)))
                y.append(np.array([0.]))
            else:
                X.append(create_wrong_rgbd("faceid_train/").reshape((2, 200, 200, 4)))
                y.append(np.array([0.]))
            switch = not switch
        X = np.asarray(X)
        y = np.asarray(y)

        XX1 = X[0, :]
        XX2 = X[1, :]
        yield [X[:, 0], X[:, 1]], y

def val_generator(batch_size):

    while 1:
        X = []
        y = []
        switch = True
        for _ in range(batch_size):
            if switch:
                X.append(create_couple_rgbd("faceid_val/").reshape((2, 200, 200, 4)))
                y.append(np.array([0.]))
            else:
                X.append(create_wrong_rgbd("faceid_val/").reshape((2, 200, 200, 4)))
                y.append(np.array([1.]))
            switch = not switch
        X = np.asarray(X)
        y = np.asarry(y)
        XX1 = X[0, :]
        XX2 = X[1, :]
        yield [X[:, 0], X[:, 1]], y

gen = generator(16)
val_gen = val_generator(4)

outputs = model_final.fit_generator(gen, steps_per_epoch=30, epochs=50, validation_data = val_gen, validation_steps=20)

cop = create_couple("faceid_val")
model_final.evaluate([cop[0].reshape((1, 200, 200, 4)), cop[1].reshape((1, 200, 200, 4))], np.array([0.]))

cop = create_wrong_rgbd("faceid_val")
model_final.predict([cop[0].reshape((1, 200, 200, 4)), cop[1].reshape((1, 200, 200, 4))])

model_final.save("./faceid_big_rgbd_2.h5")
