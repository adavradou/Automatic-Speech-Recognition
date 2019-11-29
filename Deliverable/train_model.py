import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, SimpleRNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.models import load_model, Sequential
from keras.models import Model
from keras import backend as K

import matplotlib.pyplot as plt

def train_dense(all_features,all_label,labels, model_name="model_dense"):
    current_directory = os.getcwd()
    # print('X data:\n', all_features)    # [array([-0.0003346 , ..., ], dtype=float32), array....]
    # print('Y data:\n', all_label)   # ['zero', ..., 'nine']
    all_features = np.array(all_features)
    all_label = np.array(all_label)
    print('Size of X:\n', all_features.shape)  # (77, 8000)
    print('Size of Y:\n', all_label.shape)  # (77,)
    # print('X data (as array):\n', all_features)   # [[-3.3460365e-04 ...][...]...]
    # print('Y data (as array):\n', all_label)  # ['zero' ... 'nine']

    # sc = StandardScaler()
    # x_ = sc.fit_transform(all_features)
    x_ = all_features
    y_ = all_label
    # print('Standard Scalar:\n', x_)
    le = LabelEncoder()
    y_ = le.fit_transform(all_label)
    classes = list(le.classes_)
    # print('Label Encoder:\n', y_)
    # print('classes list:\n', classes)
    np.save(current_directory + '/classes_dense.npy', le.classes_)
    y_ = np_utils.to_categorical(y_, num_classes=len(labels))  # From int to one-hot
    print('to_categorical:\n', y_)
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(x_), np.array(y_), stratify=y_, test_size=0.2,
                                                random_state=777, shuffle=True)
    print('x_tr: ', x_tr.shape, '\nx_val: ', x_val.shape, '\ny_tr: ', y_tr.shape, '\ny_val: ', y_val.shape)
    # print('y_val:\n', y_val)
    n_cols = x_tr.shape[1]

    model = Sequential()

    model.add(Dense(int(n_cols * 1.5), activation='sigmoid', input_dim=n_cols))
    # model.add(Dropout(0.5))
    # model.add(Dense(int(n_cols*1.2), activation='sigmoid'))
    model.add(Dense(int(n_cols * 0.75), activation='sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(500, activation='relu'))
    model.add(Dense(30, activation='sigmoid'))

    model.add(Dense(len(labels), activation='softmax', name='pred'))

    # Model Summary
    model.summary()

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # Early Stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40, min_delta=0.05)
    # Keep only a single checkpoint, the best over test accuracy.
    filepath = current_directory + '/dense_' + model_name + '.hdf5'
    mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.save(current_directory + '/dense' + model_name + '.hdf5', True, True)

    # Fit Model
    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=64, validation_data=(x_val, y_val))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # evaluate model
    _, accuracy = model.evaluate(x_tr, y_tr)
    print('Accuracy: %.2f' % (accuracy * 100))
    # print(y_val)
    y_pred = model.predict_classes(x_tr)

    # Scores
    score_train = model.evaluate(x_tr, y_tr, verbose=0)
    score_val = model.evaluate(x_val, y_val, verbose=0)

    print('Train Score: ', score_train, '\nValidation Score: ', score_val)

    return filepath


def train_convolutional(all_features,all_label,labels, model_name="model_conv"):
    all_features = np.array(all_features)
    current_directory=os.getcwd()

    # Neural Network Classification
    # Label Encoding and train-test split
    le = LabelEncoder()  # Encode labels from (zero, one, ...) to (0, ..., 1)
    y = le.fit_transform(all_label)
    classes = list(le.classes_)
    np.save( current_directory + '/classes_conv.npy', le.classes_)
    y = np_utils.to_categorical(y, num_classes=len(labels))  # From int to one-hot
    all_features = np.array(all_features).reshape(-1, len(all_features[0]), 1)  # only for convolutional nn (2D to 3D)
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_features), np.array(y), stratify=y, test_size=0.2,
                                                random_state=777, shuffle=True)

    K.clear_session()

    print(all_features[0].shape, len(all_features[0]))

    inputs = Input(shape=(len(all_features[0]), 1))

    # First Conv1D layer
    conv = Conv1D(8, 13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    # Flatten layer
    conv = Flatten()(conv)

    # Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    # Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)

    model = Model(inputs, outputs)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)


    filepath = current_directory + '/conv_' + model_name + '.hdf5'

    # Keep only a single checkpoint, the best over test accuracy.
    mc = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    model.save(filepath, True, True)

    history = model.fit(x_tr, y_tr, epochs=100, callbacks=[es, mc], batch_size=32, validation_data=(x_val, y_val))

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # evaluate model
    _, accuracy = model.evaluate(x_tr, y_tr)
    print('Accuracy: %.2f' % (accuracy * 100))
    # print(y_val)
    y_pred = model.predict_classes(x_tr)

    # Scores
    score_train = model.evaluate(x_tr, y_tr, verbose=0)
    score_val = model.evaluate(x_val, y_val, verbose=0)

    print('Train Score: ', score_train, '\nValidation Score: ', score_val)

    return filepath
