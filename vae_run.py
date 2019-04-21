from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.models import load_model, model_from_json, Model
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

from vae.vae import VAE


def get_train_data(X_file, Y_file, test_size=0.1):
    # read data from file
    X = np.load(X_file)
    y = np.load(Y_file)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # size = X_train.shape
    # X_train = np.reshape(X_train, (size[0]*size[1], size[2], size[3]))

    # size = X_val.shape
    # X_val = np.reshape(X_val, (size[0]*size[1], size[2], size[3]))

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)

    return X_train, y_train, X_val, y_val

    # return data


def train(data, vae_obj, model_path, batch_size=1, epochs=20):
    X_train, _, X_val, _ = data

    # input_dim = X_train.shape[2:]  # (max_word, word_vector)
    input_dim = X_train.shape[2]  # (max_word)
    timesteps = X_train.shape[1]  # max_sentence
    print(input_dim)

    # ## Create model
    vae, enc = vae_obj.vae_lstm(input_dim,
                                timesteps=timesteps,
                                intermediate_dim=32)
    if not model_path is None:
        # or load model
        vae = load_model(model_path, custom_objects={
                         'sampling': vae_obj.sampling, 'vae_loss': vae_obj.vae_loss})
    vae.summary()

    # ## Fit model
    model_checkpoint = ModelCheckpoint(filepath='checkpoint/vae_lstm-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    # csv_logger = CSVLogger(filename='checkpoint/vae_lstm_training_log.csv',
    #                       separator=',',
    #                       append=True)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0,
                                   patience=10,
                                   verbose=1)
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.2,
                                             patience=8,
                                             verbose=1,
                                             epsilon=0.001,
                                             cooldown=0,
                                             min_lr=0.00001)
    callbacks = [model_checkpoint,
                 # csv_logger,
                 early_stopping,
                 reduce_learning_rate]

    history = vae.fit(X_train, X_train,
                      validation_data=(X_val, X_val),
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=1)

    # ## Plot loss
    plt.figure(figsize=(20, 12))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='upper right', prop={'size': 24})
    plt.show()

    # ## Save model
    # save all structures and weights of vae to one file
    vae.save('model/vae.h5')
    # save weights
    vae.save_weights('model/vae_weights.h5')

    # save vae structure
    vae_json = vae.to_json()
    with open('model/vae.json', 'w') as json_file:
        json_file.write(vae_json)

    return vae


def evaluate(X_test, vae_obj, model_path, weights_path=None):
    # ## Load model

    # json_file = open(model_path, 'r')
    # vae_json = json_file.read()
    # json_file.close()
    # vae = model_from_json(vae_json, custom_objects={
    #                       'sampling': vae_obj.sampling, 'vae_loss': vae_obj.vae_loss})

    # # load weights into new model
    # vae.load_weights(weights_path)

    vae = load_model(model_path, custom_objects={
                     'sampling': vae_obj.sampling, 'vae_loss': vae_obj.vae_loss})

    # ## Predict
    preds = vae.predict(X_test, batch_size=1)
    # print(X_test)
    # print(preds)

    # pick a column to plot.
    print("[plotting...]")
    #print("X_test[:, 0, 1]", (X_test[:, 0, 1]))
    #print("preds[:, 0, 1]", (preds[:, 0, 1]))
    print("x: %s, preds: %s" % (X_test.shape, preds.shape))
    print(X_test[:, 0, 0, 0])
    print(preds[:, 0, 0, 0])

    plt.plot(X_test[:, 0, 0, 0], label='data')
    plt.plot(preds[:, 0, 0, 0], label='predict')
    plt.legend()
    plt.show()

    return preds


def encode(X, vae_obj, model_path, weights_path=None):
    # ## Load model
    vae = load_model(model_path, custom_objects={
        'sampling': vae_obj.sampling, 'vae_loss': vae_obj.vae_loss})
    vae.summary()

    enc = Model(inputs=vae.input,
                outputs=vae.get_layer("lambda_1").output)

    enc.summary()

    # ## Apply model
    preds = enc.predict(X)

    print(preds)

    # ## Save this representations
    np.save('data/x_enc.npy', preds)

    return preds


if __name__ == "__main__":
    batch_size = 1

    # ## Load training data
    data = get_train_data('data/x_train_.npy', 'data/y_train_.npy')

    vae_obj = VAE(batch_size=batch_size,
                  latent_dim=100,
                  epsilon_std=1.)
    #vae, enc = train(data, vae_obj, model_path=None, batch_size=batch_size, epochs=40)

    # Load test data
    X_test = np.load('data/x_test_.npy')

    # evaluate(X_test, vae_obj, model_path='model/vae.h5')

    encode(X_test, vae_obj, model_path='model/vae.h5')
