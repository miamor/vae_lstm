from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.models import load_model, model_from_json, Model
from sklearn.model_selection import train_test_split

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt
import joblib
import argparse

from vae.vae import VAE


# dataset = 'FLOW016'
# op = 'Op'


def get_train_data(X_file, Y_file, test_size=0.1):
    # read data from file
    X = np.load(X_file)
    y = np.load(Y_file)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)

    return X_train, y_train, X_val, y_val


def get_train_data_(X_file, Y_file, X_val_file, Y_val_file):
    # read data from file
    X_train = np.load(X_file)
    y_train = np.load(Y_file)

    X_val = np.load(X_val_file)
    y_val = np.load(Y_val_file)

    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)

    return X_train, y_train, X_val, y_val


def train(data, vae_obj, model_path, batch_size=1, epochs=20, out_dir='./checkpoint'):
    X_train, _, X_val, _ = data

    # input_dim = X_train.shape[2:]  # (max_word, word_vector)
    words = X_train.shape[2]  # (max_word)
    word_vec_dim = X_train.shape[3]  # (max_word)
    timesteps = X_train.shape[1]  # max_sentence
    # print(word_vec_dim)
    # input_shape = X_train[1:3]
    # print(input_shape)

    # ## Create model
    vae, _ = vae_obj.vae_lstm(words=words, word_vec_dim=word_vec_dim,
                              timesteps=timesteps,
                              intermediate_dim=32)
    if not model_path is None:
        # or load model
        vae = load_model(model_path, custom_objects={
                         'sampling': vae_obj.sampling, 'vae_loss': vae_obj.vae_loss})
    vae.summary()

    # ## Fit model
    model_checkpoint = ModelCheckpoint(filepath=out_dir+'/vae_lstm-{epoch:02d}_loss-{loss:.5f}_val_loss-{val_loss:.5f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    # csv_logger = CSVLogger(filename=out_dir+'/vae_lstm_training_log.csv',
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
                 reduce_learning_rate
                 ]

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
    # print("X_test[:, 0, 1]", (X_test[:, 0, 1]))
    # print("preds[:, 0, 1]", (preds[:, 0, 1]))
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
    X_enc = enc.predict(X)

    return X_enc


def train_classifier(X_train, Y_train, clf='CART', save_path='model/classifier.pkl'):
    """
    Spot Check Algorithms

    This is just for inspecting which classifier could possibly work best with our data
    """
    # models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # # evaluate each model in turn
    # results = []
    # names = []
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10, random_state=7)
    #     cv_results = model_selection.cross_val_score(
    #         model, X_train, Y_train, cv=kfold, scoring='accuracy')
    #     results.append(cv_results)
    #     names.append(name)
    #     print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))

    """
    Train the classifier
    """
    if clf == 'CART':
        classifier = DecisionTreeClassifier()
    elif clf == 'LDA':
        classifier = LinearDiscriminantAnalysis()
    elif clf == 'LR':
        classifier = LogisticRegression()
    elif clf == 'KNN':
        classifier = KNeighborsClassifier()
    elif clf == 'NB':
        classifier = GaussianNB()
    elif clf == 'SVM':
        classifier = SVC()
    classifier.fit(X_train, Y_train)

    # Save the trained model as a pickle string.
    joblib.dump(classifier, save_path)

    # View the pickled model
    return classifier


def evaluate_classifier(X, Y, classifier_path='model/classifier.pkl'):
    # Load the pickled model
    clf = joblib.load(classifier_path)

    # Use the loaded pickled model to make predictions
    predictions = clf.predict(X)

    print("\nTest accuration: {}".format(accuracy_score(Y, predictions)))
    print("Prediction :")
    print(np.asarray(predictions, dtype="int32"))
    print("Target :")
    print(np.asarray(Y, dtype="int32"))
    print(classification_report(Y, predictions))


def test(X, classifier_path='model/classifier.pkl'):
    # Load the pickled model
    clf = joblib.load(classifier_path)

    # Use the loaded pickled model to make predictions
    predictions = clf.predict(X)

    return predictions





def train_VAE(args):
    ''' Train VAE '''

    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]

    # Load training data
    # data = get_train_data('data/'+dataset+'/x_train_'+dataset+'_'+op+'.npy', 'data/'+dataset+'/y_train_'+dataset+'_'+op+'.npy')
    (X_train, y_train, X_val, y_val) = get_train_data_('data/'+dataset+'/x_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/x_val_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_val_'+dataset+'_'+op+'.npy')

    vae_obj = VAE(batch_size=1,
                  latent_dim=100,
                  epsilon_std=1.)

    vae, enc = train(data=(X_train, y_train, X_val, y_val),
                     vae_obj=vae_obj,
                     model_path=args.model_path,
                     batch_size=args.batch_size, epochs=args.epochs,
                     out_dir='checkpoint/'+dataset+'_'+op)

    # ## Load test data
    # X_test = np.load('data/x_test_.npy')
    # evaluate(X_test, vae_obj, model_path='model/vae.h5')


def encode_main(args):
    ''' Use vae to encode data '''

    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]
    model_path = args.model_path

    # model_path = 'checkpoint/__/vae_lstm-300_loss-0.01565_val_loss-0.01560.h5'

    (X_train, _, X_val, _) = get_train_data_('data/'+dataset+'/x_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/x_val_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_val_'+dataset+'_'+op+'.npy')

    vae_obj = VAE(batch_size=1,
                  latent_dim=100,
                  epsilon_std=1.)

    # Encode X_train
    # X_train_enc = encode(X_train, vae_obj, model_path='model/vae.h5')
    X_train_enc = encode(
        X_train, vae_obj, model_path=model_path)
    # Save this representations
    np.save('data/'+dataset+'/x_train_'+dataset +
            '_'+op+'_enc.npy', X_train_enc)

    # Encode X_val
    X_val_enc = encode(X_val, vae_obj, model_path=model_path)
    # Save this representations
    np.save('data/'+dataset+'/x_val_'+dataset+'_'+op+'_enc.npy', X_val_enc)


def train_clf(args):
    ''' Use encode data to train classifier '''

    dataset = args.dataset.split('_')[0]
    op = args.dataset.split('_')[1]
    clf = args.clf

    # Train
    X_train_enc = np.load('data/'+dataset+'/x_train_' +
                          dataset+'_'+op+'_enc.npy')

    if clf in ['SVM', 'LR']:
        (_, y_train, _, y_val) = get_train_data_('data/'+dataset+'/x_train_'+dataset+'_'+op+'.npy',
                                                 'data/'+dataset+'/y_train_'+dataset+'_'+op+'__.npy',
                                                 'data/'+dataset+'/x_val_'+dataset+'_'+op+'.npy',
                                                 'data/'+dataset+'/y_val_'+dataset+'_'+op+'__.npy')
    else:
        (_, y_train, _, y_val) = get_train_data_('data/'+dataset+'/x_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_train_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/x_val_'+dataset+'_'+op+'.npy',
                                                       'data/'+dataset+'/y_val_'+dataset+'_'+op+'.npy')

    print(X_train_enc)
    print('train_enc shape')
    print(X_train_enc.shape)
    print(y_train.shape)

    train_classifier(X_train_enc, y_train, clf, save_path='model/' +
                     dataset+'/classifier_'+op+'_'+clf+'.pkl')

    # Evaluate
    X_val_enc = np.load('data/'+dataset+'/x_val_' +
                        dataset+'_'+op+'_enc.npy')

    print('val_enc shape')
    print(X_val_enc.shape)
    print(y_val.shape)

    evaluate_classifier(X_val_enc, y_val, classifier_path='model/' +
                        dataset+'/classifier_'+op+'_'+clf+'.pkl')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='FLOW016_Op')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    train_VAE_Parser = subparsers.add_parser('train_VAE',
                                             help="Train a new model.")
    train_VAE_Parser.add_argument('-b', '--batch_size', type=int, default=512)
    train_VAE_Parser.add_argument('-e', '--epochs', type=int, default=300)
    train_VAE_Parser.add_argument('-m', '--model_path', type=str, default=None)

    encodeParser = subparsers.add_parser(
        'encode', help='Encode using trained VAE.')
    encodeParser.add_argument('-m', '--model_path', type=str)

    train_clf_Parser = subparsers.add_parser(
        'encode', help='Encode using trained VAE.')
    train_clf_Parser.add_argument('-c', '--clf', type=str, default='CART')


    args = parser.parse_args()

    if args.mode == 'train_VAE':
        train_VAE(args)
    elif args.mode == 'encode':
        encode_main(args)
    elif args.mode == 'train_clf':
        train_clf(args)
