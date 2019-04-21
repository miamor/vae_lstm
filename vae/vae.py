import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives


class VAE(object):
    """
    # Arguments
        batch_size: int.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.
    """
    def __init__(self, batch_size,
             latent_dim,
             epsilon_std=1.):
        self.z_mean = None
        self.z_log_sigma = None
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_sigma -
                                 K.square(self.z_mean) - K.exp(self.z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    def sampling(self, args):
        self.z_mean, self.z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self.batch_size, self.latent_dim), mean=0., stddev=self.epsilon_std)
        return self.z_mean + self.z_log_sigma * epsilon

    def vae_lstm(self, input_dim,
                 timesteps,
                 intermediate_dim):
        """
        Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator. 

        # Arguments
            input_dim: int.
            timesteps: int, input timestep dimension.
            intermediate_dim: int, output shape of LSTM. 


        # References
            - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
            - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
        """

        input = Input(shape=(10, 50, 100,))

        # LSTM encoding
        encode_sentence = TimeDistributed(LSTM(intermediate_dim, input_shape=(50,100)))(input) # series of word_vector (of one sentence) to one reprentation
        encode_doc = LSTM(intermediate_dim, return_sequences=False)(encode_sentence) # series of sentence_reprentation to one representation of doc

        encode_dense = Dense(intermediate_dim)(encode_doc)


        # VAE Z layer
        self.z_mean = Dense(self.latent_dim)(encode_dense)
        self.z_log_sigma = Dense(self.latent_dim)(encode_dense)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([self.z_mean, self.z_log_sigma])`
        latent_space = Lambda(self.sampling, output_shape=(self.latent_dim,))(
            [self.z_mean, self.z_log_sigma])

        ''' encoder, from inputs to latent space '''
        # encoder = Model(x, self.z_mean)
        encoder = Model(input, latent_space)

        ''' end-to-end autoencoder '''
        decode_dense = Dense(intermediate_dim)(latent_space)
        repeat_doc_vec = RepeatVector(timesteps)(decode_dense)
        decode_doc = LSTM(intermediate_dim, return_sequences=True)(repeat_doc_vec) # decode doc representation back to sentenses representations

        repeat_sentence_vec = TimeDistributed(RepeatVector(50))(decode_doc)
        decode_sentence = TimeDistributed(LSTM(100, return_sequences=True))(repeat_sentence_vec) # decode sentence representation back to words representations

        vae = Model(input, decode_sentence)

        vae.compile(optimizer='rmsprop', loss=self.vae_loss)

        return vae, encoder

