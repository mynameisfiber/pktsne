from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Model, Input
import numpy as np
import keras.backend as K

from utils import wrapped_partial


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    # Initialize some variables
    n = X.shape[0]                     # number of instances
    P = np.zeros((n, n))               # empty probability matrix
    beta = np.ones(n)                  # empty precision vector
    logU = np.log(u)                   # log of perplexity (= entropy)

    # Compute pairwise distances
    if verbose > 0:
        print('Computing pairwise distances...')
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add
    # a dimension
    D = sum_X + sum_X[:, None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0:
        print('Computing P-values...')
    for i in range(n):

        if verbose > 1 and print_iter and i % print_iter == 0:
            print('Computed P-values {} of {} datapoints...'.format(i, n))

        # Set minimum and maximum values for precision
        betamin = float('-inf')
        betamax = float('+inf')

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
        print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
        print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

    return P, beta


def compute_joint_probabilities(samples, batch_size=5000, d=2, perplexity=30,
                                tol=1e-5, verbose=0, print_iter=500,
                                max_tries=50):
    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0:
        print('Precomputing P-values...')
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
        # select batch
        curX = samples[start:start+batch_size]
        # compute affinities using fixed perplexity
        P[i], beta = x2p(curX, perplexity, tol, verbose=verbose,
                         print_iter=print_iter, max_tries=max_tries)
        # make sure we don't have NaN's
        P[i][np.isnan(P[i])] = 0
        # make symmetric
        P[i] = (P[i] + P[i].T)
        # obtain estimation of joint probabilities
        P[i] = P[i] / P[i].sum()
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)
    return P


# P is the joint probabilities for this batch (Keras loss functions call this
# y_true) activations is the low-dimensional output (Keras loss functions call
# this y_pred)
def tsne_loss(P, activations, batch_size=32, d=2):
    v = d - 1.
    # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
    eps = K.variable(10e-15)
    sum_act = K.sum(K.square(activations), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations,
                                                 K.transpose(activations))
    Q = (sum_act + Q) / v
    Q = K.pow(1 + Q, -(v + 1) / 2)
    Q *= K.variable(1 - np.eye(batch_size))
    Q /= K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log((P + eps) / (Q + eps))
    C = K.sum(P * C)
    return C


def create_model(shape, d=2, batch_size=32):
    # TODO: read paper to verify model shapes here
    X = X_in = Input(shape=shape)
    X = Dense(500, activation='relu')(X)
    X = Dense(500, activation='relu')(X)
    X = Dense(2000, activation='relu')(X)
    X = Dense(d)(X)
    model = Model(inputs=[X_in], outputs=[X], name='tsne')

    sgd = SGD(lr=0.1)
    loss = wrapped_partial(tsne_loss, batch_size=batch_size, d=d)
    model.compile(loss=loss, optimizer=sgd)
    return model


class PTSNE(object):
    def __init__(self, X=None, d=2, batch_size=32,
                 perplexity=30, tol=1e-5, print_iter=500, max_tries=50,
                 n_iter=100, verbose=0):
        self.d = d
        self.batch_size = batch_size
        self.perplexity = perplexity
        self.tol = tol
        self.print_iter = print_iter
        self.max_tries = max_tries
        self.n_iter = n_iter
        self.verbose = verbose
        if X is not None:
            self.fit(X)

    def fit(self, X):
        N = X.shape[0] // self.batch_size * self.batch_size
        X = np.random.permutation(X[:N])
        data_shape = X.shape[1:]
        self.model = create_model(data_shape, self.d, self.batch_size)
        P = compute_joint_probabilities(
            X,
            d=self.d,
            perplexity=self.perplexity,
            tol=self.tol,
            batch_size=self.batch_size,
            print_iter=self.print_iter,
            max_tries=self.max_tries,
            verbose=self.verbose
        )
        Y = P.reshape(X.shape[0], -1)
        self.model.fit(X, Y, batch_size=self.batch_size,
                       shuffle=False, epochs=self.n_iter,
                       verbose=self.verbose)

    def transform(self, X):
        return self.model.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
