from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Model, Input
from keras.callbacks import EarlyStopping
import numpy as np
import keras.backend as K

import itertools as IT

from utils import wrapped_partial, chunk


def Hbeta(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    sumP = np.maximum(sumP, np.finfo(sumP.dtype).eps)
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


def compute_joint_probabilities(batched_samples, batch_size, d=2,
                                perplexity=30, tol=1e-5, verbose=0,
                                print_iter=500, max_tries=100):
    # Precompute joint probabilities for all batches
    if verbose > 0:
        print('Precomputing P-values...')
    while True:
        batch = next(batched_samples)
        # compute affinities using fixed perplexity
        P, beta = x2p(batch, perplexity, tol, verbose=verbose,
                      print_iter=print_iter, max_tries=max_tries)
        # make sure we don't have NaN's
        P[np.isnan(P)] = 0
        # make symmetric
        P = (P + P.T)
        # obtain estimation of joint probabilities
        P = P / P.sum()
        P = np.maximum(P, np.finfo(P.dtype).eps)
        yield P


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
    X = X_in = Input(shape=shape, name='input')
    X = Dense(500, activation='relu')(X)
    X = Dense(500, activation='relu')(X)
    X = Dense(2000, activation='relu')(X)
    X = Dense(d)(X)
    model = Model(inputs=X_in, outputs=X, name='tsne')

    sgd = SGD(lr=0.1)
    loss = wrapped_partial(tsne_loss, batch_size=batch_size, d=d)
    model.compile(loss=loss, optimizer=sgd)
    return model


class PTSNE(object):
    def __init__(self, X=None, d=2, batch_size=32,
                 perplexity=30, tol=1e-5, print_iter=500, max_tries=100,
                 n_iter=100, verbose=0, shuffle=True,
                 n_iter_without_progress=100):
        self.d = d
        self.batch_size = batch_size
        self.perplexity = perplexity
        self.tol = tol
        self.shuffle = shuffle
        self.print_iter = print_iter
        self.max_tries = max_tries
        self.n_iter_without_progress = n_iter_without_progress
        self.n_iter = n_iter
        self.verbose = verbose
        if X is not None:
            self.fit(X)

    def fit(self, X, precalc_p=True, batch_count=None, data_shape=None):
        # TODO: better check for generator below that deals with
        # itertools.cycle types since `types.GeneratorType` doesn't work for
        # that
        if hasattr(X, '__next__'):
            if batch_count is None:
                raise Exception("For generator input, batch_count must "
                                "be specified")
            if data_shape is None:
                raise Exception("For generator input, data_shape must "
                                "be specified")
            self.model = create_model(data_shape, self.d, self.batch_size)
            if not precalc_p:
                X, X_pgen = IT.tee(X)
            else:
                X_pgen = X
            P_gen = compute_joint_probabilities(
                X_pgen,
                batch_size=self.batch_size,
                d=self.d,
                perplexity=self.perplexity,
                tol=self.tol,
                print_iter=self.print_iter,
                max_tries=self.max_tries,
                verbose=self.verbose
            )
            if precalc_p:
                P = list(IT.islice(P_gen, batch_count))
                P_gen = IT.cycle(P)
            self.model.fit_generator(
                zip(X, P_gen),
                steps_per_epoch=batch_count,
                shuffle=self.shuffle,
                epochs=self.n_iter,
                verbose=self.verbose,
                callbacks=[
                    EarlyStopping(monitor='loss', mode='min', min_delta=1e-4,
                                  patience=self.n_iter_without_progress),
                ],
            )
        else:
            batch_size = min(self.batch_size, X.shape[0])
            N = (X.shape[0] // batch_size) * batch_size
            if N != X.shape[0]:
                print(("WARNING: Data size not divisible by the batch size. "
                       "We are going to ignore the last {} "
                       "samples").format(X.shape[0] - N))
            n = N // batch_size
            X = X[:N]
            data_shape = X.shape[1:]
            self.model = create_model(data_shape, self.d, batch_size)
            P_gen = compute_joint_probabilities(
                chunk(X, batch_size),
                batch_size=batch_size,
                d=self.d,
                perplexity=self.perplexity,
                tol=self.tol,
                print_iter=self.print_iter,
                max_tries=self.max_tries,
                verbose=self.verbose
            )
            P = np.empty((n, batch_size, batch_size))
            for i, curP in enumerate(IT.islice(P_gen, n)):
                P[i] = curP
            Y = P.reshape((X.shape[0], -1))
            self.model.fit(
                X, Y,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                epochs=self.n_iter,
                verbose=self.verbose,
                batch_size=batch_size,
                callbacks=[
                    EarlyStopping(monitor='loss', mode='min', min_delta=1e-4,
                                  patience=self.n_iter_without_progress),
                ],
            )

    def transform(self, X, batch_count=None):
        if hasattr(X, '__next__'):
            if batch_count is None:
                raise Exception("For generator input, batch_count must "
                                "be specified")
            return map(self.model.predict, IT.islice(X, batch_count))
        else:
            return self.model.predict(X)

    def fit_transform(self, X, precalc_p=True, batch_count=None,
                      data_shape=None):
        self.fit(X, precalc_p=precalc_p, batch_count=batch_count,
                 data_shape=data_shape)
        return self.transform(X, batch_count=batch_count)
