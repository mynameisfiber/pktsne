from sklearn import datasets
from sklearn.manifold import TSNE
from pktsne import PTSNE
import numpy as np
import pylab as py

from utils import chunk, multiple_gaussians

from collections import defaultdict
import itertools as IT
import time


def run_comparison(name, X, color, perplexity=30, n_iter=1000):
    print(name, "ptsne")
    ptsne = (PTSNE(perplexity=perplexity, n_iter=n_iter, batch_size=1000)
             .fit_transform(X))
    py.clf()
    py.scatter(ptsne[:, 0], ptsne[:, 1], label='ptsne', c=color)
    py.title("PTSNE: " + name)
    py.savefig("test/{}_ptsne.png".format(name), dpi=300)

    print(name, "tsne")
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(X)
    py.clf()
    py.scatter(tsne[:, 0], tsne[:, 1], label='tsne', c=color)
    py.title("SciKit TSNE: " + name)
    py.savefig("test/{}_tsne.png".format(name), dpi=300)


def run_ptsne_modes(X):
    N = (X.shape[0] // 32) * 32
    p = {
        'n_iter': 10,
        'verbose': 0,
        'shuffle': False,
    }

    print("Test Normal")
    np.random.seed(18)
    ptsne_normal = PTSNE(**p).fit_transform(X)

    print("Test Generator nocache")
    np.random.seed(18)
    ptsne_gen = PTSNE(batch_size=32, **p).fit_transform(
        IT.cycle(chunk(X, 32)),
        batch_count=10,
        data_shape=X.shape[1:],
        precalc_p=False,
    )
    ptsne_gen = np.asarray(list(ptsne_gen)).reshape((N, -1))
    assert np.allclose(ptsne_normal, ptsne_gen)

    print("Test Generator cache")
    np.random.seed(18)
    ptsne_genc = PTSNE(batch_size=32, **p).fit_transform(
        IT.cycle(chunk(X, 32)),
        batch_count=10,
        data_shape=X.shape[1:],
        precalc_p=True,
    )
    ptsne_genc = np.asarray(list(ptsne_genc)).reshape((N, -1))
    assert np.allclose(ptsne_normal, ptsne_genc)


def test_curve():
    X, y = datasets.make_s_curve(n_samples=32*10, noise=.05)
    run_ptsne_modes(X)
    for perplexity in (5, 30, 50, 100):
        run_comparison("S_Curve_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)


def test_circles():
    X, y = datasets.make_circles(n_samples=32*10, noise=.05, factor=0.5)
    run_ptsne_modes(X)
    for perplexity in (5, 30, 50, 100):
        run_comparison("circle_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)


def test_gaussian():
    np.random.seed(18)
    X, y = multiple_gaussians(n_samples=32*10, n_dim=5, n_gaussians=3)
    for perplexity in (5, 30, 50, 100):
        run_comparison("gaussian_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)
    run_ptsne_modes(X)


def test_ptsne_modes_timing():
    N = 1024
    D = 5
    X, y = multiple_gaussians(n_samples=N, n_dim=D, n_gaussians=20)
    p = {
        'n_iter': 10,
        'verbose': 0,
    }

    batch_sizes = [32, 64, 128, 512]
    timings = defaultdict(list)
    for batch_size in batch_sizes:
        print("Batch size:", batch_size)

        print("Test Normal")
        np.random.seed(18)
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(X)
        timings['normal'].append((time.time() - start) / p['n_iter'])

        print("Test Generator nocache")
        np.random.seed(18)
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            IT.cycle(chunk(X, batch_size)),
            batch_count=N // batch_size,
            data_shape=(D,),
            precalc_p=False,
        )
        timings['gen_noc'].append((time.time() - start) / p['n_iter'])

        print("Test Generator cache")
        np.random.seed(18)
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            IT.cycle(chunk(X, batch_size)),
            batch_count=N // batch_size,
            data_shape=(D,),
            precalc_p=True,
        )
        timings['gen_c'].append((time.time() - start) / p['n_iter'])

    py.clf()
    for name, times in timings.items():
        py.plot(batch_sizes, times, label=name)
    py.xlabel("Batch Size")
    py.ylabel("Time per epoch")
    py.title("Timing per epoch for different run modes")
    py.yscale('log')
    py.savefig("test/run_mode_timings.png", dpi=300)


if __name__ == "__main__":
    # test_ptsne_modes_timing()
    test_gaussian()
    test_circles()
    test_curve()
