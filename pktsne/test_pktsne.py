from sklearn import datasets
from sklearn.manifold import TSNE
from pktsne import PTSNE
import numpy as np
import pylab as py
import keras.backend as K

from utils import inf_chunk, fix_random_seed, multiple_gaussians, Timer

from collections import defaultdict
import time


def run_comparison(name, X, color, perplexity=30, n_iter=1000):
    batch_size = X.shape[0]
    batch_count = 1
    ptsne_params = {
        "perplexity": perplexity,
        "n_iter": n_iter,
        "batch_size": batch_size,
        # "verbose": 1,
    }

    f, axs = py.subplots(2, 2)

    with Timer(name + " ptsne"):
        fix_random_seed()
        ptsne = (PTSNE(**ptsne_params)
                 .fit_transform(X))
    axs[0, 0].scatter(ptsne[:, 0], ptsne[:, 1], label='ptsne', c=color)
    axs[0, 0].set_title("PTSNE")

    with Timer(name + " ptsne_gen"):
        fix_random_seed()
        ptsne = (PTSNE(**ptsne_params)
                 .fit_transform(inf_chunk(X, batch_size), batch_count=batch_count,
                                data_shape=X.shape[1:], precalc_p=True))
        ptsne = np.asarray(list(ptsne)).reshape((X.shape[0], -1))
    axs[0, 1].scatter(ptsne[:, 0], ptsne[:, 1], label='ptsne', c=color)
    axs[0, 1].set_title("PTSNE gen")

    #print(name, "ptsne_gen_noc")
    #fix_random_seed()
    #ptsne = (PTSNE(**ptsne_params)
    #         .fit_transform(inf_chunk(X, batch_size), batch_count=batch_count,
    #                        data_shape=X.shape[1:], precalc_p=False))
    #ptsne = np.asarray(list(ptsne)).reshape((X.shape[0], -1))
    #axs[1, 0].scatter(ptsne[:, 0], ptsne[:, 1], label='ptsne', c=color)
    #axs[1, 0].set_title("PTSNE gen_noc")

    with Timer(name + " tsne"):
        fix_random_seed()
        tsne = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(X)
    axs[1, 1].scatter(tsne[:, 0], tsne[:, 1], label='tsne', c=color)
    axs[1, 1].set_title("SciKit")

    py.setp([a.get_xticklabels() for a in axs.flatten()], visible=False)
    py.setp([a.get_yticklabels() for a in axs.flatten()], visible=False)
    f.tight_layout()
    f.savefig("figures/{}.png".format(name), dpi=300)


def run_ptsne_modes(X):
    if K.backend() == "tensorflow":
        print("Not checking ptsne runmodes in tensorflow because of it's "
              "failure to respect fixed seeds")
        return

    batch_size = 128
    N = (X.shape[0] // batch_size) * batch_size
    batch_count = N // batch_size
    p = {
        'n_iter': 10,
        'batch_size': batch_size,
        'verbose': 0,
        'shuffle': False,
    }

    with Timer("Test Normal"):
        fix_random_seed()
        ptsne_normal = PTSNE(**p).fit_transform(X)

    with Timer("Test Generator nocache"):
        fix_random_seed()
        ptsne_gen = PTSNE(**p).fit_transform(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=X.shape[1:],
            precalc_p=False,
        )
        ptsne_gen = np.asarray(list(ptsne_gen)).reshape((N, -1))

    with Timer("Test Generator cache"):
        fix_random_seed()
        ptsne_genc = PTSNE(**p).fit_transform(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=X.shape[1:],
            precalc_p=True,
        )
        ptsne_genc = np.asarray(list(ptsne_genc)).reshape((N, -1))

    assert np.allclose(ptsne_normal, ptsne_gen)
    assert np.allclose(ptsne_normal, ptsne_genc)


def test_curve():
    X, y = datasets.make_s_curve(n_samples=128*2, noise=.05)
    run_ptsne_modes(X)
    for perplexity in (5, 30, 50, 100):
        run_comparison("S_Curve_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)


def test_circles():
    X, y = datasets.make_circles(n_samples=128*2, noise=.05, factor=0.5)
    run_ptsne_modes(X)
    for perplexity in (5, 30, 50, 100):
        run_comparison("circle_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)


def test_gaussian():
    fix_random_seed()
    X, y = multiple_gaussians(n_samples=128*2, n_dim=5, n_gaussians=3)
    run_ptsne_modes(X)
    for perplexity in (5, 30, 50, 100):
        run_comparison("gaussian_{}".format(perplexity), X, color=y,
                       perplexity=perplexity)


def test_ptsne_N_timing():
    D = 3
    p = {
        'n_iter': 100,
        'verbose': 0,
        'perplexity': 50,
    }
    batch_size = 128
    Ns = (128, 256, 512, 1024, 8192)
    timings = defaultdict(list)
    for N in Ns:
        batch_count = N // batch_size
        X, y = multiple_gaussians(n_samples=N, n_dim=D, n_gaussians=20)

        fix_random_seed()
        start = time.time()
        TSNE(**{**p, 'n_iter': 250}).fit(X)
        timings['sklearn'].append((time.time() - start) / 250)
        print("SKLearn Time: {:0.4f}s".format(timings['sklearn'][-1]))

        print("Test Normal:", N)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(X)
        timings['normal'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['normal'][-1]))

        print("Test Generator nocache:", N)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=(D,),
            precalc_p=False,
        )
        timings['gen_noc'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['gen_noc'][-1]))

        print("Test Generator cache:", N)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=(D,),
            precalc_p=True,
        )
        timings['gen_c'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['gen_c'][-1]))

    py.clf()
    for name, times in timings.items():
        py.plot(Ns, times, label=name)
    py.legend()
    py.xlabel("Data Size")
    py.ylabel("Time per epoch")
    py.title("Timing per epoch for different data sizes")
    py.yscale('log')
    py.savefig("figures/run_mode_N_timings.png", dpi=300)


def test_ptsne_batch_timing():
    N = 512
    D = 3
    X, y = multiple_gaussians(n_samples=N, n_dim=D, n_gaussians=20)
    p = {
        'n_iter': 25,
        'verbose': 0,
        'perplexity': 50,
    }

    fix_random_seed()
    start = time.time()
    TSNE(**{**p, 'n_iter': 250}).fit(X)
    sklearn_time = (time.time() - start) / 250
    print("SKLearn Time: {:0.4f}s".format(sklearn_time))

    batch_sizes = [32, 64, 128, 512]
    timings = defaultdict(list)
    for batch_size in batch_sizes:
        batch_count = N // batch_size
        print("Batch size:", batch_size)

        print("Test Normal:", batch_size)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(X)
        timings['normal'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['normal'][-1]))

        print("Test Generator nocache:", batch_size)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=(D,),
            precalc_p=False,
        )
        timings['gen_noc'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['gen_noc'][-1]))

        print("Test Generator cache:", batch_size)
        fix_random_seed()
        start = time.time()
        PTSNE(batch_size=batch_size, **p).fit(
            inf_chunk(X, batch_size),
            batch_count=batch_count,
            data_shape=(D,),
            precalc_p=True,
        )
        timings['gen_c'].append((time.time() - start) / p['n_iter'])
        print("Time: {:0.4f}s".format(timings['gen_c'][-1]))

    py.clf()
    for name, times in timings.items():
        py.plot(batch_sizes, times, label=name)
    py.axhline(y=sklearn_time, label="sklearn")
    py.legend()
    py.xlabel("Batch Size")
    py.ylabel("Time per epoch")
    py.title("Timing per epoch for different batch sizes")
    py.yscale('log')
    py.savefig("figures/run_mode_batch_timings.png", dpi=300)


if __name__ == "__main__":
    test_curve()
    test_gaussian()
    test_circles()
    test_ptsne_N_timing()
    test_ptsne_batch_timing()
