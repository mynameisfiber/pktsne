from sklearn import datasets
from sklearn.manifold import TSNE
from pktsne import PTSNE
import pylab as py


def run_test(name, X, color, perplexity=30, n_iter=1000):
    print(name, "ptsne")
    ptsne = PTSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(X)
    py.clf()
    py.scatter(ptsne[:, 0], ptsne[:, 1], label='ptsne', c=color)
    py.savefig("test/{}_ptsne.png".format(name), dpi=300)

    print(name, "tsne")
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(X)
    py.clf()
    py.scatter(tsne[:, 0], tsne[:, 1], label='tsne', c=color)
    py.savefig("test/{}_tsne.png".format(name), dpi=300)


def test_curve():
    X, y = datasets.make_s_curve(n_samples=100, noise=.05)
    for perplexity in (5, 30, 50, 100):
        run_test("S_Curve_{}".format(perplexity), X, color=y,
                 perplexity=perplexity)


def test_circles():
    X, y = datasets.make_circles(n_samples=100, noise=.05, factor=0.5)
    for perplexity in (5, 30, 50, 100):
        run_test("circle_{}".format(perplexity), X, color=y,
                 perplexity=perplexity)


if __name__ == "__main__":
    test_circles()
    test_curve()
