import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxopt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles, make_moons

cvxopt.solvers.options['show_progress'] = False


def get_separable_data():
    x, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.25, random_state=9)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y[y == 0] = -1
    return x, y


def get_inseparable_data():
    x, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=3.25, random_state=9)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y[y == 0] = -1
    return x, y


def get_nonlinear_data():
    x, y = datasets.make_classification(n_samples=200,
                                        n_features=2,
                                        n_repeated=0,
                                        class_sep=1,
                                        n_redundant=0,
                                        random_state=6)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y[y == 0] = -1
    return x, y


# нелинейные
def no_lin_data_set(nn, dd=2, k=2):
    xx = np.zeros((nn * k, dd))
    yy = np.zeros(nn * k)
    for i in range(k):
        ii = range(nn * i, nn * (i + 1))
        rad = np.linspace(0.0, 1, nn)
        theta = np.linspace(i * 4, (i + 1) * 4, nn) + np.random.randn(nn) * 0.2
        xx[ii] = np.c_[rad * np.sin(theta), rad * np.cos(theta)]
        yy[ii] = i
    yy[yy == 0] -= 1
    return xx, yy


class SVMHardMargin:
    def __init__(self):
        self.w = None
        self.margin = None
        self.b = None
        self.support_vectors = None

    def train(self, x, y):
        x_shape = len(x)
        P = cvxopt.matrix([[y[i] * y[j] * np.dot(x[i, :], x[j, :]) for j in range(x_shape)] for i in range(x_shape)])
        q = cvxopt.matrix(-np.ones(x_shape))
        A = cvxopt.matrix(np.array(y), (1, x_shape), 'd')
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(-np.eye(x_shape))
        h = cvxopt.matrix(np.zeros(x_shape))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])
        self.w = np.array([sum(alpha[i] * y[i] * x[i, 0] for i in range(x_shape)),
                           sum(alpha[i] * y[i] * x[i, 1] for i in range(x_shape))]).flatten()
        print('self.w', self.w)
        sv = (alpha > 1e-3).flatten()
        self.support_vectors = x[sv]
        sv_y = y[sv]
        self.margin = 2 / np.linalg.norm(self.w)
        self.b = np.mean(
            [sv_y[i] - np.dot(self.support_vectors[i], self.w) for i in range(len(self.support_vectors))])
        print('self.margin', self.margin)
        print('self.b', self.b)
        print('self.support_vectors', self.support_vectors)

    def predict(self, xt):
        return np.array([np.dot(self.w, xt[i]) + self.b for i in range(len(xt))])

    def predict_sign(self, xt):
        return [np.sign(np.dot(self.w, xt[i]) + self.b) for i in range(len(xt))]


class SVMSoftMargin:
    def __init__(self, c):
        self.w = None
        self.margin = None
        self.b = None
        self.c = c
        self.support_vectors = None

    def train(self, x, y):
        x_shape = len(x)
        P = cvxopt.matrix([[y[i] * y[j] * np.dot(x[i, :], x[j, :]) for j in range(x_shape)] for i in range(x_shape)])
        q = cvxopt.matrix(-np.ones(x_shape))
        A = cvxopt.matrix(np.array(y), (1, x_shape), 'd')
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.eye(x_shape), np.eye(x_shape))))
        h = cvxopt.matrix(np.vstack(np.hstack((np.zeros(x_shape), np.ones(x_shape) * self.c))))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])
        self.w = np.array([sum(alpha[i] * y[i] * x[i, 0] for i in range(x_shape)),
                           sum(alpha[i] * y[i] * x[i, 1] for i in range(x_shape))]).flatten()
        print('self.w', self.w)
        self.margin = 2 / np.linalg.norm(self.w)
        print('self.margin', self.margin)
        eps = 1e-3
        sv_mask = ((alpha > eps) & (alpha < self.c - eps)).flatten()
        self.support_vectors = x[sv_mask]
        print('self.support_vectors', self.support_vectors)
        sv_y = y[sv_mask]
        self.b = np.mean(
            [sv_y[i] - np.dot(self.support_vectors[i], self.w) for i in range(len(self.support_vectors))])
        print('self.b', self.b)

    def predict(self, xt):
        return np.array([np.dot(self.w, xt[i]) + self.b for i in range(len(xt))])

    def predict_sign(self, xt):
        return [np.sign(np.dot(self.w, xt[i]) + self.b) for i in range(len(xt))]


class SVMNonLinear:
    def __init__(self, c):
        self.w = None
        self.margin = None
        self.alpha = None
        self.b = None
        self.c = c
        self.support_vectors = None

    def rbf(self, x1, x2):
        sigma = 2 * self.x.var()
        # return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / (2 * (sigma ** 2)))
        return np.exp(-(np.linalg.norm(x1 - x2) ** 2) / sigma)

    def train(self, x, y):
        self.y = y
        self.x = x
        x_shape = len(x)
        P = cvxopt.matrix([[y[i] * y[j] * self.rbf(x[i], x[j]) for j in range(x_shape)] for i in range(x_shape)])
        q = cvxopt.matrix(-np.ones((x_shape, 1)))
        A = cvxopt.matrix(np.array(y), (1, x_shape), 'd')
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.eye(x_shape), np.eye(x_shape))))
        h = cvxopt.matrix(np.hstack((np.zeros(x_shape), np.ones(x_shape) * self.c)))
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x'])
        eps = 1e-6
        self.sv_mask_unbounded = ((self.alpha > eps) & (self.alpha < self.c - eps)).flatten()
        self.sv_mask_all = (self.alpha > eps).flatten()

        self.support_vectors = x[self.sv_mask_all]
        self.sv_y = y[self.sv_mask_all]
        self.alpha_sv = self.alpha[self.sv_mask_all]

        self.support_vectors_unbounded = x[self.sv_mask_unbounded]
        svU_y = y[self.sv_mask_unbounded]

        self.b = np.mean([svU_y[i] - sum(
            [self.alpha_sv[j] * self.sv_y[j] * self.rbf(self.support_vectors_unbounded[i], self.support_vectors[j])
             for j in range(len(self.support_vectors))]) for i in range(len(svU_y))])

        print('self.b', self.b)

    def predict(self, xt):
        return np.array([sum([self.alpha_sv[j] * self.sv_y[j] * self.rbf(xt[i], self.support_vectors[j]) for j in
                              range(len(self.support_vectors))]) + self.b for i in range(len(xt))])

    def predict_sign(self, xt):
        return np.array([np.sign(np.dot(self.w, xt[i]) + self.b) for i in range(len(xt))])


def plot_graph(x, y, model, without_sv=False):
    ax = plt.gca()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='winter')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z,
               colors='k',
               levels=[-1, 0, 1],
               alpha=0.5,
               linestyles=['--', '-', '--'])

    if without_sv is False:
        ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=100,
                   linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


svm_hard = SVMHardMargin()
x, y = get_separable_data()
svm_hard.train(x, y)
plot_graph(x, y, svm_hard)

svm_soft = SVMSoftMargin(2.0)
x, y = get_inseparable_data()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

svm_soft.train(X_train, y_train)
plot_graph(X_train, y_train, svm_soft)
plot_graph(X_test, y_test, svm_soft, without_sv=True)

svm_nonlinear = SVMNonLinear(2.0)
x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=0)
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
y[y == 0] = -1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

svm_nonlinear.train(X_train, y_train)
plot_graph(X_train, y_train, svm_nonlinear)
plot_graph(X_test, y_test, svm_nonlinear, without_sv=True)
