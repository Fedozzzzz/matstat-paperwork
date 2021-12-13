import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxopt
from sklearn import preprocessing

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
        sv_mask = ((1e-3 < alpha) & (alpha < self.c)).flatten()
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


def plot_graph(x, y, model):
    ax = plt.gca()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='winter')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.predict(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


svm_hard = SVMHardMargin()
x, y = get_separable_data()
svm_hard.train(x, y)
plot_graph(x, y, svm_hard)

svm_soft = SVMSoftMargin(2)
x, y = get_inseparable_data()
svm_soft.train(x, y)
plot_graph(x, y, svm_soft)
