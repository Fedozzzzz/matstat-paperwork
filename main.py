import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxopt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

cvxopt.solvers.options['show_progress'] = False


def get_separable_data():
    x, y = datasets.make_blobs(n_samples=100,
                               centers=2,
                               n_features=2,
                               cluster_std=1.25,
                               random_state=9)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y[y == 0] = -1
    return x, y


def get_inseparable_data():
    x, y = datasets.make_blobs(n_samples=100,
                               centers=2,
                               n_features=2,
                               cluster_std=4.25,
                               random_state=9)
    min_max_scaler = preprocessing.MinMaxScaler()
    x = min_max_scaler.fit_transform(x)
    y[y == 0] = -1
    return x, y


def get_nonlinear_data():
    x, y = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=0)
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
        self.support_vectors_unbounded = None
        self.support_vectors_bounded = None

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

        eps = 1e-3
        sv_mask_unbounded = ((alpha > eps) & (alpha < self.c - eps)).flatten()
        sv_mask_all = (alpha > eps).flatten()
        sv_mask_bounded = (alpha >= self.c - eps).flatten()

        self.support_vectors = x[sv_mask_all]
        sv_y = y[sv_mask_all]
        alpha_sv = alpha[sv_mask_all]

        self.support_vectors_unbounded = x[sv_mask_unbounded]
        self.support_vectors_bounded = x[sv_mask_bounded]
        svU_y = y[sv_mask_unbounded]

        self.w = np.array(
            [sum(alpha_sv[i] * sv_y[i] * self.support_vectors[i, 0] for i in range(len(self.support_vectors))),
             sum(alpha_sv[i] * sv_y[i] * self.support_vectors[i, 1] for i in
                 range(len(self.support_vectors)))]).flatten()
        print('self.w', self.w)
        self.margin = 2 / np.linalg.norm(self.w)
        print('self.margin', self.margin)

        print('USV', len(self.support_vectors_unbounded))
        print('BSV', len(self.support_vectors_bounded))
        self.b = np.mean(
            [svU_y[i] - np.dot(self.support_vectors_unbounded[i], self.w) for i in
             range(len(self.support_vectors_unbounded))])
        print('self.b', self.b)

    def predict(self, xt):
        return np.array([np.dot(self.w, xt[i]) + self.b for i in range(len(xt))])

    def predict_sign(self, xt):
        return [np.sign(np.dot(self.w, xt[i]) + self.b) for i in range(len(xt))]


class SVMNonLinear:
    def __init__(self, c, gamma):
        self.gamma = gamma
        self.w = None
        self.margin = None
        self.alpha = None
        self.b = None
        self.c = c
        self.support_vectors = None

    def rbf(self, x1, x2):
        sigma = None
        if self.gamma == 'scale':
            sigma = 2 * self.x.var()
            print('self.x.var():', self.x.var())
        elif self.gamma == 'auto':
            sigma = 2
        elif type(self.gamma) is float:
            sigma = self.gamma
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
        sv_mask_bounded = (self.alpha >= self.c - eps).flatten()

        self.support_vectors = x[self.sv_mask_all]
        self.sv_y = y[self.sv_mask_all]
        self.alpha_sv = self.alpha[self.sv_mask_all]

        self.support_vectors_unbounded = x[self.sv_mask_unbounded]
        self.support_vectors_bounded = x[sv_mask_bounded]
        svU_y = y[self.sv_mask_unbounded]

        print('USV', len(self.support_vectors_unbounded))
        print('BSV', len(self.support_vectors_bounded))

        self.b = np.mean([svU_y[i] - sum(
            [self.alpha_sv[j] * self.sv_y[j] * self.rbf(self.support_vectors_unbounded[i], self.support_vectors[j])
             for j in range(len(self.support_vectors))]) for i in range(len(svU_y))])

        print('self.b', self.b)

    def predict(self, xt):
        return np.array([sum([self.alpha_sv[j] * self.sv_y[j] * self.rbf(xt[i], self.support_vectors[j]) for j in
                              range(len(self.support_vectors))]) + self.b for i in range(len(xt))])

from matplotlib.pyplot import text

def plot_graph(x, y, model, c=None, g=None, is_hard_margin=False):
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

    if c is not None and g is not None:
        plt.title('C={}, sigma={}'.format(c, g))
    elif c is not None:
        plt.title('C={}'.format(c))

    if is_hard_margin is True:
        ax.scatter(model.support_vectors[:, 0],
                   model.support_vectors[:, 1],
                   s=100,
                   linewidth=1.5,
                   facecolors='none',
                   edgecolors='k',
                   label='sv')
        # ax.legend()
        return plt.show()

    ax.scatter(model.support_vectors_bounded[:, 0],
               model.support_vectors_bounded[:, 1],
               s=100,
               linewidth=1.5,
               facecolors='none',
               edgecolors='r',
               label='bsv')

    ax.scatter(model.support_vectors_unbounded[:, 0],
               model.support_vectors_unbounded[:, 1],
               s=100,
               linewidth=1.5,
               facecolors='none',
               edgecolors='k',
               label='usv')

    ax.legend()

    plt.show()


# LINEAR SVM (HARD MARGIN)
svm_hard = SVMHardMargin()
x, y = get_separable_data()
svm_hard.train(x, y)
plot_graph(x, y, svm_hard, is_hard_margin=True)

# LINEAR SVM (SOFT MARGIN)
svm_soft = SVMSoftMargin(3.0)
x, y = get_inseparable_data()
svm_soft.train(x, y)
plot_graph(x, y, svm_soft, c=3.0)
#
# NON LINEAR SVM
svm_nonlinear = SVMNonLinear(2.0, 'scale')
x, y = get_nonlinear_data()
svm_nonlinear.train(x, y)
plot_graph(x, y, svm_nonlinear)
#
# C_arr = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
# for c in C_arr:
#     print("C={}".format(c))
#     svm_soft = SVMSoftMargin(c)
#     x, y = get_inseparable_data()
#     svm_soft.train(x, y)
#     plot_graph(x, y, svm_soft, c)

# print('-------------------------------------------------------------------------------')
# C_arr = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0]
# gamma_arr = ['scale', 'auto', 0.01, 0.1, 1.0]
# for g in gamma_arr:
#     for c in C_arr:
#         print("C={}, gamma={}".format(c, g))
#         svm_nonlinear = SVMNonLinear(c, g)
#         x, y = get_nonlinear_data()
#         svm_nonlinear.train(x, y)
#         plot_graph(x, y, svm_nonlinear, c, g)
#     print('-------------------------------------------------------------------------------')

# # print("C={}, gamma={}".format(c, g))
# svm_nonlinear = SVMNonLinear(200.0, 'auto')
# x, y = get_nonlinear_data()
# svm_nonlinear.train(x, y)
# plot_graph(x, y, svm_nonlinear)

# svm_nonlinear = SVMNonLinear(10.0, 0.01)
# x, y = get_nonlinear_data()
# svm_nonlinear.train(x, y)
# plot_graph(x, y, svm_nonlinear)
#
# svm_nonlinear = SVMNonLinear(100.0, 0.01)
# x, y = get_nonlinear_data()
# svm_nonlinear.train(x, y)
# plot_graph(x, y, svm_nonlinear)
