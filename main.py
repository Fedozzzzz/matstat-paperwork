import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxopt

cvxopt.solvers.options['show_progress'] = False


class SvmHard:
    def __init__(self, c):
        self.C = c

    def train(self):
        return

    def predict(self):
        return

    def body(self, xx, yy):
        mm, nn = xx.shape
        yy = yy.reshape(-1, 1) * 1.
        xd = xx * yy
        h = np.dot(xd, xd.T) * 1.
        self.aa = cvxopt.matrix(-np.eye(mm))
        self.bb = cvxopt.matrix(np.zeros(mm))
        self.qq = cvxopt.matrix(np.zeros(1))
        self.gg = cvxopt.matrix(yy.reshape(1, -1))
        self.hh = cvxopt.matrix(-np.ones((mm, 1)))
        self.pp = cvxopt.matrix(h)
        sol = cvxopt.solvers.qp(self.pp, self.hh, self.aa, self.bb, self.gg, self.qq)
        a = np.array(sol['x'])
        self.weight = ((yy * a).T @ xx).reshape(-1, 1).flatten()
        print('Веса: ', self.weight)
        print('Марджин: ', 2 / np.linalg.norm(self.weight))
        ss = (a > 1e-4).flatten()
        self.bias = np.mean(yy[ss] - np.dot(xx[ss], self.weight))
        print('Разница между количеством целевого значения и прогнозом модели : ', self.bias)
        aa = np.ravel(sol['x'])
        svm = aa > 1e-5
        self.svmx = xx[svm]
        self.svmy = yy[svm]
        self.has_fitted = True
        return self.weight, self.bias

    def show_result(self, true_x, false_x):
        plt.plot(true_x[:, 0], true_x[:, 1], "ro")
        plt.plot(false_x[:, 0], false_x[:, 1], "mo")
        plt.scatter(self.svmx[:, 0], self.svmx[:, 1], s=100, c="c")
        xx = np.vstack((true_x, false_x))
        a0 = xx[:, 0].min() - 1
        b0 = xx[:, 0].max() - 1
        a1 = (-self.weight[0] * a0 - self.bias) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k")
        a1 = (-self.weight[0] * a0 - self.bias + 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias + 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        a1 = (-self.weight[0] * a0 - self.bias - 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias - 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        plt.axis("off")
        plt.title("Hard Margin")
        plt.show()


# генерация данных
# сепарабельные
def sep_data_set():
    x, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.25)
    true_x = x[np.where(y == 1)]
    false_x = x[np.where(y == 0)]
    return true_x, false_x


from cvxopt import matrix, solvers

# x, y = sep_data_set()

x = [[-2.24054747, -6.61741927], [-2.71068172, -8.0835143]]
y = [[11.32531171, 4.5041695], [8.87874737, 4.10529034]]

# print(x)
# print(y)

# A =
# p = matrix([1.0, 1.0])
G = matrix([[-1.0, 0.0], [0.0, -1.0]])
h = matrix([0.0, 0.0])
A = matrix([11.32531171, 4.5041695], (1, 2))
b = matrix(0.0)
Q = 2 * matrix(A.T * A)
p = - 2 * matrix(A.T * b)

sol = solvers.qp(Q, p, G, h, A, b)
print(sol['x'])

f = np.sign()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import cvxopt

cvxopt.solvers.options['show_progress'] = False


# генерация данных
# сепарабельные
def sep_data_set():
    x, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.25)
    true_x = x[np.where(y == 1)]
    false_x = x[np.where(y == 0)]
    return true_x, false_x


# несепарабельные
def no_sep_data_set():
    x, y = datasets.make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=3.25)
    true_x = x[np.where(y == 1)]
    false_x = x[np.where(y == 0)]
    return true_x, false_x


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


# ядро
def gaussian(xx, zz, sigma=0.1):
    return np.exp(-np.linalg.norm(xx - zz, axis=1) ** 2 / (2 * (sigma ** 2)))


# сами свм
# hard margin
class SvmHard:
    def __init__(self, c):
        self.C = c

    def body(self, xx, yy):
        mm, nn = xx.shape
        yy = yy.reshape(-1, 1) * 1.
        xd = xx * yy
        h = np.dot(xd, xd.T) * 1.
        self.aa = cvxopt.matrix(-np.eye(mm))
        self.bb = cvxopt.matrix(np.zeros(mm))
        self.qq = cvxopt.matrix(np.zeros(1))
        self.gg = cvxopt.matrix(yy.reshape(1, -1))
        self.hh = cvxopt.matrix(-np.ones((mm, 1)))
        self.pp = cvxopt.matrix(h)
        sol = cvxopt.solvers.qp(self.pp, self.hh, self.aa, self.bb, self.gg, self.qq)
        a = np.array(sol['x'])
        self.weight = ((yy * a).T @ xx).reshape(-1, 1).flatten()
        print('Веса: ', self.weight)
        print('Марджин: ', 2 / np.linalg.norm(self.weight))
        ss = (a > 1e-4).flatten()
        self.bias = np.mean(yy[ss] - np.dot(xx[ss], self.weight))
        print('Разница между количеством целевого значения и прогнозом модели : ', self.bias)
        aa = np.ravel(sol['x'])
        svm = aa > 1e-5
        self.svmx = xx[svm]
        self.svmy = yy[svm]
        self.has_fitted = True
        return self.weight, self.bias

    def show_result(self, true_x, false_x):
        plt.plot(true_x[:, 0], true_x[:, 1], "ro")
        plt.plot(false_x[:, 0], false_x[:, 1], "mo")
        plt.scatter(self.svmx[:, 0], self.svmx[:, 1], s=100, c="c")
        xx = np.vstack((true_x, false_x))
        a0 = xx[:, 0].min() - 1
        b0 = xx[:, 0].max() - 1
        a1 = (-self.weight[0] * a0 - self.bias) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k")
        a1 = (-self.weight[0] * a0 - self.bias + 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias + 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        a1 = (-self.weight[0] * a0 - self.bias - 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias - 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        plt.axis("off")
        plt.title("Hard Margin")
        plt.show()


# soft margin
class SvmSoft:
    def __init__(self, c):
        self.C = c

    def body(self, xx, yy):
        mm, nn = xx.shape
        yy = yy.reshape(-1, 1) * 1.
        xd = xx * yy
        h = np.dot(xd, xd.T) * 1.
        self.aa = cvxopt.matrix(np.vstack((-np.eye(mm), np.eye(mm))))
        self.bb = cvxopt.matrix(np.hstack((np.zeros(mm), np.ones(mm) * self.C)))
        self.qq = cvxopt.matrix(np.zeros(1))
        self.gg = cvxopt.matrix(yy.reshape(1, -1))
        self.hh = cvxopt.matrix(-np.ones((mm, 1)))
        self.pp = cvxopt.matrix(h)
        sol = cvxopt.solvers.qp(self.pp, self.hh, self.aa, self.bb, self.gg, self.qq)
        a = np.array(sol['x'])
        self.weight = ((yy * a).T @ xx).reshape(-1, 1).flatten()
        print('Веса: ', self.weight)
        print('Марджин: ', 2 / np.linalg.norm(self.weight))
        ss = (a > 1e-4).flatten()
        self.bias = np.mean(yy[ss] - np.dot(xx[ss], self.weight))
        print('Разница между количеством целевого значения и прогнозом модели : ', self.bias)
        aa = np.ravel(sol['x'])
        svm = aa > 1e-5
        self.svmx = xx[svm]
        self.svmy = yy[svm]
        self.has_fitted = True
        return self.weight, self.bias

    def show_result(self, true_x, false_x):
        plt.plot(true_x[:, 0], true_x[:, 1], "go")
        plt.plot(false_x[:, 0], false_x[:, 1], "yo")
        plt.scatter(self.svmx[:, 0], self.svmx[:, 1], s=100, c="c")
        xx = np.vstack((true_x, false_x))
        a0 = xx[:, 0].min() - 1
        b0 = xx[:, 0].max() - 1
        a1 = (-self.weight[0] * a0 - self.bias) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k")
        a1 = (-self.weight[0] * a0 - self.bias + 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias + 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        a1 = (-self.weight[0] * a0 - self.bias - 1) / self.weight[1]
        b1 = (-self.weight[0] * b0 - self.bias - 1) / self.weight[1]
        plt.plot([a0, b0], [a1, b1], "k--")
        plt.axis("off")
        plt.title("Soft Margin")
        plt.show()


# нелинейные
class SvmNoL:
    def __init__(self, c=1):
        self.c = c
        self.kernel = gaussian

    def body(self, xx, yy):
        self.yy = yy
        self.xx = xx
        mm, nn = xx.shape
        self.K = np.zeros((mm, mm))
        for i in range(mm):
            self.K[i, :] = self.kernel(xx[i, np.newaxis], self.xx)
        self.pp = cvxopt.matrix(np.outer(yy, yy) * self.K)
        self.qq = cvxopt.matrix(-np.ones((mm, 1)))
        self.gg = cvxopt.matrix(np.vstack((np.eye(mm) * -1, np.eye(mm))))
        self.hh = cvxopt.matrix(np.hstack((np.zeros(mm), np.ones(mm) * self.c)))
        self.aa = cvxopt.matrix(yy, (1, mm), "d")
        self.bb = cvxopt.matrix(np.zeros(1))
        sol = cvxopt.solvers.qp(self.pp, self.qq, self.gg, self.hh, self.aa, self.bb)
        self.a = np.array(sol["x"])

    def get(self, a):
        eps = 1e-5
        sv = ((a > eps) * (a < self.c)).flatten()
        self.weight = np.dot(self.xx[sv].T, a[sv] * self.yy[sv, np.newaxis])
        print('Веса: ', self.weight)
        print('Марджин: ', 2 / np.linalg.norm(self.weight))
        self.bias = np.mean(self.yy[sv, np.newaxis] - self.a[sv] * self.yy[sv, np.newaxis] * self.K[sv, sv][:,
                                                                                             np.newaxis])
        print('Разница между количеством целевого значения и прогнозом модели : ', self.bias)
        return sv

    def predict(self, xx):
        y_p = np.zeros((xx.shape[0]))
        sv = self.get(self.a)
        for i in range(xx.shape[0]):
            y_p[i] = np.sum(self.a[sv] * self.yy[sv, np.newaxis] * self.kernel(xx[i], self.xx[sv])[:, np.newaxis])
        return np.sign(y_p + self.bias)

    def show_result(self):
        x_min, x_max = self.xx[:, 0].min() - 1, self.xx[:, 0].max() + 1
        y_min, y_max = self.xx[:, 1].min() - 1, self.xx[:, 1].max() + 1
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        points = np.c_[xx.ravel(), yy.ravel()]
        zz = self.predict(points)
        zz = zz.reshape(xx.shape)
        plt.contourf(xx, yy, zz, cmap='GnBu', alpha=0.8)
        plt.scatter(self.xx[:, 0], self.xx[:, 1], c=self.yy, s=40, cmap='tab20', edgecolors='black')
        plt.axis("off")
        plt.title("Нелинейные данные")
        plt.show()


True_x_ns, False_x_ns = no_sep_data_set()
True_x_s, False_x_s = sep_data_set()
np.random.seed(1)
Xnl, Ynl = no_lin_data_set(nn=200)
Xns = np.vstack((True_x_ns, False_x_ns))
Xs = np.vstack((True_x_s, False_x_s))
True_y_ns = np.full((len(True_x_ns),), 1)
True_y_s = np.full((len(True_x_s),), 1)
False_y_ns = np.full((len(False_x_ns),), -1)
False_y_s = np.full((len(False_x_s),), -1)
Yns = np.concatenate((True_y_ns, False_y_ns))
Ys = np.concatenate((True_y_s, False_y_s))
soft_SVM = SvmSoft(2)
hard_SVM = SvmHard(2)
nl_SVM = SvmNoL()
print("Hard Margin")
hard_SVM.body(Xs, Ys)
print("Soft Margin")
soft_SVM.body(Xns, Yns)
print("Нелинейное")
nl_SVM.body(Xnl, Ynl)
hard_SVM.show_result(True_x_s, False_x_s)
soft_SVM.show_result(True_x_ns, False_x_ns)
nl_SVM.show_result()
