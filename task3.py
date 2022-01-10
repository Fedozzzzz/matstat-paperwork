import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # for data visualization
import seaborn as sns
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/FedozZz/PycharmProjects/matstat-paperwork_1/datasets/pulsar_stars.csv')

print(df.shape)

# let's preview the dataset

print(df.head())

col_names = df.columns

print(col_names)

df.columns = df.columns.str.strip()

df.columns = ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness',
              'DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']

df['target_class'].value_counts()

# X = df.drop(['target_class'], axis=1)
#
# y = df['target_class']

df_positive = df[df['target_class'] == 1]
df_negative = df[df['target_class'] == 0]

n = df_positive.size
print(n)
df_negative = df_negative.head(1639)
print(df_negative.size)

df_final = pd.concat([df_positive, df_negative])

X = df_final.drop(['target_class'], axis=1)
y = df_final['target_class']

print(df_final.size)
# print(y)
# split X and y into training and testing sets

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
C_arr = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
gamma = [1e-05, 1e-04, 1e-03,  0.01, 0.1,  1.0]
for g in gamma:
    print("gamma: {}".format(g))
    for c in C_arr:
        print("C: {}".format(c))
        rbf_svc = SVC(kernel='rbf', C=c, gamma=g)
        rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
        # print cross-validation scores with rbf kernel
        # print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))
        print('Cross-validation accuracy:{:.4f}'.format(rbf_scores.mean()))
    print("------------------------------------------------------------------")

print("-----------------------LINEAR---------------------------")
for c in C_arr:
    print("C: {}".format(c))
    rbf_svc = SVC(C=c)
    rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
    # print cross-validation scores with rbf kernel
    # print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))
    print('Cross-validation accuracy:{:.4f}'.format(rbf_scores.mean()))
print("------------------------------------------------------------------")

# print("-----------------------POLYNOMIAL---------------------------")
# gamma = [2, 3, 4, 5, 6]
# for g in gamma:
#     print("gamma: {}".format(g))
#     for c in C_arr:
#         print("C: {}".format(c))
#         rbf_svc = SVC(kernel='poly', C=c, gamma=g)
#         rbf_scores = cross_val_score(rbf_svc, X, y, cv=kfold)
#         # print cross-validation scores with rbf kernel
#         # print('Stratified Cross-validation scores with rbf kernel:\n\n{}'.format(rbf_scores))
#         print('Cross-validation accuracy:{:.4f}'.format(rbf_scores.mean()))
#     print("------------------------------------------------------------------")

C = 1000.0
gamma = 0.01
rbf_svc = SVC(kernel='rbf', C=C, gamma=gamma)

rbf_svc.fit(X_train, y_train)
y_pred = rbf_svc.predict(X_test)
print("Rbf Accuracy:{}".format(accuracy_score(y_test, y_pred)))


C = 10.0
linear_svc = SVC(C=C)
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
print("Linear Accuracy:{}".format(accuracy_score(y_test, y_pred)))
# import SVC classifier

# import metrics to compute accuracy
# from sklearn.metrics import accuracy_score
#
# # instantiate classifier with default hyperparameters
# svc = SVC()
#
# # fit classifier to training set
# svc.fit(X_train, y_train)
#
# # make predictions on test set
# y_pred = svc.predict(X_test)
#
# # compute and print accuracy score
# print('Model accuracy score with default hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
#
# # instantiate classifier with rbf kernel and C=100
# svc = SVC(C=100.0)
#
# # fit classifier to training set
# svc.fit(X_train, y_train)
#
# # make predictions on test set
# y_pred = svc.predict(X_test)
#
# # compute and print accuracy score
# print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
#
# # instantiate classifier with linear kernel and C=1.0
# linear_svc = SVC(kernel='linear', C=1.0)
#
# # fit classifier to training set
# linear_svc.fit(X_train, y_train)
#
# # make predictions on test set
# y_pred_test = linear_svc.predict(X_test)
#
# # compute and print accuracy score
# print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'.format(accuracy_score(y_test, y_pred_test)))

# linear_svc = SVC(kernel='linear')
#
# linear_scores = cross_val_score(linear_svc, X, y, cv=kfold)
#
# # print cross-validation scores with linear kernel
#
# print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(linear_scores))
#
# # print average cross-validation score with linear kernel
#
# print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(linear_scores.mean()))
