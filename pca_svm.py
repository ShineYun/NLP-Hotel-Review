#!/usr/bin/env python
# -*- coding: utf-8  -*-
# PCA  SVM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from classsify_model_selection import modelSelection
# 获取数据 [1995 rows x 400 columns]
df = pd.read_csv('2000_data.csv')
y = df.iloc[:, 1]
x = df.iloc[:, 2:]
review_data = pd.read_csv('./review_WordVec_data.csv')
review_label = pd.read_csv('./sentiment_model/review_data/review_hotel_0312.csv')['Sentiment_status']
# PCA降维
##计算全部贡献率
n_components = 400
pca = PCA(n_components=n_components)
pca.fit(x)
# print pca.explained_variance_ratio_

##PCA作图
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
plt.show()

##根据图形取100维
x_pca = PCA(n_components=100).fit_transform(x)
review_data_pca = pd.DataFrame(PCA(n_components=100).fit_transform(review_data))

# split train/test datasets
x_train, x_test, y_train, y_test = train_test_split(review_data_pca, review_label, train_size=0.8, random_state=1)

best_model = modelSelection(x_train,y_train,x_test,y_test)
# SVM (RBF)
# using training data with 100 dimensions
# clf_rf = RandomForestClassifier()
# clf_rf.fit(x_train, y_train)
# print('score:',clf_rf.score(x_test,y_test))
# clf = svm.SVC(C = 2, probability = True)
# clf.fit(x_pca,y)

# print('review score:',clf_rf.score(review_data_pca,review_label))
# test_pred = pd.DataFrame(clf_rf.predict(test_pca))
#
# print('Test Accuracy: %.2f' % clf_rf.score(x_pca, y))
# test_res = pd.concat([test_pca, test_pred], axis=1)
# test_res.to_csv('./review_res.csv')
# Create ROC curve on global dataset
pred_probas = best_model.predict_proba(review_data_pca)[:, 1]  # score

fpr, tpr, _ = metrics.roc_curve(review_label, pred_probas)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.title('ROC curve of total dataset')
plt.show()
