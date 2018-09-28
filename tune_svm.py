import pandas as pd
import numpy as np
from sklearn import svm,model_selection,grid_search,metrics
import matplotlib.pyplot as plt

#os.chdir('/Users/sambeet/Desktop/cn/')

labels_train = np.loadtxt('labels_train.txt', dtype=float)
labels_test = np.loadtxt('labels_test.txt', dtype=float)

all_features_train = np.loadtxt('all_features_train.txt', dtype=float)
all_features_test = np.loadtxt('all_features_test.txt', dtype=float)

bigram_features_train = np.loadtxt('bigram_features_train.txt', dtype=float)
bigram_features_test = np.loadtxt('bigram_features_test.txt', dtype=float)

stylistic_features_train = np.loadtxt('stylistic_features_train.txt',dtype=float)
stylistic_features_test = np.loadtxt('stylistic_features_test.txt',dtype=float)

param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3], 'C': [100,200,500,1000]}]
scoring = 'roc_auc'
clf = svm.SVC()
scores = model_selection.GridSearchCV(estimator=clf,param_grid=param_grid,cv=5,n_jobs=-1,refit=False,return_train_score=True,verbose=3,scoring=scoring)
scores.fit(all_features_train,labels_train)

train_scores_mean = scores.cv_results_['mean_train_score']
train_scores_std = scores.cv_results_['std_train_score']
test_scores_mean = scores.cv_results_['mean_test_score']
test_scores_std = scores.cv_results_['std_test_score']
lw = 2
param_range = range(1,5)
plt.figure().set_size_inches(8, 6)
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.show()

final_model = svm.SVC(C=100,kernel='rbf',gamma=1e-3)
final_model.fit(all_features_train,labels_train)

Y_pred = final_model.predict(X=all_features_test)
#metrics.roc_auc_score(y_score=final_model.predict_proba(X=all_features_test),y_true=labels_test)
metrics.accuracy_score(y_pred=Y_pred,y_true=labels_test)
metrics.confusion_matrix(y_pred=Y_pred,y_true=labels_test)
metrics.f1_score(y_pred=Y_pred,y_true=labels_test,average='micro')
metrics.precision_score(y_pred=Y_pred,y_true=labels_test)
metrics.recall_score(y_pred=Y_pred,y_true=labels_test)

dataset1['svm_pred'] = final_model.predict(X=all_features)
dataset1.to_csv('svm_output.csv',index=False)
