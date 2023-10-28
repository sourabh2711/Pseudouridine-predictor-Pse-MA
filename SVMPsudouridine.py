# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:05:38 2020

@author: win 10
"""


# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:20:06 2020

@author: win 10
"""

import csv
import numpy as np
from statistics import mean
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef, roc_auc_score
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score


from sklearn.naive_bayes import GaussianNB
import xgboost
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def loadCsv(filename):
    trainSet = []
    testSet = []
    lines = csv.reader(open(filename, 'r'))
    dataset = list(lines)
    #print("training set {}".format(dataset[0]))
    for i in range(len(dataset[0])-1):
            for row in dataset:
                    try:
                            row[i] = float(row[i])
                    except ValueError:
                            print("Error with row",i,":",row[i])
                            pass
                    row[-1]=int(float(row[-1]))
    trainSet = dataset        
    return trainSet

def gen_non_lin_separable_data():
    filename = 'S3_run.csv'

    trainingSet = loadCsv(filename)
    trainingSet=np.array(trainingSet)
    
    X1 = trainingSet[:, 0:74]  
    y1 = [row[-1] for row in trainingSet]
    print(X1[0].shape)
    return X1, y1    # use this to return just X and y



    # X = trainingSet[:, 0:22] 
    # y = trainingSet[:, -1]
    # return X,y
    # # filename = 'train_S2_1.csv'

    # trainingSet = loadCsv(filename)
    # trainingSet=np.array(trainingSet)
    
    # X1 = trainingSet[:, 0:22]  # we only take the first two features.
    # y1 = [int(row[-1]) for row in trainingSet]    
    
    # filename = 'test_S2_1.csv'

    # testSet = loadCsv(filename)
    # testSet=np.array(testSet)
    # X2 = testSet[:, 0:22]  # we only take the first two features.
    # y2 = [int(row[-1]) for row in testSet]    
    # return X1, y1, X2, y2                    # use this to return the 4 diff components of X and y

# def split_train(X1, y1, X2, y2):
#     X1_train = X1[:50]
#     y1_train = y1[:50]
#     X2_train = X2[:50]
#     y2_train = y2[:50]
    
#     X_train = np.vstack((X1_train, X2_train))
#     y_train = np.hstack((y1_train, y2_train))
    
#     return X_train, y_train

# def split_test(X1, y1, X2, y2):
#     X1_test = X1[50:]
#     y1_test = y1[50:]
#     X2_test = X2[50:]
#     y2_test = y2[50:]
#     X_test = np.vstack((X1_test, X2_test))
#     y_test = np.hstack((y1_test, y2_test))
#     return X_test, y_test



#This is same as Polynomial kernel
# def polyKernel(x1, x2, p=2):
#     x1 = x1.flatten()
#     x2 = x2.flatten()
#     sim = (1 + np.dot(x1, x2)) ** p
#     return sim



#cobbDKernelone is the utility kernel 
# k0= 1.1 and k1=a; p=2 is the degree alpha of the utility kernel
def utilitykernel(x1, x2, a, p=2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    #print(a)
    sim = 1.1 + (a*np.dot(x1, x2)**p )
    return sim

def gaussianKernelGramMatrix(X1, X2, a, K_function=utilitykernel):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, a)
    return gram_matrix


#X1, y1, X2, y2 = gen_non_lin_separable_data()
X, y = gen_non_lin_separable_data()
#X_train, y_train = split_train(X1, y1, X2, y2)
#X_test, y_test = split_test(X1, y1, X2, y2)
#X, y = gen_non_lin_separable_data()
#for i in range(20,22):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)
#print('\n i={}'.format(i))

# filename = 'X_train.csv'
# trainingSet = loadCsv(filename)
# X_train=np.array(trainingSet)

# filename = 'X_test.csv'
# trainingSet = loadCsv(filename)
# X_test=np.array(trainingSet)

# filename = 'y_train.csv'
# trainingSet = loadCsv(filename)
# y_train=np.array(trainingSet)

# filename = 'y_test.csv'
# trainingSet = loadCsv(filename)
# y_test=np.array(trainingSet)


#SVM
C=1.2
clf = svm.SVC(C = C, kernel="precomputed")


#the value of 'a' controls K1 of the utility kernel
for a in np.arange(90.2,90.3,0.9):
    print('\n a ={}'.format(a))
    model = clf.fit( gaussianKernelGramMatrix(X_train,X_train, a), y_train)
    p_test = model.predict(gaussianKernelGramMatrix(X_test, X_train, a))
    acc = accuracy_score(y_test, p_test)
    print('Accuracy: %.3f' % acc)

    
    cm = confusion_matrix(y_test, p_test)
    print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
    #confusionmatrix = np.matrix(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    #print('False Positives\n {}'.format(FP))
    #print('False Negetives\n {}'.format(FN))

    #print('True Positives\n {}'.format(TP))
    #print('True Negetives\n {}'.format(TN))
    TPR = TP/(TP+FN)
    print('Sensitivity {} \nAverage sensitivity {}'.format(TPR, mean(TPR)))
    TNR = TN/(TN+FP)
    print('Specificity {} \nAverage specificity {}'.format(TNR, mean(TNR)))
    Precision = TP/(TP+FP)
    #print('Precision \n {}'.format(Precision))
    Recall = TP/(TP+FN)
    #print('Recall \n {}'.format(Recall))
    Acc = (TP+TN)/(TP+TN+FP+FN)
    #print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
    Fscore = 2*(Precision*Recall)/(Precision+Recall)
    #print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
#=============================================================================
    # correct = np.sum(p_test == y_test)
    # print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
    # k=cohen_kappa_score(y_test, p_test)
    # print('Çohen Kappa \n{}'.format(k))
    m = matthews_corrcoef(y_test, p_test)
    print('Mathews Correlation Coeff \n{}'.format(m))
    roc_auc = roc_auc_score(y_test, p_test)
    #print('AUC \n{}'.format(roc_auc))
    # g = (Precision*Recall)**(1/2)
    # print('G-measure \n{}'.format(g))
#=============================================================================


# Gaussian NB
# gnb = GaussianNB()

# p_test = gnb.fit(X_train, y_train).predict(X_test)
# acc = accuracy_score(y_test, p_test)
# print('Accuracy: %.3f' % acc)
# cm = confusion_matrix(y_test, p_test)
# print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
# #confusionmatrix = np.matrix(cm)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# #print('False Positives\n {}'.format(FP))
# #print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
# print('Sensitivity {} \nAverage sensitivity {}'.format(TPR, mean(TPR)))
# TNR = TN/(TN+FP)
# print('Specificity {} \nAverage specificity {}'.format(TNR, mean(TNR)))
# Precision = TP/(TP+FP)
# #print('Precision \n {}'.format(Precision))
# Recall = TP/(TP+FN)
# #print('Recall \n {}'.format(Recall))
# Acc = (TP+TN)/(TP+TN+FP+FN)
# print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
# Fscore = 2*(Precision*Recall)/(Precision+Recall)
# print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
# # =============================================================================
# #     correct = np.sum(p_test == y_test)
# #     print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
# #     k=cohen_kappa_score(y_test, p_test)
# #     print('Çohen Kappa \n{}'.format(k))
# m = matthews_corrcoef(y_test, p_test)
# print('Mathews Correlation Coeff \n{}'.format(m))
# roc_auc = roc_auc_score(y_test, p_test)
# print('AUC \n{}'.format(roc_auc))


#Decision Tree
# clf = DecisionTreeClassifier(max_depth =3, random_state = 42)

# clf.fit(X_train, y_train)
# p_test = clf.predict(X_test)
# acc = accuracy_score(y_test, p_test)
# print('Accuracy: %.3f' % acc)
# cm = confusion_matrix(y_test, p_test)
# print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
# #confusionmatrix = np.matrix(cm)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# #print('False Positives\n {}'.format(FP))
# #print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
# print('Sensitivity {} \nAverage sensitivity {}'.format(TPR, mean(TPR)))
# TNR = TN/(TN+FP)
# print('Specificity {} \nAverage specificity {}'.format(TNR, mean(TNR)))
# Precision = TP/(TP+FP)
# #print('Precision \n {}'.format(Precision))
# Recall = TP/(TP+FN)
# #print('Recall \n {}'.format(Recall))
# Acc = (TP+TN)/(TP+TN+FP+FN)
# print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
# Fscore = 2*(Precision*Recall)/(Precision+Recall)
# print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
# # =============================================================================
# #     correct = np.sum(p_test == y_test)
# #     print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
# #     k=cohen_kappa_score(y_test, p_test)
# #     print('Çohen Kappa \n{}'.format(k))
# m = matthews_corrcoef(y_test, p_test)
# print('Mathews Correlation Coeff \n{}'.format(m))
# roc_auc = roc_auc_score(y_test, p_test)
# print('AUC \n{}'.format(roc_auc))


#Random Forest
# clf = RandomForestClassifier(max_depth=2, n_estimators = 2, random_state=1)
# clf.fit(X_train, y_train)
# p_test = clf.predict(X_test)
# acc = accuracy_score(y_test, p_test)
# print('Accuracy: %.3f' % acc)
# cm = confusion_matrix(y_test, p_test)
# print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
# #confusionmatrix = np.matrix(cm)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# #print('False Positives\n {}'.format(FP))
# #print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
# print('Sensitivity {} \nAverage sensitivity {}'.format(TPR, mean(TPR)))
# TNR = TN/(TN+FP)
# print('Specificity {} \nAverage specificity {}'.format(TNR, mean(TNR)))
# Precision = TP/(TP+FP)
# #print('Precision \n {}'.format(Precision))
# Recall = TP/(TP+FN)
# #print('Recall \n {}'.format(Recall))
# Acc = (TP+TN)/(TP+TN+FP+FN)
# print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
# Fscore = 2*(Precision*Recall)/(Precision+Recall)
# #print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
# # =============================================================================
# #     correct = np.sum(p_test == y_test)
# #     print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
# #     k=cohen_kappa_score(y_test, p_test)
# #     print('Çohen Kappa \n{}'.format(k))
# m = matthews_corrcoef(y_test, p_test)
# print('Mathews Correlation Coeff \n{}'.format(m))
# roc_auc = roc_auc_score(y_test, p_test)
# print('AUC \n{}'.format(roc_auc))



#XGBoost


# model = XGBClassifier()
# model.fit(X_train, y_train)
# p_test = model.predict(X_test)

# acc = accuracy_score(y_test, p_test)
# print('Accuracy: %.3f' % acc)
# cm = confusion_matrix(y_test, p_test)
# print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
# #confusionmatrix = np.matrix(cm)
# FP = cm.sum(axis=0) - np.diag(cm)
# FN = cm.sum(axis=1) - np.diag(cm)
# TP = np.diag(cm)
# TN = cm.sum() - (FP + FN + TP)
# #print('False Positives\n {}'.format(FP))
# #print('False Negetives\n {}'.format(FN))

# #print('True Positives\n {}'.format(TP))
# #print('True Negetives\n {}'.format(TN))
# TPR = TP/(TP+FN)
# print('Sensitivity {} \nAverage sensitivity {}'.format(TPR, mean(TPR)))
# TNR = TN/(TN+FP)
# print('Specificity {} \nAverage specificity {}'.format(TNR, mean(TNR)))
# Precision = TP/(TP+FP)
# #print('Precision \n {}'.format(Precision))
# Recall = TP/(TP+FN)
# #print('Recall \n {}'.format(Recall))
# Acc = (TP+TN)/(TP+TN+FP+FN)
# print('Test Áccuracy \n{} \nAverage Test Acc \n{}'.format(Acc, mean(Acc)))
# Fscore = 2*(Precision*Recall)/(Precision+Recall)
# #print('Test FScore \n{} \nAverage Test Fscore \n{}'.format(Fscore, mean(Fscore)))
# # =============================================================================
# #     correct = np.sum(p_test == y_test)
# #     print("CobbD (neg) sigma=0.1 p=9, k=1.1; %d out of %d predictions correct" % (correct, len(y_test)))
# #     k=cohen_kappa_score(y_test, p_test)
# #     print('Çohen Kappa \n{}'.format(k))
# m = matthews_corrcoef(y_test, p_test)
# print('Mathews Correlation Coeff \n{}'.format(m))
# roc_auc = roc_auc_score(y_test, p_test)
# print('AUC \n{}'.format(roc_auc))
