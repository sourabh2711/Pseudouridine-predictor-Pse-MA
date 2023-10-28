# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:35:45 2023

@author: Dell
"""

"""
Created on Sat Oct 21 17:01:53 2023

@author: Dell
"""

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
from sklearn.model_selection import KFold, cross_val_score


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
    #y1 = [row[-1] for row in trainingSet]
    y1 = trainingSet[:,-1]
    print(X1.shape)
    print(y1.shape)
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
    # return X1, y1, X2, y2                    
    # use this to return the 4 diff components of X and y

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
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=1)

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


C=1.0
clf = svm.SVC(C = C, kernel="precomputed")


#for i in range(0,43):
kf = KFold(n_splits = 5, shuffle=(True), random_state=34)
#    print('\n i={}'.format(i))
#the value of 'a' controls K1 of the utility kernel
a=544.4
#print(X_train[0])
#X = X.to_numpy()
sen_arr = np.empty((5, 1))
spe_arr = np.empty((5, 1))
acc_arr = np.empty((5, 1))
MCC_arr = np.empty((5, 1))
AUC_arr = np.empty((5, 1))
print('a = {}'.format(a))
x=0


for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # model = clf.fit( gaussianKernelGramMatrix(X_train,X_train, a), y_train)
    # p_test = model.predict(gaussianKernelGramMatrix(X_test, X_train, a))


    # Gaussian NB
    gnb = GaussianNB()
    p_test = gnb.fit(X_train, y_train).predict(X_test)
    
    
    #Decision Tree
    #clf = DecisionTreeClassifier(max_depth =3, random_state = 42)

    #clf.fit(X_train, y_train)
    #p_test = clf.predict(X_test)
    

    #Random Forest
    # clf = RandomForestClassifier(max_depth=2, n_estimators = 2, random_state=1)
    # clf.fit(X_train, y_train)
    #p_test = clf.predict(X_test)
    
    #XGBoost


    # model = XGBClassifier()
    # model.fit(X_train, y_train)
    # p_test = model.predict(X_test)



    
    #print('\n\n\n\nAccuracy Score: {:.4f}'.format(accuracy_score(y_test, p_test)))   
    
    #print("\n", classification_report(y_pred, y_test))
    cm = confusion_matrix(y_test, p_test)
    # # print('\n'.join([''.join(['{:5}'.format(item) for item in row]) for row in cm]))
    confusionmatrix = np.matrix(cm)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    TPR = TP/(TP+FN)
    #print('Sensitivity \n {}'.format(TPR))
    TNR = TN/(TN+FP)
    #print('Specificity \n {}'.format(TNR))
    m = matthews_corrcoef(y_test, p_test)
    #print('Mathews Correlation Coeff \n{}'.format(m))
    roc_auc = roc_auc_score(y_test, p_test)
    #print('AUC \n{}'.format(roc_auc))
    
    sen_arr[x] = mean(TPR)
    spe_arr[x] = mean(TNR)
    acc_arr[x] = accuracy_score(y_test, p_test)
    MCC_arr[x] = matthews_corrcoef(y_test, p_test)
    AUC_arr[x] = roc_auc_score(y_test, p_test)
    x=x+1
    
    
 
#print(sen_arr)
print("%0.4f sensitivity with a standard deviation of %0.2f" %(sen_arr.mean(), sen_arr.std()))
print("%0.4f specificity with a standard deviation of %0.2f" %(spe_arr.mean(), spe_arr.std())) 
print("%0.4f accuracy with a standard deviation of %0.2f" %(acc_arr.mean(), acc_arr.std())) 
print("%0.4f MCC with a standard deviation of %0.2f" %(MCC_arr.mean(), MCC_arr.std()))  
print("%0.4f AUC with a standard deviation of %0.2f" %(AUC_arr.mean(), AUC_arr.std()))
# y_true.append(y_test[0])
# y_pred.append(p_test[0])
    
