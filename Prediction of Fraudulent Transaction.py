#Importing few libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

#Changing working director
os.chdir("...\\Credit Card")

#Reading the creditcard file
FullRaw = pd.read_csv(".....\\creditcard.csv")

FullRaw.head()
FullRaw.shape
#(284807, 31)

# Increase the print output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

#Checking for Missing Value
FullRaw.isnull().sum()
# =============================================================================
# #No Missing Value
# Time      0
# V1        0
# V2        0
# V3        0
# V4        0
# V5        0
# V6        0
# V7        0
# V8        0
# V9        0
# V10       0
# V11       0
# V12       0
# V13       0
# V14       0
# V15       0
# V16       0
# V17       0
# V18       0
# V19       0
# V20       0
# V21       0
# V22       0
# V23       0
# V24       0
# V25       0
# V26       0
# V27       0
# V28       0
# Amount    0
# Class     0
# =============================================================================

#Checking whether the dataset is balanced or imbalanced (% Split of "1" in case of Fraud and "0" otherwise i.e. Normal(say) in "Class" Column)
Split_Percentage = FullRaw["Class"].value_counts() * 100 / len(FullRaw)
# =============================================================================
# 0    99.827251
# 1     0.172749
# #creditcard datset is highly imbalanced
# =============================================================================

#Visualization of "Class" distribution in terms of historgram
FullRaw.Class.hist(bins=2)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Records")

#No. of Fraud and Normal Transaction
Fraud = FullRaw[FullRaw["Class"]==1]
#[492 rows x 31 columns]
Normal = FullRaw[FullRaw["Class"]==0]
#[284315 rows x 31 columns]

## As data has highly imbalanced one, it is quite difficult to find any significant variable, as we have two known variable (Time & Amount) for the Amount we can perform T-Test.
#Summary of the Fraud Txn 
Fraud["Amount"].describe()
# =============================================================================
# count     492.000000
# mean      122.211321
# std       256.683288
# min         0.000000
# 25%         1.000000
# 50%         9.250000
# 75%       105.890000
# max      2125.870000
# =============================================================================

#Summary of the Normal Txn
Normal["Amount"].describe()
# =============================================================================
# count    284315.000000
# mean         88.291022
# std         250.105092
# min           0.000000
# 25%           5.650000
# 50%          22.000000
# 75%          77.050000
# max       25691.160000
# =============================================================================

#Correlation check #For Logistic Regrssion not important
CorrDf = FullRaw.corr()
sns.heatmap(CorrDf)
#sns.heatmap(CorrDf, xticklabels= CorrDf.columns, yticklabels= CorrDf.columns, cmap = 'coolwarm_r')
#No considerable Correlation

#Divide FullRaw into Train and Test by random sampling:
from sklearn.model_selection import train_test_split
Train, Test = train_test_split(FullRaw, test_size=0.3, random_state = 123) # Split 70-30%

Train.shape
#(199364, 31)
Test.shape
#(85443, 31)

#Divide into Xs (Indepenedents) and Y (Dependent)
Train_X = Train.drop(['Class'], axis = 1).copy()
Train_Y = Train['Class'].copy()

Test_X = Test.drop(['Class'], axis = 1).copy()
Test_Y = Test['Class'].copy()

Train_X.shape #(199364, 30)
Test_X.shape #(85443, 30)

###############################################################################
# Importing SMOTE to handle the class imbalanced problem
from imblearn.over_sampling import SMOTE

Train_Y.value_counts() #Counting the No. of 0s and 1s
# =============================================================================
# 0    199032
# 1       332
# # Imbalanced Class
# =============================================================================

smt = SMOTE(random_state = 12)
Train_X, Train_Y = smt.fit_sample(Train_X, Train_Y)

Train_Y.value_counts() #Counting the No. of 0s and 1s after SMOTE
# =============================================================================
# 1    199032
# 0    199032
# # Now the calss is balanced
# =============================================================================

# =============================================================================
# # Model building using logistic regression after SMOTE
# =============================================================================
from statsmodels.api import Logit
import statsmodels.api as sm
Train_X = sm.add_constant(Train_X)
Test_X = sm.add_constant(Test_X)

M1 = Logit(Train_Y, Train_X) #Model Defination
M1_Model = M1.fit() #Model Building
M1_Model.summary() #Model Output/Summary

# Prediction and Validation
Test_X['Test_Prob'] = M1_Model.predict(Test_X) # Store probability predictions in "Text_X" df
Test_X.columns

# Classify 0 or 1 based on 0.5 cutoff
Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)
Test_X.columns
#Test_X['Test_Class'].value_counts() / len(Test_X)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Test_X['Test_Class'], Test_Y))
#0.9914445887901876

print(confusion_matrix(Test_X['Test_Class'], Test_Y))
# =============================================================================
# [[84568    16]
#  [  715   144]]
# =============================================================================

print(classification_report(Test_X['Test_Class'], Test_Y))
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       0.99      1.00      1.00     84584
#            1       0.90      0.17      0.28       859
# 
#     accuracy                           0.99     85443
#    macro avg       0.95      0.58      0.64     85443
# weighted avg       0.99      0.99      0.99     85443
# =============================================================================

# =============================================================================

# Hyperparameter Tuning: Grid search cross validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#This line is applicable if you run with the earlier model else this line is not required
Test_X.drop(['Test_Prob','Test_Class'], axis =1, inplace = True) #Droping the earlier model's extra columns

M1_Model = LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']
solver = ['saga','sag']

# Create regularization hyperparameter space
C = np.logspace(3,-3,7)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty, solver = solver)

# Create grid search using 5-fold cross validation
Grid_M1_Model_Acc = GridSearchCV(M1_Model, hyperparameters, cv=3, verbose=5)

# Fit grid search
Best_Model = Grid_M1_Model_Acc.fit(Train_X, Train_Y) #Need to check the code
print(Best_Model.best_estimator_)
# =============================================================================
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='auto', n_jobs=None, penalty='l2',
#                    random_state=None, solver='sag', tol=0.0001, verbose=0,
#                    warm_start=False)
# =============================================================================

clf = LogisticRegression(C=1.0, penalty='l2', solver='sag');
clf.fit(Train_X, Train_Y)

#Test_X.drop(['Test_Prob','Test_Class'], axis =1, inplace = True)

Test_X['Test_Prob'] = clf.predict(Test_X)
Test_X.columns

Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)
Test_X.columns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Test_X['Test_Class'], Test_Y))
#0.937443675900893

print(confusion_matrix(Test_X['Test_Class'], Test_Y))
# =============================================================================
# [[80004    66]
#  [ 5279    94]]
# =============================================================================

print(classification_report(Test_X['Test_Class'], Test_Y))
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       0.94      1.00      0.97     80070
#            1       0.59      0.02      0.03      5373
# 
#     accuracy                           0.94     85443
#    macro avg       0.76      0.51      0.50     85443
# weighted avg       0.92      0.94      0.91     85443
# =============================================================================

#This line is applicable if you run with the earlier model else this line is not required
Test_X.drop(['Test_Prob','Test_Class'], axis =1, inplace = True) #Droping the earlier model's extra columns

# Random Forest without SMOTE
from sklearn.ensemble import RandomForestClassifier

M1_RF = RandomForestClassifier(random_state = 123)
M1_RF = M1_RF.fit(Train_X, Train_Y)

# Prediction and Validation
Test_X['Test_Prob'] = M1_RF.predict(Test_X)
Test_X.columns

# Classify 0 or 1 based on 0.5 cutoff
Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)
Test_X.columns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Test_X['Test_Class'], Test_Y))
#0.9995669627705019
print(confusion_matrix(Test_X['Test_Class'], Test_Y))
# =============================================================================
# [[85277    31]
#  [    6   129]]
# =============================================================================
print(classification_report(Test_X['Test_Class'], Test_Y))
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     85308
#            1       0.81      0.96      0.87       135
# 
#     accuracy                           1.00     85443
#    macro avg       0.90      0.98      0.94     85443
# weighted avg       1.00      1.00      1.00     85443
# =============================================================================

#This line is applicable if you run with the earlier model else this line is not required
Test_X.drop(['Test_Prob','Test_Class'], axis =1, inplace = True) #Droping the earlier model's extra columns

# SVM without SMOTE
from sklearn.svm import SVC
M1 = SVC()
M1_Model = M1.fit(Train_X, Train_Y) 

# Prediction and Validation
Test_X['Test_Prob'] = M1_Model.predict(Test_X)
Test_X.columns

# Classify 0 or 1 based on 0.5 cutoff
Test_X['Test_Class'] = np.where(Test_X['Test_Prob'] >= 0.5, 1, 0)
Test_X.columns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(Test_X['Test_Class'], Test_Y))
#0.9981274065751436
print(confusion_matrix(Test_X['Test_Class'], Test_Y))
# =============================================================================
# [[85283   160]
#  [    0     0]]
# =============================================================================
print(classification_report(Test_X['Test_Class'], Test_Y))
# =============================================================================
#               precision    recall  f1-score   support
# 
#            0       1.00      1.00      1.00     85443
#            1       0.00      0.00      0.00         0
# 
#     accuracy                           1.00     85443
#    macro avg       0.50      0.50      0.50     85443
# weighted avg       1.00      1.00      1.00     85443
# =============================================================================

# AUPRC on Random Forest (Best Model)
from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(Test_X['Test_Class'], Test_Y)
 
#Area Under the Precision-Recall Table
P_R_Table = pd.DataFrame()
P_R_Table["Recall"] = recall
P_R_Table["Precision"] = precision
 
# Plot Area Under the Precision-Recall Curve (AUPRC)
sns.lineplot(P_R_Table["Recall"], P_R_Table["Precision"])
 
# Area under the Precision-Recall Curve (AUPRC)
auc(recall, precision)
#0.8809378889044939

