# setup and importing
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import seaborn as sns

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],"k--") #"k--" -> dashed line 
    plt.axis([0,1,0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

#Initilizing seed and random values
seed_value = 44
os.environ['PYTHONHASHSEED']=str(seed_value)

# Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

data_file ="ai4i2020.csv" 
data = pd.read_csv(data_file)
train_data,test_data = train_test_split(data, test_size = 0.33, random_state = seed_value)
train_data.to_csv('train.csv',index=0)
test_data.to_csv('test.csv',index=0)

# Binary classification
X=train_data.iloc[:,3:8]
Y=train_data.iloc[:,8]
X_test=test_data.iloc[:,3:8]
Y_test=test_data.iloc[:,8]

print(data.head())

## Random Forest Classifier
clf = RandomForestClassifier(max_depth=4, random_state=seed_value)
clf = clf.fit(X, Y)
rfc_pred=clf.predict(X_test)
print("Random Forest model confusion matrix:")
print(confusion_matrix(Y_test,rfc_pred))
print()
print("Random Forest model accuracy(in %):", metrics.accuracy_score(Y_test, rfc_pred)*100)
classification_report(Y_test,rfc_pred)
print("~~~~~~~~~~~~~~~")

## SDG Classifier
sgd = make_pipeline(SGDClassifier(max_iter=1000, tol=1e-3),)
sgd.fit(X, Y)
sdg_pred=sgd.predict(X_test)
print("SDG model confusion matrix:")
print(confusion_matrix(Y_test,sdg_pred))
print()
print("SGD model accuracy(in %):", metrics.accuracy_score(Y_test, sdg_pred)*100)
classification_report(Y_test,sdg_pred)
print("~~~~~~~~~~~~~~~")

## Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X, Y) 
gnb_pred = gnb.predict(X_test)
print("GNB model confusion matrix:")
print(confusion_matrix(Y_test,gnb_pred))
print()
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(Y_test, gnb_pred)*100)
classification_report(Y_test,gnb_pred)
print("~~~~~~~~~~~~~~~")

## K-Nearest Neighbors Classifier
k=KNeighborsClassifier()
k.fit(X,Y)
knn_pred=k.predict(X_test)
print("KNN model confusion matrix:")
print(confusion_matrix(Y_test,knn_pred))
print()
print("KNN model accuracy(in %):", metrics.accuracy_score(Y_test,knn_pred)*100)
classification_report(Y_test,knn_pred)
print("~~~~~~~~~~~~~~~")

## Logistic Regression Classifier
lr=LogisticRegression()
lr.fit(X,Y)
lr_pred=lr.predict(X_test)
print("LR model confusion matrix:")
print(confusion_matrix(Y_test,lr_pred))
print()
print("LR model accuracy(in %):", metrics.accuracy_score(Y_test,lr_pred)*100)
classification_report(Y_test,lr_pred)
print("~~~~~~~~~~~~~~~")

## Decision Tree Classifer
d=DecisionTreeClassifier()
d.fit(X,Y)
d_pred=d.predict(X_test)
print("DT model confusion matrix:")
print(confusion_matrix(Y_test,d_pred))
print()
print("DT model accuracy(in %):", metrics.accuracy_score(Y_test,d_pred)*100)
classification_report(Y_test,d_pred)
print("~~~~~~~~~~~~~~~")

## Gradient Boost Classifier
gbc=GradientBoostingClassifier()
gbc.fit(X,Y)
gbc_pred=gbc.predict(X_test)
print("GB model confusion matrix:")
print(confusion_matrix(Y_test,gbc_pred))
print()
print("GB model accuracy(in %):", metrics.accuracy_score(Y_test,gbc_pred)*100)
classification_report(Y_test,gbc_pred)
print("~~~~~~~~~~~~~~~")

## Support Vector Machine Classifier
svc=SVC()
svc.fit(X,Y)
svc_pred=svc.predict(X_test)
print("SV model confusion matrix:")
print(confusion_matrix(Y_test,svc_pred))
print()
print("SV model accuracy(in %):", metrics.accuracy_score(Y_test,svc_pred)*100)
classification_report(Y_test,svc_pred)
print("~~~~~~~~~~~~~~~")

## Print ROC Curve Figures
plt.title('RFC ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,rfc_pred)
plot_roc_curve(fpr, tpr)

plt.title('SDG ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,sdg_pred)
plot_roc_curve(fpr, tpr)

plt.title('GNB ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,gnb_pred)
plot_roc_curve(fpr, tpr)

plt.title('KNN ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,knn_pred)
plot_roc_curve(fpr, tpr)

plt.title('LR ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,lr_pred)
plot_roc_curve(fpr, tpr)

plt.title('DT ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,d_pred)
plot_roc_curve(fpr, tpr)

plt.title('GB ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,gbc_pred)
plot_roc_curve(fpr, tpr)

plt.title('SV ROC Curve')
fpr, tpr, thresholds = metrics.roc_curve(Y_test,svc_pred)
plot_roc_curve(fpr, tpr)


## Print Confusion Matrixes 
plt.title('RFC Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,rfc_pred),cmap='viridis',annot=True);
plt.show()

plt.title('SDG Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,sdg_pred),cmap='viridis',annot=True)
plt.show()

plt.title('GNB Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,gnb_pred),cmap='viridis',annot=True)
plt.show()

plt.title('KNN Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,knn_pred),cmap='viridis',annot=True)
plt.show()

plt.title('LR Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,lr_pred),cmap='viridis',annot=True)
plt.show()

plt.title('DT Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,d_pred),cmap='viridis',annot=True)
plt.show()

plt.title('GB Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,gbc_pred),cmap='viridis',annot=True)
plt.show()

plt.title('SV Confusion Matrix')
sns.heatmap(confusion_matrix(Y_test,svc_pred),cmap='viridis',annot=True)
plt.show()
