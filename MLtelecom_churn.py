import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

column_header=["State","Account Length","Area Code","Phone","International Plan","VMail Plan","VMail Message","Day Mins","Day Calls","Day Charge","Eve Mins","Eve Calls","Eve Charge","Night Mins","Night Calls","Night Charge","International Mins","International calls","International Charge","CustServ Calls","Churn"]
data=pd.read_csv("telecom_churn_data.txt", delimiter=",",header=None,names=column_header,index_col=False)
data.head()
data.info()
data.describe()
data.shape

data.hist(bins=50, figsize=(20,15))

plt.figure(figsize = (15,20))
plt.subplot(5,3,1)
sns.barplot(data['Churn'], data['CustServ Calls'])
plt.subplot(5,3,2)
sns.barplot(data['Churn'], data['International Charge'])
plt.subplot(5,3,3)
sns.barplot(data['Churn'], data['International calls'])
plt.subplot(5,3,4)
sns.barplot(data['Churn'], data['International Mins'])
plt.subplot(5,3,5)
sns.barplot(data['Churn'], data['Night Charge'])
plt.subplot(5,3,6)
sns.barplot(data['Churn'], data['Night Calls'])
plt.subplot(5,3,7)
sns.barplot(data['Churn'], data['Night Mins'])
plt.subplot(5,3,8)
sns.barplot(data['Churn'], data['Eve Charge'])
plt.subplot(5,3,9)
sns.barplot(data['Churn'], data['Eve Calls'])
plt.subplot(5,3,10)
sns.barplot(data['Churn'], data['Eve Mins'])
plt.subplot(5,3,11)
sns.barplot(data['Churn'], data['Day Charge'])
plt.subplot(5,3,12)
sns.barplot(data['Churn'], data['Day Calls'])
plt.subplot(5,3,13)
sns.barplot(data['Churn'], data['Day Mins'])
plt.subplot(5,3,14)
sns.barplot(data['Churn'], data['VMail Message'])
plt.suptitle("Variables relation to dependent variable - churn", fontsize = 20)
plt.show()

data_1=data.drop(["State","Area Code","Phone"],axis=1)
X = data_1.drop(['Churn'],axis=1).values
y = data_1['Churn'].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def accuracy_precision(cm):
    tp, fn, fp, tn = cm.ravel()
    accuracy=(tp+tn)/cm.sum()
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    print("The accuracy of the model is %s:" %accuracy)
    print("The precision of the model is %s:" %precision)
    print("The recall of the model is %s:" %recall)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=42)
classifier.fit(X_train,y_train)

y_train_predict=classifier.predict(X_train)
cm_train=confusion_matrix(y_train,y_train_predict)
accuracy_precision(cm_train)

y_test_predict=classifier.predict(X_test)
cm_test=confusion_matrix(y_test,y_test_predict)
accuracy_precision(cm_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(kernel='rbf',C=100,random_state=42)
#parameters=[{'C':[1,10,100,1000],'kernel':['linear','rbf']}]
#grid_search=GridSearchCV(estimator=classifier, param_grid=parameters,scoring='roc_auc',cv=10, n_jobs=-1)
classifier.fit(X_train_pca,y_train)

y_train_predict=classifier.predict(X_train_pca)
cm_train=confusion_matrix(y_train,y_train_predict)
accuracy_precision(cm_train)

y_test_predict=classifier.predict(X_test_pca)
cm_test=confusion_matrix(y_test,y_test_predict)
accuracy_precision(cm_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
classifier=DecisionTreeClassifier(random_state=42,splitter='best')
parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini']},{'min_samples_split':[2,3,4,5],'criterion':['entropy']}]
grid_search_dt=GridSearchCV(estimator=classifier, param_grid=parameters, scoring='roc_auc',cv=10, n_jobs=-1)
grid_search_dt=grid_search_dt.fit(X_train,y_train)
best_score=grid_search_dt.best_score_
best_parameters=grid_search_dt.best_params_

y_train_predict=grid_search_dt.predict(X_train)
cm_train=confusion_matrix(y_train,y_train_predict)
accuracy_precision(cm_train)

y_test_predict=grid_search_dt.predict(X_test)
cm_test=confusion_matrix(y_test,y_test_predict)
accuracy_precision(cm_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
classifier=RandomForestClassifier(random_state=42,n_estimators=23)
parameters=[{'min_samples_split':[2,3],'criterion':['gini','entropy'],'min_samples_leaf':[1,2]}]
grid_search_rf=GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy',cv=10, n_jobs=-1)
grid_search_rf.fit(X_train,y_train)
best_score=grid_search_rf.best_score_
best_parameters=grid_search_rf.best_params_

y_train_predict=grid_search_rf.predict(X_train)
cm_train=confusion_matrix(y_train,y_train_predict)
accuracy_precision(cm_train)

y_test_predict=grid_search_rf.predict(X_test)
cm_test=confusion_matrix(y_test,y_test_predict)
accuracy_precision(cm_test)
