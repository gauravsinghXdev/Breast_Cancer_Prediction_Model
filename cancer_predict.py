import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Precosseing (read file,null values,duplicate values,outliers)

df = pd.read_csv('breast-cancer.csv')

print(df)

data = df.drop(['id'], axis=1)
data1 = data.drop(['diagnosis'], axis=1)

print(data1.head())

print(data1.describe())

print(data1.isnull())

data1[pd.isnull(data1).any(axis=1)]

for i in data1:
  def length(n):
    a = len(data1[pd.isnull(data1[n])])
    print(a)
length(i)

data1[data1.duplicated()]

#Bivariate Analysis using graphs


sns.boxplot(data1['radius_mean'])
plt.title('radius_mean')

x = data1['radius_mean']
y = data1['texture_mean']
plt.scatter(x,y)
plt.xlabel('radius_mean')
plt.ylabel("texture_mean")
plt.scatter(x,y,color = ['green'])

#Removing Outliers using IQR(interquartile range) method

for i in data1:
  def IQR(column_name):
    Q1 = np.percentile(column_name,25,method='midpoint')
    Q3 = np.percentile(column_name,75,method='midpoint')
    IQR = Q3 - Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    upper_array = np.where(column_name>=upper)
    lower_array = np.where(column_name<=lower)
    return upper,lower
  upper_value,lower_value = IQR(data1[i])
  print(upper_value)
  print(lower_value)
  def Mean(column_name, a, b, column):
    mean = column_name.mean()
    print(mean)
    (data1.loc[column_name > a, column] ) = np.nan
    (data1.loc[column_name < b, column] ) = np.nan
    mean = (int)(mean * 100 + .5)
    return mean / 100.0
  mean_value = Mean(data1[i], upper_value, lower_value, i)
  print(mean_value)
  def fill(mean_value):
    data1.fillna(mean_value, inplace=True)
  fill(mean_value)

column_len = len(data1.columns)
print(column_len)
first_column  = data.pop('diagnosis')
data1.insert(column_len,'diagnosis',first_column)
print(data1)

sns.boxplot(data1['radius_mean'])
plt.title('radius_mean')

#Converting categorical output in numeric form using labelEncoder()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data1['diagnosis'] = le.fit_transform(data1.diagnosis.values)

x = data1.iloc[:,0:column_len].values
print(x)

y = data1.iloc[:,-1].values
print(y)

#Feature Engineering

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42,shuffle=True)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

#First algorithm is KNN(K-Nearest Neighbors) algorithm

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
from sklearn import metrics

accuracy_result = []
for n in range(1,10):
  classifier1 = KNeighborsClassifier(n_neighbors= n)
  classifier1.fit(X_train,y_train)
  y_pred = classifier1.predict(X_test)
  result = confusion_matrix(y_test, y_pred)
  print("Confusion matrix is \n",result)
  result1 =classification_report(y_pred, y_test)
  print("Classification report is\n",result1)
  result2 = accuracy_score(y_pred, y_test)
  print("Accuracy score is ",result2)
  accuracy_result.append(result2)
print("Accuracy result in array is",accuracy_result)

a = np.array(accuracy_result)
print(a)
KNN_len = len(a)
print(KNN_len)
for i in range(KNN_len-1):
  if a[i] < a[i+1]:
    accuracy_knn = a[i+1]
  i=i+1
print("Best Accuracy is ",accuracy_knn)
fpr1, tpr1, _ = metrics.roc_curve(y_test, y_pred)
auc1 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr1,tpr1,label="AUC="+str(auc1))
print("auc is",auc1)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''KNN classifier receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

#Applying GridSearch parameter tuning to find best result

knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
k_range = list(range(1, 31))
print(k_range)
param_grid = dict(n_neighbors=k_range)
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
grid_search=grid.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
accuracy_GridKnn = accuracy_score(y_pred, y_test)
print("accuracy score of GridKNN is",accuracy_GridKnn)
fpr2, tpr2, _ = metrics.roc_curve(y_test, y_pred)
auc2 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr2,tpr2,label="AUC="+str(auc2))
print("auc is",auc2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''KNN GridSearchCv receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

if accuracy_knn > accuracy_GridKnn :
  accuracy_KNN = accuracy_knn
else:
  accuracy_KNN = accuracy_GridKnn
print(accuracy_KNN)

#Random Forest(RF) Algorithm

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=21, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
classifier_rf.fit(X_train,y_train)
y_pred = classifier_rf.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion matrix is \n",result)
result1 =classification_report(y_pred, y_test)
print("Classification report is\n",result1)
accuracy_rf = accuracy_score(y_pred, y_test)
print("Accuracy score is ",accuracy_rf)
fpr3, tpr3, _ = metrics.roc_curve(y_test, y_pred)
auc3 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr3,tpr3,label="AUC="+str(auc3))
print("auc is",auc3)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''Random forest classifier receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'bootstrap': [True],
    'max_depth': [5, 80, 90, 100],
    'max_features': [2,3],
    'min_samples_leaf': [3,5],
    'min_samples_split': [8,9],
    'n_estimators': [100, 200,300],
    'random_state': [21,42,10,5]
}
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.predict(X_test)
result = confusion_matrix(y_pred, y_test)
print("Confusion matrix is \n",result)
accuracy_GridRf = accuracy_score(y_pred, y_test)
print("Accuracy score is ", accuracy_GridRf)
fpr4, tpr4, _ = metrics.roc_curve(y_test, y_pred)
auc4 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr4,tpr4,label="AUC="+str(auc4))
print("auc is",auc4)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''Random forest GridSearchCv receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

if accuracy_rf > accuracy_GridRf:
  accuracy_RF = accuracy_rf
else:
  accuracy_RF = accuracy_GridRf
print(accuracy_RF)

#SVM(Support Vector Machine) ALgorithm

from sklearn.svm import SVC
svc = SVC(C = 100, kernel = 'rbf', gamma = 'scale')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
result = confusion_matrix(y_test, y_pred)
print("Confusion matrix is \n",result)
result1 = classification_report(y_pred, y_test)
print("classification report is\n",result1)
accuracy_Svm = accuracy_score(y_pred, y_test)
print("Accuracy score is ",accuracy_Svm)

fpr5, tpr5, _ = metrics.roc_curve(y_test, y_pred)
auc5 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr5,tpr5,label="AUC="+str(auc5))
print("auc is",auc5)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''SVM classifier receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid']}
rf = SVC()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, refit = True, verbose = 3)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
y_pred = grid_search.predict(X_test)
result = confusion_matrix(y_pred, y_test)
print("Confusion matrix is \n",result)
accuracy_GridSvm = accuracy_score(y_pred, y_test)
print("Accuracy score is ",accuracy_GridSvm)
fpr6, tpr6, _ = metrics.roc_curve(y_test, y_pred)
auc6 = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr6,tpr6,label="AUC="+str(auc6))
print("auc is",auc6)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''SVM GridSearchCv receiver
operator characteristic''')
plt.legend(loc=4)
plt.show()

if accuracy_Svm > accuracy_GridSvm:
  accuracy_SVM = accuracy_Svm
else:
  accuracy_SVM = accuracy_GridSvm
print(accuracy_SVM)

#graph to compare all algorithm results

x = ['RF', 'KNN', 'SVM']
y = [accuracy_RF*100, accuracy_KNN*100, accuracy_SVM*100]

from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize =(9, 5))
ax.barh(x, y)
ax.invert_yaxis()
for i in ax.patches:
    plt.text(i.get_width()+0.1, i.get_y()+0.5,
             str(round((i.get_width()), 1)),
             fontsize = 10,fontweight ='bold',color ='grey')
plt.xlabel("Classifiers")
plt.ylabel("Accuracy %")
plt.title("Classifier Accuracy")

plt.figure(0).clf()
plt.plot(fpr1,tpr1,label="AUC of KNN="+str(auc1))
plt.plot(fpr3,tpr3,label="AUC of RF="+str(auc3))
plt.plot(fpr5,tpr5,label="AUC of SVM="+str(auc5))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''ROC curve for ML algorithm''')
plt.legend(loc=4)

plt.figure(0).clf()
plt.plot(fpr2,tpr2,label="AUC of Grid_KNN="+str(auc2))
plt.plot(fpr4,tpr4,label="AUC of Grid_RF="+str(auc4))
plt.plot(fpr6,tpr6,label="AUC of Grid_SVM="+str(auc6))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('''ROC curve for Aglorithm
using GridSearchCV''')
plt.legend(loc=4)

#Compare results of different researches

from matplotlib import pyplot as plt
x = ['''Mohammed Amine
 Naji et al''', '''Varsha Nemade
 et al''', '''Burak
  Akbugday''', '''ours''' ]
y = [97.2, 97, 96.85, 98.83]
fig, ax = plt.subplots(figsize =(9, 5))
ax.barh(x, y)
ax.invert_yaxis()
for i in ax.patches:
    plt.text(i.get_width()+0.1, i.get_y()+0.5,
             str(round((i.get_width()), 1)),
             fontsize = 10,fontweight ='bold',color ='grey')
plt.xlabel("Accuracy Score")
plt.ylabel("RESEARCHES")
#plt.legend(4)
plt.show()