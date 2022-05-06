from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'D:\Piyusha\Python Project\healthcare-dataset-stroke-data.csv')

data

data.shape

data.info()

data.isnull().sum()

data['bmi'].value_counts()

data['bmi'].describe()
data['bmi'].fillna(data['bmi'].mean(),inplace=True)
data['bmi'].describe()

data.isnull().sum()
data.drop('id',axis=1,inplace=True)
data


from matplotlib.pyplot import figure
figure(num=None,figsize=(8,6),dpi=800,facecolor='w',edgecolor='k')

data.plot(kind='box')
plt.show()

data.head()

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
gender=enc.fit_transform(data['gender'])
smoking_status=enc.fit_transform(data['smoking_status'])
work_type=enc.fit_transform(data['work_type'])
Residence_type=enc.fit_transform(data['Residence_type'])
ever_married=enc.fit_transform(data['ever_married'])
data['work_type']=work_type
data['ever_married']=ever_married
data['Residence_type']=Residence_type
data['smoking_status']=smoking_status
data['gender']=gender
data

data.info()
X=data.drop('stroke',axis=1)
X.head

Y=data['stroke']
Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
X_train

Y_train

X_test

Y_test

data.describe()

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
X_train_std

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train_std,Y_train)
DecisionTreeClassifier()
dt.feature_importances_

X_train.columns


Y_pred=dt.predict(X_test_std)
Y_pred

from sklearn.metrics import accuracy_score
ac_dt=accuracy_score(Y_test,Y_pred)
ac_dt



from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_std,Y_train)
LogisticRegression()
Y_pred_lr=lr.predict(X_test_std)
Y_pred_lr
array([0, 0, 0, ..., 0, 0, 0], dtype=int64)
ac_lr=accuracy_score(Y_test,Y_pred_lr)
ac_lr



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train_std,Y_train)
KNeighborsClassifier()
Y_pred=knn.predict(X_test_std)
ac_knn=accuracy_score(Y_test,Y_pred)
ac_knn

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train_std,Y_train)
RandomForestClassifier()
Y_pred=rf.predict(X_test_std)
ac_rf=accuracy_score(Y_test,Y_pred)
ac_rf

from sklearn.svm import SVC

sv=SVC()
sv.fit(X_train_std,Y_train)

ac_sv=accuracy_score(Y_test,Y_pred)

ac_sv

ac_lr

plt.bar(['Decision Tree','Logistic','KNN','Random Forest','SVM'],[ac_dt,ac_lr,ac_knn,ac_rf,ac_sv])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()