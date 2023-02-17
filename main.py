import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import imblearn
#data = pd.read_csv('/Users/hamid/Downloads')
data = pd.read_csv('/Users/hamid/Downloads/creditcard.csv')

data.head(5)
data.info()
data.isnull().sum().all() # to check if there exists any Nan"
"to get the idea of how imbalance the data is"
fraud_data = data.Class.value_counts()[0]/sum(data.Class.value_counts())
non_fraud_data = data.Class.value_counts()[1]/sum(data.Class.value_counts())
print(f'Fraud data makes {round(fraud_data*100,2)}%  of all data and non fraude data makes {round(non_fraud_data*100,2)}% of all data')
data.columns
#sns.violinplot(axes = axes[0],x = data.V1)
#sns.pairplot(data.iloc[:,:5])
#sns.distplot(x=data.Time)
data.Time.describe()
data.skew()

fig, ax = plt.subplots()
sns.violinplot(data.Amount, ax=ax)
ax.set_xlim(min(data.Amount),10000)

from sklearn.preprocessing import RobustScaler
robust = RobustScaler()
data.Amount = robust.fit_transform(data.Amount.values.reshape(-1,1))
data.Time = robust.fit_transform(data.Time.values.reshape(-1,1))
data[data.Amount >50].count()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
'i used stratified shuffle to preserve the class balance in train and test sets'
y = data.Class
x = data.drop('Class',axis=1)
sss = StratifiedShuffleSplit(5)
for train,test in sss.split(x,y):
    print('train :',train , "test:",test)
    x_train,y_train = x.iloc[train],y.iloc[train]
    x_test,y_test = x.iloc[test],y.iloc[test]
y_test.shape
x_test.shape
x_train.shape[0]/(x_train.shape[0]+x_test.shape[0])

y_train.value_counts()
data = data.sample(frac=1)
fraud = data.loc[data['Class']==1]
non_fraud = data.loc[data['Class']==0][:492]
non_fraud.shape
fraud.shape
under_sample_data = pd.concat([fraud,non_fraud])
under_sample_data = under_sample_data.sample(frac=1)
under_sample_data.head(10)
sns.heatmap(under_sample_data.corr(),cmap='coolwarm_r',annot_kws={'size':20})
#sns.heatmap(data.corr(),cmap='coolwarm_r',annot_kws={'size':20})
under_sample_data.corr().loc[:,'Class'].sort_values()

f, axes = plt.subplots(ncols=4,nrows=2, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=under_sample_data, ax=axes[0][0])
axes[0][0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=under_sample_data, ax=axes[0][1])
axes[0][1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=under_sample_data, ax=axes[0][2])
axes[0][2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=under_sample_data, ax=axes[0][3])
axes[0][3].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V11", data=under_sample_data,  ax=axes[1][0])
axes[1][0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=under_sample_data, ax=axes[1][1])
axes[1][1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=under_sample_data, ax=axes[1][2])
axes[1][2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=under_sample_data, ax=axes[1][3])
axes[1][3].set_title('V19 vs Class Positive Correlation')
plt.show()

#we are going to remove the outliers in V12 , V2 ,V10 , V14 using interquartile range method"
# #sns.distplot(under_sample_data['V14'].loc[under_sample_data["Class"]==1].values,fit=norm)

threshhold =(np.percentile(under_sample_data[under_sample_data['Class']==1]['V14'].values,75) -
np.percentile(under_sample_data[under_sample_data['Class']==1]['V14'].values,25)) *1.5

threshhold_high = np.percentile(under_sample_data[under_sample_data['Class']==1]['V14'].values,75) + threshhold
threshhold_low = np.percentile(under_sample_data[under_sample_data['Class']==1]['V14'].values,25) - threshhold

#time to find the outliers"
a = (under_sample_data[under_sample_data['Class']==1]['V14'] >threshhold_high) | (under_sample_data[under_sample_data['Class']==1]['V14'] < threshhold_low)
under_sample_data[under_sample_data['Class']==1]['V14'][a].index
under_sample_data = under_sample_data.drop(under_sample_data[under_sample_data['Class']==1]['V14'][a].index)


threshhold =(np.percentile(under_sample_data[under_sample_data['Class']==0]['V14'].values,75) -
np.percentile(under_sample_data[under_sample_data['Class']==0]['V14'].values,25)) *1.5

threshhold_high = np.percentile(under_sample_data[under_sample_data['Class']==0]['V14'].values,75) + threshhold
threshhold_low = np.percentile(under_sample_data[under_sample_data['Class']==0]['V14'].values,25) - threshhold
a = (under_sample_data[under_sample_data['Class']==0]['V14'] >threshhold_high) | (under_sample_data[under_sample_data['Class']==0]['V14'] < threshhold_low)
under_sample_data = under_sample_data.drop(under_sample_data[under_sample_data['Class']==0]['V14'][a].index)

#sns.boxplot(x="Class", y="V2", data=under_sample_data)

"----------------------------------------------------------------------"
threshhold =(np.percentile(under_sample_data[under_sample_data['Class']==1]['V2'].values,75) -
np.percentile(under_sample_data[under_sample_data['Class']==1]['V2'].values,25)) *1.5

threshhold_high = np.percentile(under_sample_data[under_sample_data['Class']==1]['V2'].values,75) + threshhold
threshhold_low = np.percentile(under_sample_data[under_sample_data['Class']==1]['V2'].values,25) - threshhold
a = (under_sample_data[under_sample_data['Class']==1]['V2'] >threshhold_high) | (under_sample_data[under_sample_data['Class']==1]['V2'] < threshhold_low)
under_sample_data[under_sample_data['Class']==1]['V2'][a].index
under_sample_data = under_sample_data.drop(under_sample_data[under_sample_data['Class']==1]['V2'][a].index)

'----------------------------------------------------------'

threshhold =(np.percentile(under_sample_data[under_sample_data['Class']==0]['V2'].values,75) -
np.percentile(under_sample_data[under_sample_data['Class']==0]['V2'].values,25)) *1.5

threshhold_high = np.percentile(under_sample_data[under_sample_data['Class']==0]['V2'].values,75) + threshhold
threshhold_low = np.percentile(under_sample_data[under_sample_data['Class']==0]['V2'].values,25) - threshhold
a = (under_sample_data[under_sample_data['Class']==0]['V2'] >threshhold_high) | (under_sample_data[under_sample_data['Class']==0]['V2'] < threshhold_low)
under_sample_data[under_sample_data['Class']==0]['V2'][a].index
under_sample_data = under_sample_data.drop(under_sample_data[under_sample_data['Class']==0]['V2'][a].index)

'--------------------------------------------------------------------'

threshhold =(np.percentile(under_sample_data[under_sample_data['Class']==1]['V10'].values,75) -
np.percentile(under_sample_data[under_sample_data['Class']==1]['V10'].values,25)) *1.5

threshhold_high = np.percentile(under_sample_data[under_sample_data['Class']==1]['V10'].values,75) + threshhold
threshhold_low = np.percentile(under_sample_data[under_sample_data['Class']==1]['V10'].values,25) - threshhold
a = (under_sample_data[under_sample_data['Class']==1]['V10'] >threshhold_high) | (under_sample_data[under_sample_data['Class']==1]['V10'] < threshhold_low)
under_sample_data[under_sample_data['Class']==1]['V10'][a].index
under_sample_data = under_sample_data.drop(under_sample_data[under_sample_data['Class']==1]['V10'][a].index)

sns.boxplot(x="Class", y="V10", data=under_sample_data)

'-----------------------------------------------------------------------'

y_under= under_sample_data['Class']
x_under = under_sample_data.drop('Class',axis=1)

cov_mat = np.cov(x_under.T)
eig_vlas,eig_vecs = np.linalg.eig(cov_mat)
print("vectors \n" , eig_vecs)
print('values \n',eig_vlas)
eig_pairs = [(np.abs(eig_vlas[i]),eig_vecs[:,i]) for i in range(len(eig_vlas))]
eig_pairs.sort(key=lambda x:x[0],reverse=True)
print('eigenvalues in descending order :')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vlas)
var_exp = [(i/tot)*100 for i in sorted(eig_vlas,reverse=True)]
print(sum(var_exp[:5]))
"by reduction from 20 features to 5 features, we preserve 86% of the data"

'---------------------'
'PCA with scikit learn'
from sklearn.decomposition import PCA
x_pca = PCA(5,random_state=42).fit_transform(x_under.values)

"time for classification the undersampled method"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_pca, y_under, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


training_score = cross_val_score(LogisticRegression(penalty='l2',C=.100),X_train,y_train,cv=5)
training_score.mean()*100

training_score = cross_val_score(SVC(),X_train,y_train,cv=5)
training_score.mean()*100

training_score = cross_val_score(KNeighborsClassifier(),X_train,y_train)
training_score.mean()*100

log_reg_params = {"penalty": ['l2','l1'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(solver='liblinear',max_iter=10000), log_reg_params)
#print( grid_log_reg.best_estimator_)
grid_log_reg.fit(X_train, y_train)
print(grid_log_reg.best_estimator_)
print(grid_log_reg.best_score_)
from sklearn.metrics import accuracy_score
classifier = LogisticRegression(C=10)
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
print(f"Accuracy of the classifier is: {accuracy_score(y_test, pred)}")
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))
print(f"Accuracy of the classifier is: {accuracy_score(y_train, classifier.predict(X_train))}")
sns.heatmap((confusion_matrix(y_test, pred)),annot=True)
from sklearn.metrics import recall_score
print(f"Recall Score of the classifier is: {recall_score(y_test, pred)}")
from sklearn.metrics import f1_score
print(f"F1 Score of the classifier is: {f1_score(y_test, pred)}")


'--------------------------------------------------------------------'
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import KFold
y=data.Class.values
x = data.drop('Class',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.20)
X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)
classifier = RandomForestClassifier(max_features=3, max_depth=2 ,n_estimators=10, random_state=3, criterion='entropy', n_jobs=1, verbose=1 )
results = cross_val_score(classifier,X_smote, y_smote, scoring='f1')
print(results,'*'*10,np.mean(results))
classifier.fit(X_smote,y_smote)
smote_prediction = classifier.predict(X_test)
print(confusion_matrix(y_test, smote_prediction))
print(f"F1 Score of the classifier is: {f1_score(y_test, smote_prediction)}")
print(f"Recall Score of the classifier is: {recall_score(y_test, smote_prediction)}")


