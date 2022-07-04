import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dabl

# %%
data = pd.read_csv('StudentsPerformance.csv')
print(data.shape)
# %%
data.head()
# %%
data.describe()
# %%
data.select_dtypes('object').nunique()
# %%
no_of_columns = data.shape[0]
percentage_of_missing_data = data.isnull().sum() / no_of_columns
print(percentage_of_missing_data)
# %%
plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('fivethirtyeight')
dabl.plot(data, target_col='math score')
# %%
plt.rcParams['figure.figsize'] = (18, 6)
plt.style.use('fivethirtyeight')
dabl.plot(data, target_col='reading score')
# %%
data[['lunch', 'gender', 'math score', 'writing score',
      'reading score']].groupby(['lunch', 'gender']).agg('median')
# %%
data[['test preparation course',
      'gender',
      'math score',
      'writing score',
      'reading score']].groupby(['test preparation course', 'gender']).agg('median')
# %%
plt.rcParams['figure.figsize'] = (15, 5)
sns.countplot(data['gender'], palette='bone')
plt.title('Comparison of Males and Females', fontweight=30)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
# %%
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('ggplot')

sns.countplot(data['race/ethnicity'], palette='pink')
plt.title('Comparison of various groups', fontweight=30, fontsize=20)
plt.xlabel('Groups')
plt.ylabel('count')
plt.show()
# %%
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.countplot(data['parental level of education'], palette='Blues')
plt.title('Comparison of Parental Education', fontweight=30, fontsize=20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()
# %%
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('tableau-colorblind10')

sns.countplot(data['math score'], palette='BuPu')
plt.title('Comparison of math scores', fontweight=30, fontsize=20)
plt.xlabel('score')
plt.ylabel('count')
plt.xticks(rotation=90)
plt.show()
# %%
import warnings

warnings.filterwarnings('ignore')

data['total_score'] = data['math score'] + data['reading score'] + data['writing score']

sns.distplot(data['total_score'], color='magenta')

plt.title('comparison of total score of all the students', fontweight=30, fontsize=20)
plt.xlabel('total score scored by the students')
plt.ylabel('count')
plt.show()
# %%
from math import *
import warnings

warnings.filterwarnings('ignore')

data['percentage'] = data['total_score'] / 3

for i in range(0, 1000):
    data['percentage'][i] = ceil(data['percentage'][i])

plt.rcParams['figure.figsize'] = (15, 9)
sns.distplot(data['percentage'], color='orange')

plt.title('Comparison of percentage scored by all the students', fontweight=30, fontsize=20)
plt.xlabel('Percentage scored')
plt.ylabel('Count')
plt.show()


# %%
def getgrade(percentage, status):
    if status == 'Fail':
        return 'E'
    if percentage >= 90:
        return 'O'
    if percentage >= 80:
        return 'A'
    if percentage >= 70:
        return 'B'
    if percentage >= 60:
        return 'C'
    if percentage >= 40:
        return 'D'
    else:
        return 'E'


data['grades'] = data.apply(lambda x: getgrade(x['percentage'], x['status']), axis=1)

data['grades'].value_counts()
# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['test preparation course'] = le.fit_transform(data['test preparation course'])

data['lunch'] = le.fit_transform(data['lunch'])

data['race/ethnicity'] = data['race/ethnicity'].replace('group A', 1)
data['race/ethnicity'] = data['race/ethnicity'].replace('group B', 2)
data['race/ethnicity'] = data['race/ethnicity'].replace('group C', 3)
data['race/ethnicity'] = data['race/ethnicity'].replace('group D', 4)
data['race/ethnicity'] = data['race/ethnicity'].replace('group E', 5)

data['parental level of education'] = le.fit_transform(data['parental level of education'])

data['gender'] = le.fit_transform(data['gender'])

data['pass_math'] = le.fit_transform(data['pass_math'])

data['pass_reading'] = le.fit_transform(data['pass_reading'])

data['pass_writing'] = le.fit_transform(data['pass_writing'])

data['status'] = le.fit_transform(data['status'])

# %%
x = data.iloc[:, :14]
y = data.iloc[:, 14]

print(x.shape)
print(y.shape)
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# %%
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)
#%%
from sklearn.decomposition import PCA

pca = PCA(n_components = None)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

pca = PCA(n_components = 2)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
#%%
from sklearn.linear_model import  LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
#%%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot = True, cmap = 'Greens')
plt.title('Confusion Matrix for Logistic Regression', fontweight = 30, fontsize = 20)
plt.show()
#%%
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))
#%%
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot = True, cmap = 'Reds')
plt.title('Confusion Matrix for Random Forest', fontweight = 30, fontsize = 20)
plt.show()
