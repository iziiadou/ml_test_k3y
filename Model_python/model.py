from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import xgboost as xgb
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
df=pd.read_csv('ai4i2020.csv')
df
df.head()
#getting the names of columns
print(df.columns)
print(df.shape)
print(df.skew())
print(df.corr())



#Global declartions of function names
global Head
global Size
global Column_names
global Describe
global Shape
global Count
global Value_count
global ISNULL
global Tail
global Ndim
global Nunique
global Memory_usage
global Duplicated
global ISNA
global DTYPES
global CORR
global Info
global operations

def Head(value=5):
            print('\033[1m'+'displaying the', value, 'rows'+'\033[0m')
            a=df.head(value)
            return a
            print("--------------------------------------------------------------------------")
Head()

def Tail():
    print('\033[1m'+"The last five rows of the dataframe are"+'\033[0m')
    co3=df.tail()
    return(co3)
    print("--------------------------------------------------------------------------")
Tail()

def Describe():
    print('\033[1m'+"The Description of our dataset is:"+'\033[0m')
    des=df.describe()
    return(des)
    print("--------------------------------------------------------------------------")
Describe()

def Size():
    print('\033[1m'+"The size of dataset is :"+'\033[0m')
    siz=df.size
    print(siz,'\n')
    print("--------------------------------------------------------------------------")
Size()

def Count():
    print('\033[1m'+"The count of non null values are:"+'\033[0m')
    co=df.count()
    print(co,'\n')
    print("--------------------------------------------------------------------------")
Count()

def Memory_usage():
    print('\033[1m'+"The total memory used is :"+'\033[0m')
    co6=df.memory_usage()
    print(co6,'\n')
    print("--------------------------------------------------------------------------")
Memory_usage()

def DTYPES():
    print('\033[1m'+"The datatypes are :"+'\033[0m')
    co9=df.dtypes
    print(co9,'\n')
    print("--------------------------------------------------------------------------")
DTYPES()

def Info():
    print('\033[1m'+"The info of data set is :"+'\033[0m')
    co11=df.info()
    print("--------------------------------------------------------------------------")
Info()

def operations(df,x):
    if df[x].dtype=="float64":
        print('\033[1m'+'', x, 'rows'+'\033[0m')
        print('\033[1m'+"It is a quantitaive data \n"+'\033[0m')
        print("The mean is :\n",df[x].mean())
        print("The median is :\n",df[x].median())
        print("The Standard Deviation is \n",df[x].std())
        q1=df[x].quantile(0.25)
        q2=df[x].quantile(0.5)
        q3=df[x].quantile(0.75)
        IQR=q3-q1
        LLP=q1-1.5*IQR
        ULP=q3+1.5*IQR
        print("The quartiles are q1 : \n",q1)
        print("The quartiles are q2 : \n",q2)
        print("The quartiles are q3 :\n ",q3)
        print("The Uppler limit point of the data is \n",ULP)
        print("The lower limit point of the data is \n ",LLP)
        if df[x].min()>LLP and df[x].max()<ULP:
            print("The outliers are not present \n")
            print("--------------------------------------------------------------------------")

        else:

            print("The outliers are present \n")
            print("The outliers are :")
            print(df[df[x].values>ULP][x])
            print(df[df[x].values<LLP][x])

            print("--------------------------------------------------------------------------")

    elif df[x].dtype=="int64":
        print('\033[1m'+'', x, 'rows'+'\033[0m')
        print('\033[1m'+"It is a quantitaive data \n"+'\033[0m')
        print("The mean is : \n",df[x].mean())
        print("The median is : \n",df[x].median())
        print("The Standard Deviation is \n",df[x].std())
        q1=df[x].quantile(0.25)
        q2=df[x].quantile(0.5)
        q3=df[x].quantile(0.75)
        IQR=q3-q1
        LLP=q1-1.5*IQR
        ULP=q3+1.5*IQR
        print("The quartiles are q1 : \n",q1)
        print("The quartiles are q2 : \n",q2)
        print("The quartiles are q3 : \n",q3)
        print("The Uppler limit point of the data is \n",ULP)
        print("The lower limit point of the data is \n",LLP)
        if df[x].min()>LLP and df[x].max()<ULP:
            print("The outliers are not present \n")

            print("--------------------------------------------------------------------------")

        else:

            print("The outliers are present \n")
            print("The outliers are :")
            print(df[df[x].values>ULP][x])
            print(df[df[x].values<LLP][x])
            print("--------------------------------------------------------------------------")
    else:

        print('\033[1m'+"The data is Qualitative \n"+'\033[0m')


        if df[x].nunique()==1:
            print('\033[1m'+"The data is singular \n"+'\033[0m')
            print("The mode is :",df[x].mode())
            print("The count of mode is \n",df[x].value_counts())
        elif df[x].nunique()==2:
            print('\033[1m'+"The data is Binary \n"+'\033[0m')
            print("The mode is :",df[x].mode())
            print("The count of mode is \n",df[x].value_counts())
        elif df[x].nunique()>2:
            print('\033[1m'+"The data is Multi \n"+'\033[0m')
            print("The mode is :",df[x].mode())
            print("The count of mode is \n",df[x].value_counts())

        print("--------------------------------------------------------------------------")

c=df.columns
for i in c:
    operations(df,i)
    print("\n")

def Summary():
        print('\033[1m'+"The Summary of data is  \n"+'\033[0m')
        print("The shape of the datset is :",df.shape)
        print("The sixe o the data set is :",df.size)
        print("The dimensions of the dataset are:",df.ndim)
        print("The memory usage of the data set are",df.memory_usage())
        print("The data types of the dataset are:",df.dtypes)
        print("--------------------------------------------------------------------------")

Summary()

def Column_Summary():
        print('\033[1m'+"The Column wise Summary of data is  \n"+'\033[0m')
        k=df.columns
        for i in k:
            print('\033[1m'+'', i, 'rows'+'\033[0m')
            print("The Shape of the column ",i,"is ",df[i].shape)
            print("The Size of the column ",i,"is ",df[i].size)
            print("The Dimensions of the column ",i,"is ",df[i].ndim)
            print("The Memory used by the column ",i,"is ",df[i].memory_usage())
            print("The Data types  of the column ",i,"is ",df[i].dtypes)
            print("--------------------------------------------------------------------------")
Column_Summary()

df['Machine failure'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
df['TWF'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
df['HDF'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
df['PWF'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
df['OSF'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.show()
df['RNF'].value_counts().plot(kind='pie', autopct='%1.1f%%')
df1=df[df['Machine failure']==1][['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].apply(pd.value_counts)
df2=df.drop(['TWF', 'HDF', 'PWF', 'OSF', 'RNF'],axis=1)
sns.pairplot(df2,hue='Machine failure')
plt.show()

df3=df2[df2['Machine failure']==1][['UDI']].apply(pd.value_counts)

percentage=(len(df3)/len(df))*100
df2.columns
df4=df2[df2['Machine failure']==1][['Type']].apply(pd.value_counts)
df5=df2[df2['Machine failure']==1][['Air temperature [K]']].apply(pd.value_counts)
df6=df2[df2['Machine failure']==1][['Process temperature [K]']].apply(pd.value_counts)
df7=df2[df2['Machine failure']==1][['Rotational speed [rpm]']].apply(pd.value_counts)
df8=df2[df2['Machine failure']==1][['Torque [Nm]']].apply(pd.value_counts)
df9=df2[df2['Machine failure']==1][['Tool wear [min]']].apply(pd.value_counts)



def LABEL_ENCODING(c1):
    from sklearn import preprocessing
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
 
    # Encode labels in column 'species'.
    df[c1]= label_encoder.fit_transform(df[c1])
 
    df[c1].unique()
    return df

LABEL_ENCODING('Product ID')

LABEL_ENCODING('Type')

df.rename(columns = {'Air temperature [K]':'Airtemp'}, inplace = True)

df.rename(columns={'Process temperature [K]':'Processtemp'} ,inplace=True)

df.rename(columns={'Rotational speed [rpm]':'Rotationalspeed'} ,inplace=True)

df.rename(columns={'Torque [Nm]':'Torque'} ,inplace=True)

df.rename(columns={'Tool wear [min]':'Toolwear'} ,inplace=True)


X=df.drop('Machine failure',axis=1)
y=df['Machine failure']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
D_train = xgb.DMatrix(data=X_train, label=Y_train)
D_test = xgb.DMatrix(data=X_test, label=Y_test)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)

import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))

model.dump_model('dump.raw.txt')
plot_tree(model)
plt.show()
