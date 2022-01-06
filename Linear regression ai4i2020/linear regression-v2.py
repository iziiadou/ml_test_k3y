"""
Linear Regression Ai4i2020
"""

import pandas as pd
import numpy as np
import pandas_profiling
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Importing the dataset
df = pd.read_csv('ai4i2020.csv')
#x = df.iloc[:, -1].values
#y = df.iloc[:, 1].values

df.info()
df.head()
profile = df.profile_report()
df.columns

x = df[['Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]

#x = df[['Process temperature [K]', 'Rotational speed [rpm]', 'Tool wear [min]']]

#replace with the following to also consider Tool Wear Failure (TWF) etc
#x = df[['Process temperature [K]', 'Rotational speed [rpm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

y = df[['Machine failure']]
#replace with y = df[['Air temperature [K]']] to predict the corresponding value
 
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)

#print ("x_train: ", x_train.head(10))
#print ("y_train: ", y_train.head(10))
#print ("x_test: ", x_test.head())
#print ("y_test: ", y_test.head())

linear = LinearRegression()
linear.fit(x_train, y_train)

##### THIS PART PRODUCES CORRELATION FIGURES #####
#Correlation of torque to rotational speed
sns.lmplot(x='Torque [Nm]',y='Rotational speed [rpm]',data=df,aspect=2,height=10,hue='Machine failure')

plt.xlabel('Torque [Nm]: as Independent variable')
plt.ylabel('Rotational speed [rpm]: as Dependent variable')
plt.title('Torque [Nm] Vs Rotational speed [rpm]');
plt.savefig("LR-torque-rotation.png")

#Correlation of torque to temp
sns.lmplot(x='Torque [Nm]',y='Process temperature [K]',data=df,aspect=2,height=10,hue='Machine failure')
plt.xlabel('Torque [Nm]: as Independent variable')
plt.ylabel('Process temperature [K]: as Dependent variable')
plt.title('Torque [Nm] Vs Process temperature [K]');
plt.savefig("LR-torque-temp.png")

# Visualising the Test set results
#plt.plot(x_test, linear.predict(x_test), color = 'blue')
#plt.xlabel('Failure')
#plt.ylabel('Prediction')
#plt.title('Test vs Prediction');
#plt.savefig("test-prediction.png")

# Visualising the Training set results
#plt.plot(x_train, linear.predict(x_train), color = 'blue')
#plt.xlabel('Train')
#plt.ylabel('Prediction')
#plt.title('Train vs Prediction (failure)');
#plt.savefig("train-prediction.png")
################################################



#sns.displot(df['Rotational speed [rpm]'])
#plt.savefig("rpm.png")

"""
Coefficient and intercept
"""
print (linear.intercept_)
print (linear.coef_)

print("Accuracy test Score on Train data :",linear.score(x_train, y_train))
print("Accuracy test Score on Test data :",linear.score(x_test, y_test))

filename = 'LR_model.sav'
pickle.dump(linear, open(filename, 'wb'))

# 'R, 'Torque [Nm]', 'Tool wear [min]']]

process_temp = float(input("Enter Process temperature [K]: (e.g., 311) "))
rotation_speed = float(input("Enter Rotational speed [rpm]: (e.g., 1530)"))
torque = float(input("Enter Torque [Nm]: (e.g., 30) "))
tool_wear = float(input("Enter Tool wear [min]: (e.g., 21)"))

k = linear.predict([[process_temp, rotation_speed, torque, tool_wear]])
if k<0:
	k = 0
elif k>1:
	k = 1

print("\nPossibility of failure (%):", k*100)
