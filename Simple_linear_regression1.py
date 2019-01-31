# Simple Linear Regression

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# importing datasets
dataset = pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

# splitting dataset into traing and testing 
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting 
y_pred=regressor.predict(X_test)



# Visualising the training results
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary VS Experience ( Training Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
# Visualising the test results
plt.scatter(X_test,y_test,color="black")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("Salary VS Experience ( Test Set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
regressor.score(X_train,y_train)

