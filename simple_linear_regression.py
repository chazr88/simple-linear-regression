import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _update_other_results
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



dataset = pd.read_csv('Salary_Data.csv')
#X will be used to grab the first 3 columns in the dataset. They will be our features. 
#The first ":" specifies the rows, the second specifies the columns.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

###################
#Splitting the dataset into the Training and Test set
##################


#Splitting the data into 4 sets. x training and test, and y traning and test.
#We want to secify only 20% of this go to the test sets. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,  random_state = 0)

# print(x_train)
# print(x_test)


########################
#Training the Simple Linear Regression model on the Training set
#########################

#NOTE regression is when you have to predict a continous real value. 
#Here we are creating the regressor from the LinearRegression class. That class is going to help
#us build the Simple Linear Regression Model.
regressor = LinearRegression()
regressor.fit(x_train, y_train)


##############################
#Predicting the Test set results
#############################
#Based on the information of the traning data, we are going to predict the Salary based on the Years of Experience of the test set. 

#Here we create a var to hold the predicted salaries based of the years of experience of x_test
y_pred = regressor.predict(x_test)


#######################
#Visualising the Training set results
#########################
#2D plot with x axis being YOE and y being the salaries. 

#Adds the points of real salaries
plt.scatter(x_train, y_train, color = 'red')
#Adds the regression line of the predicted salaries
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


##############################
#Visualising the Test set results
##############################

#Adds the points of real salaries
plt.scatter(x_test, y_test, color = 'red')
#Adds the regression line of the predicted salaries
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
