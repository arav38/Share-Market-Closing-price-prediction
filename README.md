# Share-Market-Closing-price-prediction
Share Market Analysis

#Install the dependencies
#import quandl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
pd.set_option("display.max_column" ,None)
pd.set_option("display.max_column" ,None)

# Get the stock data
df = pd.read_csv(r"C:\Users\akshay\Downloads\15-11-2018-TO-14-11-2019YESBANKALLN.csv")

# Take a look at the data
print(df.head())

#I only need the Adjusted Close (Adj. Close) price, so I am getting data only from the column ‘Adj. Close’ and storing this back into the #variable ‘df’.
# Get the Adjusted Close Price 
df = df[['Close Price']] 
# Take a look at the new data 
print(df.head())

#Now , I’m creating a variable called forecast_out to store the number of days into the future that I want to predict. if I decide I only #want to look 20 days into the future, I can simply change this variable from 30 to 20, and the program will predict now 20 days into the #future. I also need a column (the target or dependent variable) that will hold the predicted price values 30 days into the future.The #future price that I want that’s 30 days into the future is just 30 rows down from the current Adj.close price
#I will create a new column called ‘Prediction’ and populate it with data from the Adj. Close column but shifted 30 rows up to get the #price of the next 30 days, and then print the last 5 rows of the new data set.
#Since I shifted the data up 30 rows, the last 30 rows of data for the new column ‘Prediction’ will be empty or contain the value ‘NaN’ 

# A variable for predicting 'n' days out into the future
forecast_out = 2 #'n=30' days
#Create another column (the target ) shifted 'n' units up
df['Prediction'] = df[['Close Price']].shift(-forecast_out)
#print the new data set
print(df.tail())



#create the independent data set (X)This is the data set that i will use to train the machine learning model(s). To do this I will #create a variable called ‘X’ , and convert the data into a numpy (np) array after dropping the ‘Prediction’ column, then store this new #data into ‘X’.
#Then I will remove the last 30 rows of data from ‘X’, and store the new data back into ‘X’. Last but not least I print the data.

### Create the independent data set (X)  #######
# Cont the dataframe to a numpy arverray
X = np.array(df.drop(['Prediction'],1))

#Remove the last '30' rows
X = X[:-forecast_out]
print(X)


#now I will create the dependent data set called ‘y’. This is the target data, the one that holds the future price predictions. create #this new data set ‘y’, I will convert the data frame into a numpy array and from the ‘Prediction’ column, store it into a new variable #called ‘y’ and then remove the last 30 rows of data from ‘y’. Then I will print ‘y’ to make sure their are no NaN’s.

### Create the dependent data set (y)  #####
# Convert the dataframe to a numpy array 
y = np.array(df['Prediction'])
# Get all of the y values except the last '30' rows
y = y[:-forecast_out]
print(y)

#I can split them up into 80% training and 20 % testing data for the model(s).
# Split the data into 80% training and 20% testing

#
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#I can start creating and training the models 
#I test the model by getting the score also known as the coefficient of determination R² of the prediction. The best possible score is 1.0, and the model returns a score of 0.9274190417518909

# Create and train the Support Vector Machine (Regressor) 
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr_rbf.fit(x_train, y_train)

#Next I will create & train the Linear Regression model !
# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence

# Create and train the Linear Regression  Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

#lr confidence:  0.9304578441175994

#I am ready to do some forecasting / predictions. I will take the last 30 rows of data from the data frame of the Adj. Close price, and #store it into a variable called x_forecast after transforming it into a numpy array and dropping the ‘Prediction’ column of course. #Then I will print the data to make sure the 30 rows are all there.
# Set x_forecast equal to the last 30 rows of the original data set from Adj. Close column
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

# Print linear regression model predictions for the next '30' days
lr_prediction = lr.predict(x_forecast)
print("Linear Prediction Result",lr_prediction)

#I have arrived at the moment of truth. I will print out the future price (next 30 days) predictions of Amazon stock using the linear #regression model, and then print out the Amazon stock price predictions for te next 30 days of the support vector machine using the #x_forecast data !
print()
# Print support vector regressor model predictions for the next '30' days
svm_prediction = svr_rbf.predict(x_forecast)
print("SVM Prediction Result:",svm_prediction)
