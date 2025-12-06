#-------------------------------------------------------------------------
# AUTHOR: Timothy Tsang
# FILENAME: naive_bayes.py
# SPECIFICATION:
# FOR: CS 4440- Assignment #4
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

# Function to discretize temperature values to the nearest class value
def discretize(value):
    return min(classes, key=lambda x: abs(x - value))

#reading the training data
#--> add your Python code here
df = pd.read_csv('weather_training.csv', sep=',', header=0)
X_training = df.iloc[:, 1:-1].values.astype(float) # Features: Humidity, Wind Speed, Wind Bearing, Visibility, Pressure
y_training_original = df.iloc[:, -1].values.astype(float)  # Original temperature values

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
y_training = np.array([discretize(value) for value in y_training_original])

#reading the test data
#--> add your Python code here
test_data = pd.read_csv('weather_test.csv')
X_test = test_data.iloc[:, 1:-1].values.astype(float)  # Features: Humidity, Wind Speed, Wind Bearing, Visibility, Pressure
y_test_original = test_data.iloc[:, -1].values.astype(float) # Original temperature values

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
# Discretize temperature values to the nearest class value
y_test = np.array([discretize(value) for value in y_test_original])

#loop over the hyperparameter value (s)
#--> add your Python code here
highest_accuracy = 0.0

for s in s_values:

    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here
    correct_predictions = 0
    predictions = clf.predict(X_test)
    
    for i in range(len(y_test_original)):
        predicted_value = predictions[i]
        real_value = y_test_original[i]
        
        if real_value == 0:
            if predicted_value == 0:
                correct_predictions += 1
        else:
            percent_diff = 100 * (abs(predicted_value - real_value) / abs(real_value))
            if percent_diff <= 15:
                correct_predictions += 1
    
    accuracy = correct_predictions / len(y_test_original)

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy:.2f}, Parameters: s={s}")



