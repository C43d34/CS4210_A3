#-------------------------------------------------------------------------
# AUTHOR: Cleighton Greer
# FILENAME: svm.py
# SPECIFICATION: develop most optimal SVM classifier for a given set of data using Grid Search Method for parameters:(c, degree, kernel, and decision_function_shape) 
# FOR: CS 4210- Assignment #3
# TIME SPENT: 
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

configuration_iteration = 0
best_config_accuracy = 0
best_configuration = ["", "", "", ""]
for c_val in c:
    for deg in degree:
        for kernel_type in kernel:
           for func_shape in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                svm_model = svm.SVC(C=c_val, degree=deg, kernel=kernel_type, decision_function_shape=func_shape)

                #Fit SVM to the training data
                svm_model.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                ##TESTING
                correct_prediction_cnt = 0
                total_prediction_cnt = len(X_test)
                for i in range(0, len(X_test)):
                    test_prediction = svm_model.predict([X_test[i]])
                    correct_prediction_cnt = correct_prediction_cnt+1 if test_prediction[0] == y_test[i] else correct_prediction_cnt
                

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                print(f"\n~~~~~~~~~~~~~~~~~~~Iteration {configuration_iteration}")
                accuracy = correct_prediction_cnt/total_prediction_cnt

                if accuracy > best_config_accuracy: #new configuration better accuracy
                    best_configuration = [c_val, deg, kernel_type, func_shape]
                    best_config_accuracy = accuracy
                #otherwise, retain old best configuration   

                print(f"Current Best SVM Configuration: C:{best_configuration[0]}, Deg:{best_configuration[1]}, Kernel:{best_configuration[2]}, Dec Func Shape:{best_configuration[3]}")                 
                print(f"Accuracy: {best_config_accuracy:.3f}")

                configuration_iteration += 1





