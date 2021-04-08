#-------------------------------------------------------------------------
# AUTHOR: Jonathan Lu
# FILENAME: svm.py
# SPECIFICATION: Combine SVM hyperparameters to predict best performance
# FOR: CS 4210 - Assignment #3
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
highestParameters = ()

#reading the data in a csv file
with open('C:/Users/Administrator/Desktop/optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('C:/Users/Administrator/Desktop/optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
for c_value in c: #iterates over c
    for degree_value in degree: #iterates over degree
        for kernel_type in kernel: #iterates kernel
           for df_shape in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=c_value, degree=degree_value, kernel=kernel_type, decision_function_shape=df_shape)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                numCorrect = 0
                for instance in dbTest:
                    class_predicted = clf.predict([instance[:-1]])[0]
                    numCorrect += int(class_predicted == instance[-1])
                currentAccuracy = numCorrect / len(dbTest)

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if currentAccuracy > highestAccuracy:
                    highestAccuracy = currentAccuracy
                    print("Highest SVM accuracy so far: %.5f, Parameters: a=%d, degree=%d, kernel=%s, decision_fucntion_shape=%s" % (highestAccuracy, c_value, degree_value, kernel_type, df_shape))
                    highestParameters = (c_value, degree_value, kernel_type, df_shape)

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
#--> add your Python code here
print("Highest SVM accuracy: %.5f, Parameters: a=%d, degree=%d, kernel=%s, decision_fucntion_shape=%s" % (highestAccuracy, *highestParameters))