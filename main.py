import pandas as pd

#Setting up training data
claims = pd.read_csv('COVIDClaimsTrimmed.csv', sep=',')

X = claims[['Claims Paid for Testing','Claims Paid for Treatment','Claims Paid for Vaccine']]
y = claims['State']

#Class distribution
class_count = {}
for state in y:
    class_count[state] = class_count.get(state, 0) + 1
for c in class_count:
    print("Class: {} has {:4} occurences".format(c, class_count[c]))

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
testSize = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


#Logistic Regression Start
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
pipe.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}%".format(100*pipe.score(X_train, y_train)))
print("Test set score: {:.2f}%".format(100*pipe.score(X_test, y_test)))


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

#plot KNN confusion matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.title("KNN Confusion Matrix")
plt.show(block=False)

#plot logistic regression confusion matrix
plot_confusion_matrix(pipe, X_train, y_train)
plt.title("Logistic Regression Confusion Matrix")
plt.show(block=False)



#Used to stop the script from exiting
input()