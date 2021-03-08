from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

# Question 1
print("1. The dataset has 7 features")


# Setting up training data
claims = pd.read_csv('COVIDClaimsTrimmed.csv')

X = claims[['Claims Paid for Testing',
            'Claims Paid for Treatment', 'Claims Paid for Vaccine']]
y = claims['State']

# Class distribution
class_count = {}
for state in y:
    class_count[state] = class_count.get(state, 0) + 1

print("2. \n  a.", sep="")
for c in class_count:
    print("    Class: {} has {:4} occurences".format(c, class_count[c]))

# random_state: set seed for random# generator
# test_size: default 25% testing, 75% training
testSize = 0.25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=testSize, random_state=42)

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(X_train, marker='o', s=40, hist_kwds={
                         'bins': 15}, figsize=(9, 9), cmap=cmap)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['Claims Paid for Testing'], X_train['Claims Paid for Treatment'], X_train['Claims Paid for Vaccine'],
           marker='o', s=100)
ax.set_xlabel('Claims Paid for Testing')
ax.set_ylabel('Claims Paid for Treatment')
ax.set_zlabel('Claims Paid for Vaccine')
plt.show()

# Logistic Regression Start
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
pipe.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Logistic Regression Training set score: {:.2f}%".format(
    100*pipe.score(X_train, y_train)))
print("Logistic Regression Test set score: {:.2f}%".format(
    100*pipe.score(X_test, y_test)))

# SVM
# Create classifier object: Create a nonlinear SVM classifier
# kernel, default=’rbf’ = radial basis function
svc = SVC(C=10, gamma='auto', random_state=100)
svc.fit(X_train, y_train)
print("SVM Gaussian Training set score: {:.2f}%".format(
    100*svc.score(X_train, y_train)))
print("SVM Gaussian Test set score: {:.2f}%".format(
    100*svc.score(X_test, y_test)))


# KNN
knn = KNeighborsClassifier(n_neighbors=2600, weights='uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(
    100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))


# Question 3
print("3. Dataset partition: {}% training and {}% testing".format(
    (100*(1-testSize)), (100 * testSize)))


# plot KNN confusion matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.title("KNN Confusion Matrix")
plt.show(block=False)

# plot logistic regression confusion matrix
plot_confusion_matrix(pipe, X_train, y_train)
plt.title("Logistic Regression Confusion Matrix")
plt.show(block=False)


# plot SVM confusion matrix
plot_confusion_matrix(svc, X_test, y_test)
plt.title("SVM Confusion Matrix")
plt.show(block=False)


# Used to stop the script from exiting
input()
