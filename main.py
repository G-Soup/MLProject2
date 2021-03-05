import pandas as pd

#Setting up training data
claims = pd.read_csv('COVIDClaimsTrimmed.csv', sep=',')

X = claims[['Claims Paid for Testing','Claims Paid for Treatment','Claims Paid for Vaccine']]
y = claims['State']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
testSize = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)



#Logistic Regression Start
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=100, random_state=42)
lr.fit(X_train, y_train)

#########################################
# Coefficients of linear model (b_1,b_2,...,b_p): log(p/(1-p)) = b0+b_1x_1+b_2x_2+...+b_px_p
#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))

# Estimate the accuracy of the classifier on future data, using the test data
##########################################################################################
print("Training set score: {:.2f}%".format(100*lr.score(X_train, y_train)))
print("Test set score: {:.2f}%".format(100*lr.score(X_test, y_test)))


# Create classifier object: kNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print("kNN Training set score: {:.2f}%".format(100*knn.score(X_train, y_train)))
print("kNN Test set score: {:.2f}%".format(100*knn.score(X_test, y_test)))

