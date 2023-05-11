
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Logistic_Regression import LogisticRegression, np 
from sklearn.linear_model import LogisticRegression as lr
from timeit import default_timer as timer

# load breast cancer data 
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Scratch Logistic Regression Model 
start = timer()
clf = LogisticRegression(lr = 0.01)
clf.fit(X_train, y_train) 
predictions = clf.predict (X_test)
stop = timer()

# calc the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print (f"Buildin Logistic Regression Accuracy: {round(accuracy,2)}")
# print the time taken by Bulid in Logistic Regression model
scratch_time = round((stop - start),4)
print (f"Logistic Regression from scratch model toke: {scratch_time}")



# build in Logistic regression 

start = timer()
# define the classifer with  
clf = lr(max_iter=1000,C = 0.01)
# fit the data 
clf.fit(X_train,y_train)
# predict the output
predictions = clf.predict(X_test)
stop = timer()

# calc the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print (f"Buildin Logistic Regression Accuracy: {round(accuracy,2)}")

# print the time taken by Bulid in Logistic Regression model
buildin_time = round((stop - start),4)
print (f"Logistic Regression from buildin model toke: {buildin_time}")


# compare between Scratch-Logistic Regression and Buildin 

print (f"Time taken by scrach Logistic Regression = {scratch_time//buildin_time}x time taken from buildin Logistic Regression.")