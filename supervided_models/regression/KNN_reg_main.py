# Main Section 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNN_buildin

from KNN import KNN,np

# load regression dataset 
X, y = datasets.make_regression (n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# calc the time taken per classifier
from timeit import default_timer as timer

#--------------------------------- From Scratch KNN reg ------------------

start = timer()

# define the classifer with K= 5 
clf = KNN(5)

# fit the data 
clf.fit(X_train,y_train)

# predict the output
predictions = clf.predict(X_test)

# calc the error
error = np.mean(abs(y_test - predictions))
print (f"Scratch KNN MAE Error: {round(error,2)}")
stop = timer()

# print the time taken by scratch model
scratch_time = round((stop - start),4)
print (f"KNN from scratch MOdel toke: {scratch_time}")

#--------------------------------- build in KNN reg ------------------------

start = timer()

# define the classifer with K = 5 
clf = KNN_buildin(5)

# fit the data 
clf.fit(X_train,y_train)

# predict the output
predictions = clf.predict(X_test)

# calc the error
error = np.mean(abs(y_test - predictions))
print (f"Buildin KNN MAE Error: {round(error,2)}")

stop = timer()

# print the time taken by Bulid in KNN model
buildin_time = round((stop - start),4)
print (f"KNN from scratch MOdel toke: {buildin_time}")


# compare between Scratch-KNN and Buildin 

print (f"Time taken by scrach KNN = {scratch_time//buildin_time}x time taken from buildin KNN.")

