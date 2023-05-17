from sklearn.model_selection import train_test_split
from sklearn import datasets  
from sklearn import svm
from SVM import SVM,np
from timeit import default_timer as timer 



X, y = datasets.make_blobs(
    n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# buildin
start = timer()
clf = svm.SVC(kernel = 'linear', C = 0.001,max_iter=1000)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
stop = timer()

# calc the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print (f"Buildin SVM Accuracy: {round(accuracy,2)}")
# print the time taken by Bulid in SVM model
buildin_time = round((stop - start),4)
print (f"SVM from scratch model toke: {buildin_time}")





# from scratch 
start = timer()
clf = SVM()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
stop = timer()

# calc the accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print (f"Scrach SVM Accuracy: {round(accuracy,2)}")
# print the time taken by scratch in SVM model
scratch_time = round((stop - start),4)
print (f"SVM from scratch model toke: {scratch_time}")



# compare between SVM Classification scratch and Buildin 

print (f"Time taken by scrach SVM = {scratch_time//buildin_time}x time taken from buildin SVM.")