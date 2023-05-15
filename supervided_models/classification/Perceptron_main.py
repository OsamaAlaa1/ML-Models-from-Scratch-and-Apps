from sklearn.model_selection import train_test_split
from sklearn import datasets
from Perceptron import np,Perceptron
from sklearn.linear_model import Perceptron as PC

from timeit import default_timer as timer 

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

X, y = datasets.make_blobs(
    n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

# Scratch Perceptron 
start = timer()
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)
end = timer()

print(f"Time Token by scratch Perceptron Model: {round(end-start,3)}")
print("Scratch Perceptron classification accuracy", accuracy(y_test, predictions))

# Build In Perceptron 
start = timer()
p = PC(alpha = 0.01, max_iter = 1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)
end = timer()

print(f"Time Token by Buildin Perceptron Model: {round(end-start,3)}")
print("Buildin Perceptron classification accuracy", accuracy(y_test, predictions))