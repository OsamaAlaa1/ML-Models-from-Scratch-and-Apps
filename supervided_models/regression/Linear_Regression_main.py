import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Linear_Regression import LinearRegression


# load the dataset 
X, y = datasets.make_regression (n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# model section 
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)
predictions = linear_reg.predict(X_test)

# create mean absolute error 
def mae(y_test,predictions):
    return np.mean(abs(y_test - predictions))

# print the mean square error 
print(mae(y_test,predictions))

# plot the final tuned line
y_pred_line = linear_reg.predict (X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()