import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
import Perceptron as ppn
import math
def sigmoid(z):
    return 1.0/ (1.0 + np.exp(-z))
z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)
#print(sigma_z)
plt.plot(z, sigma_z)
plt.axvline(0.0, color='k') # place intersection at zero
plt.ylim(-0.1, 1.1) # ylim bottom set to -0.1, top set to 1.1 in the graph frame
plt.xlabel('z')
plt.ylabel('$\sigma (z)$')
# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()

def loss_1(z):
    return - np.log(sigmoid(z))
def loss_0(z):
    return - np.log(1 - sigmoid(z))
z = np.arange(-10, 10, 0.1) # create list of values between -10 and 10 with 0.1 intervals
sigma_z = sigmoid(z)
c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label='L(w, b) if y=1')
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, linestyle='--', label='L(w, b) if y=0')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w, b)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
     Parameters
     ------------
     eta : float
        Learning rate (between 0.0 and 1.0)
     n_iter : int
        Passes over the training dataset.
     random_state : int
        Random number generator seed for random weight
        initialization.
     Attributes
     -----------
     w_ : 1d-array
        Weights after training.
     b_ : Scalar
        Bias unit after fitting.
     losses_ : list
        Mean squared error loss function values in each epoch.
     """
    def __init__(self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter= n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
         Parameters
         ----------
         X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the
            number of examples and n_features is the
            number of features.
         y : array-like, shape = [n_examples]
            Target values.
         Returns
         -------
         self : Instance of LogisticRegressionGD
         """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0] # per definition
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self


    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# NOTE we will only consider setosa and versicolor flowers (class 0 and 1) to train our data.
# Since our logistic regression model only works for binary classification tasks when we fit it.
X_train_01_subset = ppn.X_train_std[(ppn.y_train == 0) | (ppn.y_train == 1)]
y_train_01_subset = ppn.y_train[(ppn.y_train == 0) | (ppn.y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.3,
                            n_iter=1000,
                            random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
ppn.plot_decision_regions(X=X_train_01_subset,
                          y=y_train_01_subset,
                          classifier=lrgd)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal wdith [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr') # solver='liblinear' is a highly optimized C/C++ as alternative solver
lr.fit(ppn.X_train_std, ppn.y_train)
ppn.plot_decision_regions(ppn.X_combined_std,
                          ppn.y_combined,
                          classifier=lr,
                          test_idx=range(105, 150))
plt.xlabel('Petal width [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(lr.predict_proba(ppn.X_test_std[:3, :]))
print(lr.predict_proba(ppn.X_test_std[:3, :]).sum(axis=1))
print(lr.predict_proba(ppn.X_test_std[:3, :]).argmax(axis=1))
print(lr.predict(ppn.X_test_std[:3, :])) # the more convenient way to obtain class labels

# predict class label of a single flower example
print(lr.predict((ppn.X_test_std[0, :].reshape(1, -1)))) #scikit-learn expect two-dimensional arrays as input. So a single row has to be sliced and converted into a two-dimensional array.

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c,
                            multi_class='ovr')
    lr.fit(ppn.X_train_std, ppn.y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='Petal length')
plt.plot(params, weights[:, 1],
         linestyle='--',
         label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()


