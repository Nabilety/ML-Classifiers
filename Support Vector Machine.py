import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import Perceptron as ppn
from sklearn.linear_model import SGDClassifier

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(ppn.X_train_std, ppn.y_train)
ppn.plot_decision_regions(ppn.X_combined_std,
                          ppn.y_combined,
                          classifier=svm,
                          test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# SVC uses LIBSVM, equivalent to C/C++ library specialized for SVMs, similar to the LIBLINEAR for the Logistic Regression
# They allow for quick training of large amounts of linear classifiers.
# But sometimes when datasets are too large to fit the computer memory, we have to use SGDClassifier class.
# SGDClassifier class suports online learning via partial_fit method. Concept is similar to stochastic gradient algorithm.

# initialized SGD version of perceptron (loss='perceptron'), logistic regression (loss='log') and SVM (loss='hinge'):
ppn_sgd = SGDClassifier(loss='perceptron')
lr_sgd = SGDClassifier(loss='log')
svm_sgd = SGDClassifier(loss='hinge')
print(ppn_sgd)
print(lr_sgd)
print(svm_sgd)

# Solving non-linear problems using a kernel SVM

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='royalblue',
            marker='s',
            label='Class 1')
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            c='tomato',
            marker='o',
            label='Class 0')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# Using the kernel trick to find separating hyperplanes in higher dimensional space
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
ppn.plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(ppn.X_train_std, ppn.y_train)
ppn.plot_decision_regions(ppn.X_combined_std,
                          ppn.y_combined,
                          classifier=svm,
                          test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

svm = SVC(kernel='rbf', random_state=1, gamma=100, C=1.0) # example of overfitting, due to high generalization error on unseen data.
svm.fit(ppn.X_train_std, ppn.y_train)
ppn.plot_decision_regions(ppn.X_combined_std,
                          ppn.y_combined,
                          classifier=svm,
                          test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
