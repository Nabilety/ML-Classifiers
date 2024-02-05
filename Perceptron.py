from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # petal length and width for all samples
y = iris.target # class labels

print('Class labels:', np.unique(y)) #Setosa = 0, Versicolor = 1, Virginica = 2

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardizing the features.
# Remember the aim here is to achieve faster convergence,
# and have prevent our algorithm from being affected by the feature with the highest scale
# For instance, x1 values could range between [0,1] while x2 values range between [100,1000]
# And because of this difference, x2 would dominate and overshoot during the optimization.
# But when we standardize, we make sure to transform each feature is centered at 0,
# and each feature has a standard deviation at 1.
# This way we prevent overshooting, and give each feature equal importance, by shrinking all the data
sc = StandardScaler()
sc.fit(X_train)
#print(X_train) # row 1: [1.4 0.2]
print(sc.mean_) # Mean value for each feature, µ_j: [3.78952381 1.19714286]
print(sc.scale_) # Standard deviation for each feature, σ_j: [1.79299822 0.76275904]
X_train_std = sc.transform(X_train)
#print((1.4 - 3.78952381) / 1.79299822) # standardization: x_j' = (x_j - µ_j) / σ_j
#print(X_train_std) # row 1: [[-1.33269725 -1.30728421]
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d ' % (y_test != y_pred).sum()) # if-clause summing up how many test results were misclassified
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) # note y_test is true class labels, y_pred are the predicted
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #print(x1_min, x1_max, x2_min, x2_max)
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='none',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='Test set')


# Training a perceptron model using the standardized training data:


#print(X_train_std)
#print(X_test_std)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('figures/03_01.png', dpi=300)
plt.show()

