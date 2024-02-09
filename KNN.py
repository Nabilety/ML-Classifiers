from sklearn.neighbors import KNeighborsClassifier
import Perceptron as ppn
import matplotlib.pyplot as plt


# K-nearest neighbors - a lazy learning algorithm

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski') # distance is Euclidian if p=2, Manhatan if p=1

knn.fit(ppn.X_train_std, ppn.y_train)
ppn.plot_decision_regions(ppn.X_combined_std, ppn.y_combined, classifier=knn, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
