from scipy.special import comb
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) * error ** k * (1 - error) ** (n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)
print(ensemble_error(n_classifier=11, error=0.25))

# relationship between ensemble and base errors in a line graph
error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=11, error=error) for error in error_range]

plt.plot(error_range,
         ens_errors,
         label='Ensemble error',
         linewidth=2)
plt.plot(error_range,
         error_range,
         linestyle='--',
         label='Base error',
         linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()


# # Combining classifiers via majority vote

# ## Implementing a simple majority vote classifier
# bincount counts the number of occurrences of each class label
# argmax returns index with position of highest count, corresponding to majority class label (assuming the class label starts at 0)
np.argmax(np.bincount([0, 0, 1],
                      weights=[0.2, 0.2, 0.6]))

# Alternatively we can use predict_proba to return the probability of a predicted class label.
# This can be useful if the classifiers in our ensemble are well calibrated.

# Assume our classifiers C_j returns following class membership probabilities for a particular example x:
# C_1(x) -> [0.9, 0.1], C_2(x) -> [0.8, 0.2], C_3(x) -> [0.4, 0.6]
# Using the same weights 0.2, 0.2, 0.6 like before we can calculate individual class probabilities:
# argmax_i[p(i_0 | x) , p(i_1 | x)]
# p(i_0 | x) 0.2 × 0.9 + 0.2 × 0.8 + 0.6 × 0.4
# p(i_1 | x) 0.2 × 0.1 + 0.2 × 0.2 + 0.6 × 0.6

ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
print(p)
print(np.argmax(p)) # thus the predicted class is label 0


# Implement MajorityVoteClassifier:
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble

    vote : str, {'classlabel', 'probability'} (default='classlabel')
      If 'classlabel' the prediction is based on the argmax of
        class labels. Else if 'probability', the argmax of
        the sum of probabilities is used to predict the class label
        (recommended for calibrated classifiers).

    weights : array-like, shape = [n_classifiers], optional (default=None)
      If a list of `int` or `float` values are provided, the classifiers
      are weighted by importance; Uses uniform weights if `weights=None`.

    """
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {
            key: value for key, value in _name_estimators(classifiers)
        }
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        y : array-like, shape = [n_examples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote must be 'probability' "
                             f"or 'classlabel'"
                             f"; got (vote={self.vote}")
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Number of classifiers and'
                             f' weights must be equal'
                             f'; got {len(self.weights)} weights,'
                             f' {len(self.classifiers)} classifiers')
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Matrix of training examples.

        Returns
        ----------
        maj_vote : array-like, shape = [n_examples]
            Predicted class labels.

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else: # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples and
            n_features is the number of features.

        Returns
        ----------
        avg_proba : array-like, shape = [n_examples, n_classes]
            Weighted average probability for each class per example.
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba # useful when computing ROC AUC

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out

# The preceeding implementation for MajorityVoteClassifier is also available as sklearn.ensemble.VotingClassifier
# Although more sophisticated version of this, we implemented this for demonstration purposes

# ## Using the majority voting principle to make predictions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
"""
try:
    df_wine = pd.read_csv('wine.data', header=None, encoding='utf-8')
except FileNotFoundError:
    print("File not found! Please make sure 'wine.data exists")
"""
# Following example we will use two features sepal width and petal length.
# Furthermore we classify only from Iris-versicolor and Iris-virginica even though our MajorityVoteClassifier
# generalizes to mulitclass, which we will compute the ROC AUC for.
iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
# Using this training data we wil train three classifiers
# Logistic regression, Decision tree, k-nearest neighbors
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1 = LogisticRegression(penalty='l2',
                          C=0.001,
                          solver='lbfgs',
                          random_state=1)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
# Unlike decision trees, KNN and LR are not scale-invariant.
# Even though the Iris features are all measured on the same scale (cm), it is good practice
# to work with standardized features. There we use pipelines on KNN and LR
pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')

# Majority Rule (hard) Voting
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels += ['Majority voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')

# # Evaluating and tuning the ensemble classifier
# remember test dataset must not be used for model selections.
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label=f'{label} (auc = {roc_auc:.2f}')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()
# both ensemble classifier and logistic regression performs well on the test dataset
# the latter is most likely due to the high variance (in this case, because the sensitivity of how we split the dataset)
# given the small size of the dataset


# Decision region of the ensemble classifier
# we standardize the training features prior to model fitting, although it is not needed for
# logistic regression and KNN since the pipeline will automatically take care of that.
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                  X_train_std[y_train == 0, 1],
                                  c='blue',
                                  marker='^',
                                  s=50)

    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                  X_train_std[y_train == 1, 1],
                                  c='green',
                                  marker='o',
                                  s=50)

    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-3.5, -5.,
         s='Sepal width [standardized]',
         ha='center', va='center', fontsize=12)
plt.text(-12.5, 4.5,
         s='Petal length [standardized]',
         ha='center', va='center',
         fontsize=12, rotation=90)

plt.show()


# Before turning individual classifier's parameter for ensemble classification,
# let's call the get_params method to get an idea of how we can access individual parameters inside a GridSearchCV
clf_params = mv_clf.get_params()
print(clf_params)

# now lets tune inverse regularization parameter C for LR and depths for decision tree for show
from sklearn.model_selection import GridSearchCV
params = {'decisiontreeclassifier': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean_score = grid.cv_results_['mean_test_score'][r]
    std_dev = grid.cv_results_['std_test_score'][r]
    params = grid.cv_results_['params'][r]
    print(f'{mean_score:.3f} +/- {std_dev:.2f} {params}')

print(f'Best parameters: {grid.best_params_}')
print(f'ROC AUC: {grid.best_score_:.2f}')

# **Note**
# By default, the default setting for `refit` in `GridSearchCV` is `True` (i.e., `GridSeachCV(..., refit=True)`), which means that we can use the fitted `GridSearchCV` estimator to make predictions via the `predict` method, for example:
#
#     grid = GridSearchCV(estimator=mv_clf,
#                         param_grid=params,
#                         cv=10,
#                         scoring='roc_auc')
#     grid.fit(X_train, y_train)
#     y_pred = grid.predict(X_test)
#
# In addition, the "best" estimator can directly be accessed via the `best_estimator_` attribute.

grid.best_estimator_.classifiers

mv_clf = grid.best_estimator_

mv_clf.set_params(**grid.best_estimator_.get_params())

mv_clf



# # Bagging -- Building an ensemble of classifiers from bootstrap samples
# We will consider only Wine class 2 and 3, and select only 2 features
# - Alcohol and OD280/OD315 to make classification more complex
try:
    df_wine = pd.read_csv('wine.data', header=None, encoding='utf-8')
except FileNotFoundError:
    print("File not found! Please make sure 'wine.data exists")

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# drop 1 class
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=None)
bag = BaggingClassifier(estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=1,
                        random_state=1)

# Calculate accuracy score and compare performance of bagging with performance of single unpruned decision tree
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

# Based on accuracy values above, the unpruned decision tree predicts all class labels
# of the training examples correctly; but the substantially lower test accuracy indicates high variance (overfit)
bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)
print(f'Bagging train/test accuracies '
      f'{bag_train:.3f}/{bag_test:.3f}')

# Thus the bagging classifier has a slightly better generalization performance on the test dataset

# Compare decision regions between decision tree and bagging classifier:
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))

for idx, clf, tt in zip([0, 1],
                        [tree, bag],
                        ['Decision tree', 'Bagging']):
    clf.fit(X_train, y_train)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0],
                       X_train[y_train == 0, 1],
                       c='blue', marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                       X_train[y_train == 1, 1],
                       c='green', marker='o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,
         s='Alcohol',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

# Note bagging is ineffective in reducing model bias, meaning models too simple to capture trends in data well
# So that's why we want to perform bagging on an ensemble of classifier with low bias, i.e. unpruned decision trees.




# # Leveraging weak learners via adaptive boosting

# ## How boosting works

# Using concrete example on pp. 231
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
correct = (y == yhat)
weights = np.full(10, 0.1)
print(weights)

# compute wieghted error rate ε
# ~correct invert the array such that np.mean(~correct) comptues the proportion of incorrect predicitons
# (True counts as value 1 and False as 0, meaning the classification error)
epsilon = np.mean(~correct)
print(epsilon)


# compute coefficient α_j
alpha_j = 0.5 * np.log((1 - epsilon)/epsilon)
print(alpha_j)

# compute weighted vector w := w × exp(-α_j × yhat × y)

# if yhat_i is correct, then yhat_i × y_i will have a positive sign as we decrease the ith weight, since α_j is positive
update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
print(update_if_correct)

# similarly, we increase the ith weight if yhat_i is incorrectly labeled
update_if_wrong_1 = 0.1 * np.exp(-alpha_j * 1 * -1)
print(update_if_wrong_1)

# We then use these values to update the weight as follows:
weights = np.where(correct == 1,
                   update_if_correct,
                   update_if_wrong_1)
print(weights)

# normalize the weights so that they sum up to 1:
normalized_weights = weights / np.sum(weights)
print(normalized_weights)

# and so each weight that corresponds to a correctly classified example will be reduced from initial value 0.1 to 0.0714
# the next round of boosting. Similarly the  weights incorrectly classified wil increase from 0.1 to 0.1667




# ## Applying AdaBoost using scikit-learn

# Train AdaBoostClassifier on 500 decision tree stumps
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              random_state=1,
                              max_depth=1)
ada = AdaBoostClassifier(estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print(f'Decision tree train/test accuracies '
      f'{tree_train:.3f}/{tree_test:.3f}')

# As we can see the decision tree stump seems to underfit on the training data in contrast to the unpruned decision tree from before
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print(f'AdaBoost train/test accuracies '
      f'{ada_train:.3f}/{ada_test:.3f}')

# thus, AdaBoost model predict all class labels of the training dataset correctly, and also show the slightly improved test dataset
# However, we can also see that we introduced additional variance with our attempt to reduce the model bias - a greater gap between training and test performance.



# Compare decision regions between decision tree and AdaBoost classifier:
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(1, 2,
                        sharex='col',
                        sharey='row',
                        figsize=(8, 3))
for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Decision tree', 'Adaboost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0, 0],
                       X_train[y_train==0, 1],
                       c='blue',
                       marker='^')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                       X_train[y_train == 1, 1],
                       c='green',
                       marker='o')
    axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
plt.tight_layout()
plt.text(0, -0.2,
         s='OD280/OD315 of diluted wines',
         ha='center',
         va='center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

# from this we can conlcude that the AdaBoost model provides a more complex decision boundary than the decision stump.
# additionally, the Adaboost separates feature space very similarly to the bagging classifier.

# As a concluding note about the ensemble techniques, it's worth knowing that ensemble learning increases computational
# complexity compared to individual classifiers. In practice we need to take that into account, whether we want to pay the price
# of increased computational costs for an often relatively modest improvement in predictive performance.

# the tradeoff is simply; if the additiona accuracy gain measured doesn't justify the engineering effort needed, then don't implement









# # Gradient boosting -- training an ensemble based on loss gradients

# ## Comparing AdaBoost with gradient boosting

# ## Outlining the general gradient boosting algorithm

# ## Explaining the gradient boosting algorithm for classification

# ## Illustrating gradient boosting for classification

















# ## Using XGboost

import xgboost as xgb
# fit gradient boosting classifier with 1000 trees (rounds). Learning rate between 0.01 and 0.1 typically.
# Note learning rate is used for scaling predictions from individual rounds.
# So in hinsight the lower the rate, the more estimators are required to achieve accurate prediction
# Max_depth for individual decision trees set to 4.
# use_label_encoder=False disables warning messages informing users that XGBoost isn't converting labels by defualt anymore
# and so it expects users to provide labels in an integer format starting with label 0
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01,
                          max_depth=4, random_state=1,
                          use_label_encoder=False)
gbm = model.fit(X_train, y_train)
y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)
gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)
print(f'XGboost train/test accuracies '
      f'{gbm_train:.3f}/{gbm_test:.3f}')














