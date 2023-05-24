from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
# print(type(iris))
# print(list(iris))
# print(type(iris["data"]))
# print(type(iris["target"]))
# print(type(iris["frame"]))
# print(iris["data"])
# print(iris["target"])
# print(iris["frame"])

X = iris.data[:, 2:]
y = iris["target"]

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Actually nothing.
