import numpy as np
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

m = 100
X = 6*np.random.rand(m, 1)-3
y = 0.5*X**2+X+2+np.random.randn(m, 1)

tree_reg = DecisionTreeRegressor(min_samples_leaf=10)
tree_reg.fit(X, y)
