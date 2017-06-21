import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

import os
os.environ["PATH"] += os.pathsep + 'C:\graphviz\bin'

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(test_data)
print(clf.predict(test_data))

# from IPython.display import Image
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())

# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])



# import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")