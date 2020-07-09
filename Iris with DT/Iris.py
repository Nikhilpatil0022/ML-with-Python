from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

print("Feature names of iris dataset:")
print(iris.feature_names)

print("Target names of iris dataset:")
print(iris.target_names)


'''print("First 10 elements fron iris dataset:")

for i in range(9):
    print("ID: {}, Label: {}, Feature: {}".format(i,iris.data[i],iris.target[i]))'''

#indices of removed elements
test_index = [15,65,110]

#training data with removed elements
train_target = np.delete(iris.target,test_index)
train_data = np.delete(iris.data,test_index,axis = 0)

#Testing data for testing on training data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

#Decision tree
classifier = tree.DecisionTreeClassifier()

#Apply training data to form tree
classifier.fit(train_data,train_target)

print("Values that we removed(Expected to be predicted):")
print(test_target)

print("Result of testing:")
print(classifier.predict(test_data))

#Visualisation
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(classifier,
               feature_names = iris.feature_names,
               class_names=iris.target_names,
               filled = True);
fig.savefig('IrisTree.pdf')
