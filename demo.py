from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import accuracy_score

X = [[181,80,44], [177,70,43], [160,60,38], [154, 54, 37], [166,65,40], [190,90,47],
	 [175, 64,39], [177,70,40], [159,55,37], [171,75,42], [181,85,43]]
Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

clf2 = svm.SVC()
clf2 = clf2.fit(X,Y)

clf3 = GaussianNB()
clf3 = clf3.fit(X,Y)

clf4 = neighbors.KNeighborsClassifier()
clf4 = clf4.fit(X,Y)

X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],
		[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

Y_predict = clf.predict(X_test)
Y_predict2 = clf2.predict(X_test)
Y_predict3 = clf3.predict(X_test)
Y_predict4 = clf4.predict(X_test)

print("Prediction by Naive Bayes:", Y_predict2)
print("Decision Tree: ", accuracy_score(Y_test, Y_predict))
print("Support Vector Classifier: ", accuracy_score(Y_test, Y_predict2))
print("Naive Bayes: ", accuracy_score(Y_test, Y_predict3))
print("Neighbors: ", accuracy_score(Y_test, Y_predict4))