from sklearn import tree
# following datasets use a 2D array for storing trainig data for this first learning application 
# [[weight in grams, shape], ...]
# eg. [[110, "smooth"], [120, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# now we change String features as 0 for "smooth" and 1 for "bumpy"
features = [[110, 0], [130, 0], [150, 1], [170, 1]]
# Labels as 1 for Orance and 0 Apple 
labels = [1,1,0,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print(clf.predict([[175, 1]]))