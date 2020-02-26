from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1,],[0,1]]
Y = [0, 1, 1]
clf = RandomForestClassifier(n_estimators=100,warm_start=True,max_features=None)
clf = clf.fit(X, Y)

P=[[1,0]]

A=clf.predict(P)

print(A)
