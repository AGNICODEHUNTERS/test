import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

x,y = make_classification(
    n_samples = 10000, n_features = 4,
    n_informative = 2, n_redundant = 0,
    random_state = 0, shuffle = False
)
#print(x)
clf = RandomForestClassifier(max_depth=2, random_state = 0,n_estimators=10)
clf.fit(x,y)
print(clf.score(x,y))
