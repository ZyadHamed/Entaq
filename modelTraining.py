import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
dataset = load("dataset.joblib")
arr = dataset.drop('label', axis=1).to_numpy()
arr = np.reshape(arr, (1680, 240, 240))

print("All Data Have been Loaded!")
arr = np.reshape(arr, (1680, -1))
labels = dataset['label'].to_numpy()
#training_arr = load("hogvectors.joblib")
print("Hogs have been loaded successfully!")
#clf = svm.SVC(gamma=0.001)
clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
X_train, X_test, y_train, y_test = train_test_split(
    arr, labels, test_size=0.3, shuffle=True
)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)
dump(clf, 'myOwnModel.joblib')