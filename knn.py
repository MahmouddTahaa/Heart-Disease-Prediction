import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset = pd.read_csv('./datasets/preprocessed_data.csv')

X = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = KNeighborsClassifier()
classifier.fit(X_train,y_train)

pickle.dump(classifier, open('./models/knn.pkl', 'wb'))