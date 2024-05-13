import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('./datasets/preprocessed_data.csv')

X = df.drop('HeartDisease', axis=1)
Y = df['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

dt_classifier = DecisionTreeClassifier()

dt_classifier = dt_classifier.fit(x_train, y_train)

rf_classifier = RandomForestClassifier(verbose=1)
rf_classifier = rf_classifier.fit(x_train, y_train)

pickle.dump(dt_classifier, open('./models/dt.pkl', 'wb'))
pickle.dump(rf_classifier, open('./models/rf.pkl', 'wb'))