import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import pickle

# Import Data
data = pd.read_csv('./datasets/preprocessed_data.csv')
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA
pca = PCA(n_components=0.95, random_state=42)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# MLP Classifier with PCA
mlp_classifier_pca = MLPClassifier(max_iter=1000, activation='relu', solver='adam', random_state=42)
mlp_classifier_pca.fit(x_train_pca, y_train)

pickle.dump(mlp_classifier_pca, open('./models/mlp.pkl', 'wb'))

# # Predict
# y_pred_mlp_pca = mlp_classifier_pca.predict(x_test_pca)

# # Accuracy
# accuracy_mlp_pca = accuracy_score(y_test, y_pred_mlp_pca)
# print(f'MLP with PCA Accuracy: {round((accuracy_mlp_pca*100), 2)}%')
