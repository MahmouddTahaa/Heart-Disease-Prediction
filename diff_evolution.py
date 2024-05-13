import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('./datasets/preprocessed_data.csv')

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate(individual):
    # Convert binary individual to indices of selected features
    selected_features = [index for index, bit in enumerate(individual) if bit]
    if len(selected_features) == 0:
        return (0,)  # Ensure at least one feature is selected
    # Train a Random Forest classifier and calculate accuracy
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train.values[:, selected_features], y_train)
    y_pred = clf.predict(X_test.values[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    return (accuracy)


# Differential Evolution parameters
population_size = 5
crossover_rate = 0.9
mutation_rate = 0.05
generations = 3

# Create initial population with random binary individuals (feature subsets)
population = X.iloc[:population_size, :].values.copy()


# Execute Differential Evolution algorithm
for gen in range(generations):
    print("Generation:", gen)
    for i in range(population_size):
        print("Individual:", i)
        print(population[i])

        # Select three individuals randomly from the population
        a, b, c = np.random.choice(population_size, 3, replace=False)

        # Perform mutation
        mutant = population[a] + mutation_rate * (population[b] - population[c])
        mutant = np.clip(mutant, 0, 1)  # Ensure values are within [0, 1]
        print(mutant)
        # Perform crossover
        crossover_points = population[i] < crossover_rate
        trial = np.where(crossover_points, mutant, population[i])
        print(trial)
        # Evaluate trial individual and replace if better
        if evaluate(trial) > evaluate(population[i]):
            population[i] = trial
            
            
# Get the best individual (subset of features)
best_individual_index = np.argmax([evaluate(ind) for ind in population])
best_individual = population[best_individual_index]
selected_features = [index for index, bit in enumerate(best_individual) if bit]

selected_features.pop(0)

# Load the dataset
data = pd.read_csv('./datasets/preprocessed_data.csv')

# Extract features (X) and target variable (y)
X = data.loc[:, data.columns[selected_features]]
y = data['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier using the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

pickle.dump(clf, open('./models/evolution.pkl', 'wb'))