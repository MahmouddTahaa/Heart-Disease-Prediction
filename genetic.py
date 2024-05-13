import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def get_best_features(x):
    # HeartDiseases Data

    scaler = StandardScaler()

    data = pd.read_csv(x)[:20000]
    data.dropna(inplace=True)
    cols = data.columns
    label_encoder = LabelEncoder()
    for col in cols:
        data[col] = label_encoder.fit_transform(data[col])

    X = data.drop(columns="HeartDisease").values
    X = scaler.fit_transform(X)
    y = data["HeartDisease"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )

    # Fitness Function () based on mean square error of model
    def fitness_function(selected_features):
        selected_indices = []

        for i, bit in enumerate(selected_features):
            if bit == 1:
                selected_indices.append(i)
        if len(selected_indices) == 0:
            return float("inf")

        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        model = LinearRegression()
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    def genetic_algorithm(population_size, num_generations, num_features):
        population = np.random.randint(2, size=(population_size, num_features))

        for generation in range(num_generations):
            fitness_scores = []

            for individual in population:
                fitness_scores.append(fitness_function(individual))
            elite_indices = np.argsort(fitness_scores)[
                : int(population_size * 0.3)
            ]  # Top 30%

            elite_population = []
            for i in elite_indices:
                elite_population.append(population[i])

            offspring = []
            while len(offspring) < population_size - len(elite_population):
                parent1, parent2 = random.choices(elite_population, k=2)
                crossover_point = random.randint(1, num_features - 1)
                child = np.concatenate(
                    (parent1[:crossover_point], parent2[crossover_point:])
                )
                # mutation
                mutation_point = random.randint(0, num_features - 1)
                child[mutation_point] = 1 - child[mutation_point]  # Flip the bit
                offspring.append(child)

            # new
            population = elite_population + offspring
            best_individual = population[np.argmin(fitness_scores)]
            #print(f"Generation {generation + 1}: Best Individual = {best_individual}, Best MSE = {min(fitness_scores)}")

        # Return the best individual found of each generation
        return population[np.argmin(fitness_scores)]

    num_features = X_train.shape[1]  # Number of features
    best_features = genetic_algorithm(
        population_size=10, num_generations=20, num_features=num_features
    )
    #print(f"Best features found:{best_features}")

    selected_feature_indices = []
    for i in range(len(best_features)):
        if best_features[i] == 1:
            selected_feature_indices.append(i)
    selected_feature_names = data.iloc[:, 1:].columns[selected_feature_indices]

    return selected_feature_names.values


selected_features = get_best_features("./datasets/preprocessed_data.csv")

data = pd.read_csv('./datasets/preprocessed_data.csv')

X = data.loc[:, selected_features]
y = data['HeartDisease']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier using the selected features
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train.loc[:, selected_features], y_train)

pickle.dump(clf, open('./models/genetic.pkl', 'wb'))

