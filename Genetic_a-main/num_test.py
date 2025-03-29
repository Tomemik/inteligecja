import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools

url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
df = pd.read_csv(url, header=None, compression='gzip')

# Nazwy kolumn
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]
df.columns = columns

# Kodowanie danych kategorycznych
le = LabelEncoder()
for col in ["protocol_type", "service", "flag", "label"]:
    df[col] = le.fit_transform(df[col])

# Podział na dane treningowe i testowe
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Algorytm genetyczny do selekcji cech
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def init_individual():
    return [random.randint(0, 1) for _ in range(X_train.shape[1])]

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja oceny z karą za dużą liczbę cech
def evaluate(individual):
    selected_features = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected_features:
        return 0,
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    model = DecisionTreeClassifier()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Kara za liczbę cech
    penalty_factor = 0.00002  # Można dostosować
    penalty = penalty_factor * len(selected_features)
    
    return accuracy - penalty,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Parametry algorytmu genetycznego
population = toolbox.population(n=50)
NGEN = 20
CXPB, MUTPB = 0.5, 0.2

# Uruchomienie algorytmu
for gen in range(NGEN):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values, child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# Najlepsze rozwiązanie
best_individual = tools.selBest(population, 1)[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
print("Wybrane cechy:", selected_features)

# Klasyfikacja z wybranymi cechami
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

model = DecisionTreeClassifier()
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# Ocena końcowa
final_accuracy = accuracy_score(y_test, y_pred)
print("Końcowa dokładność klasyfikatora:", final_accuracy)

print("Nazwy wybranych cech:", list(X_train.columns[selected_features]))