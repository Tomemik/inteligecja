
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from deap import base, creator, tools, algorithms
import random
from collections import defaultdict

# === Wczytanie danych z KDD Cup 99 ===
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

# Losowanie po 200 próbek na klasę
df = df.groupby("label").apply(lambda x: x.sample(n=min(len(x), 200), random_state=42)).reset_index(drop=True)


# Wypisz liczbę próbek dla każdej klasy
print("\\n📊 Liczba próbek dla każdej klasy (200 losowych):")
print(df["label"].value_counts())

# Kodowanie kolumn kategorycznych (poza label)
for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Dane wejściowe
X = df.drop(columns=["label"])
y = df["label"]

# Normalizacja
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Parametry GA
num_features = X.shape[1]
POP_SIZE = 50
N_GEN = 30
CXPB = 0.5
MUTPB = 0.2

# Wagi
weights = np.ones(num_features)
important_indices = [22, 23, 24, 25, 38, 39]
for idx in important_indices:
    weights[idx] = 2.0

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: random.uniform(0, 1))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja dystansu
def weighted_euclidean(a, b, weights):
    diff = np.array(a) - np.array(b)
    return np.sqrt(np.sum(weights * diff ** 2))

# GA dla każdej klasy
ideal_chromosomes = {}

for target_class in y.unique():
    print(f"Trenuję GA dla klasy: {target_class}")
    target_samples = X[y == target_class].values
    other_samples = X[y != target_class].values

    def fitness_func(individual):
        dist_to_class = np.mean([weighted_euclidean(individual, s, weights) for s in target_samples])
        dist_to_others = np.mean([weighted_euclidean(individual, s, weights) for s in other_samples])
        return (dist_to_others - dist_to_class,)

    toolbox.register("evaluate", fitness_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POP_SIZE)
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GEN, verbose=False)
    best = tools.selBest(pop, k=1)[0]
    ideal_chromosomes[target_class] = best

    toolbox.unregister("evaluate")
    toolbox.unregister("mate")
    toolbox.unregister("mutate")
    toolbox.unregister("select")

# Klasyfikacja
def classify(sample, chrom_db):
    distances = {label: weighted_euclidean(sample, np.array(chrom), weights) for label, chrom in chrom_db.items()}
    return min(distances, key=distances.get)

# Test
X_test, y_test = shuffle(X.values, y.reset_index(drop=True), random_state=42)
X_test = X_test[:len(X_test)]
y_test = y_test[:len(X_test)]

# Skuteczność per klasa
total_per_class = defaultdict(int)
correct_per_class = defaultdict(int)

for i in range(len(X_test)):
    true_label = y_test.iloc[i]
    pred = classify(X_test[i], ideal_chromosomes)
    total_per_class[true_label] += 1
    if pred == true_label:
        correct_per_class[true_label] += 1

print("\\n🎯 Skuteczność klasyfikacji (dla każdej klasy):")
for label in total_per_class:
    acc = 100 * correct_per_class[label] / total_per_class[label]
    print(f"{label:20s}: {acc:.2f}% ({correct_per_class[label]}/{total_per_class[label]})")

overall_accuracy = sum(correct_per_class.values()) / len(X_test) * 100
print(f"\\n📈 Skuteczność całkowita: {overall_accuracy:.2f}%")
