import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Wczytanie danych
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

# Kodowanie kategorycznych danych
categorical_cols = ["protocol_type", "service", "flag", "label"]
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Dodatkowa kolumna z oryginalną etykietą
df["label_str"] = le_dict["label"].inverse_transform(df["label"])

# Wybrane cechy przez algorytm genetyczny (podmień na swoje, jeśli masz inne!)
selected_features = [0, 2, 4, 8, 15, 28, 31, 35, 36, 37]
selected_feature_names = [df.columns[i] for i in selected_features]
print("Używamy cech:", selected_feature_names)

# Dane wejściowe (X) i etykiety (y)
X = df[selected_feature_names]
y = df["label_str"]

# Normalizacja
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === TWORZENIE IDEALNYCH CHROMOSOMÓW DLA KAŻDEJ KLASY ===
ideal_chromosomes = {}
for attack_type in y.unique():
    attack_samples = X_scaled[y == attack_type]
    ideal_chromosomes[attack_type] = np.mean(attack_samples, axis=0)

print("\nUtworzono idealne chromosomy dla klas:", list(ideal_chromosomes.keys()))

# === FUNKCJE DISTANSE I KLASYFIKACJI ===
def weighted_distance(a, b, weights=None):
    if weights is None:
        weights = np.ones(len(a))
    return np.sum(weights * np.abs(a - b))

def classify_by_distance(sample, ideal_chromosomes, weights=None):
    distances = {
        label: weighted_distance(sample, chrom, weights)
        for label, chrom in ideal_chromosomes.items()
    }
    predicted_label = min(distances, key=distances.get)
    return predicted_label, distances

# === WYBÓR DOWOLNEGO WIERSZA ===
sample_index = int(input("\nPodaj numer próbki (np. 1234): "))
sample = X_scaled[sample_index]
real_label = y.iloc[sample_index]

predicted, distances = classify_by_distance(sample, ideal_chromosomes)

# === WYNIKI ===
print(f"\n📌 Rzeczywista etykieta próbki: {real_label}")
print(f"✅ Najbardziej podobna do: {predicted}")

print("\n📏 Dystanse do idealnych chromosomów:")
for label, dist in sorted(distances.items(), key=lambda x: x[1]):
    print(f"{label}: {dist:.4f}")


print(f"Próbka #{sample_index} to faktycznie: {y.iloc[sample_index]}")
