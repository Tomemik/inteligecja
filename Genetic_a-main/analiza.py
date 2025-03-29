import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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

# Kodowanie kolumn kategorycznych
categorical_cols = ["protocol_type", "service", "flag", "label"]
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Kolumna z oryginalną etykietą jako string
df["label_str"] = le_dict["label"].inverse_transform(df["label"])

# UŻYWANIE WSZYSTKICH CECH
X = df.drop(columns=["label", "label_str"])
y = df["label_str"]

# Normalizacja
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# === TWORZENIE WAG CECH ===

# Lista nazw wszystkich cech (po kolumnach X)
all_feature_names = X.columns.tolist()

# Inicjalizacja wag – domyślnie 1.0 dla każdej cechy
weights = np.ones(len(all_feature_names))

# Ustawienie większych wag dla najważniejszych cech wg dokumentacji KDD/NSL-KDD
important_features = {
    "serror_rate": 2.0,
    "srv_serror_rate": 2.0,
    "dst_host_serror_rate": 2.0,
    "dst_host_srv_serror_rate": 2.0,
    "count": 1.8,
    "srv_count": 1.8,
    "dst_host_same_srv_rate": 1.5,
    "dst_host_same_src_port_rate": 1.5
}

# Przypisanie wag
for feat_name, weight in important_features.items():
    if feat_name in all_feature_names:
        feat_index = all_feature_names.index(feat_name)
        weights[feat_index] = weight


# Tworzenie idealnych chromosomów
ideal_chromosomes = {}
for label in y.unique():
    ideal_chromosomes[label] = np.mean(X_scaled[y == label], axis=0)

# Funkcje dystansu i klasyfikacji
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
    return predicted_label

# Weryfikacja na 1000 losowych próbkach
num_samples = 100000
sampled_df = df.sample(n=num_samples)
X_sampled = scaler.transform(sampled_df.drop(columns=["label", "label_str"]))
y_sampled = sampled_df["label_str"].values

correct = 0

print("🔍 Weryfikacja 1000 losowych próbek (wszystkie cechy):\n")

for i in range(num_samples):
    sample = X_sampled[i]
    true_label = y_sampled[i]
    predicted_label = classify_by_distance(sample, ideal_chromosomes, weights)

    if predicted_label == true_label:
        print(f"{i+1:4d}. ✅ {true_label}")
        correct += 1
    else:
        print(f"{i+1:4d}. ❌ {true_label} → przewidziano: {predicted_label}")

# Ogólna skuteczność
accuracy = correct / num_samples * 100
print(f"\n🎯 Skuteczność klasyfikacji: {correct}/{num_samples} ({accuracy:.2f}%)")
