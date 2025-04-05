import numpy as np
import pandas as pd
from pydantic import BaseModel

import argparse
import random


# got from https://www.tensorflow.org/datasets/catalog/kddcup99
feature_names = ["duration", "protocol_type", "service", "flag", 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label']

test_df = pd.read_csv("kddcup.data.gz", names=feature_names, compression='gzip') # http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
train_df = pd.read_csv("corrected.gz", names=feature_names, compression='gzip') # http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz

# fix label names
train_df["label"] = train_df["label"].apply(lambda x: x[:-1])
test_df["label"] = test_df["label"].apply(lambda x: x[:-1])

discrete = ["protocol_type", "service", "flag"]
boolean = ["logged_in", "root_shell", "su_attempted", "is_hot_login", "is_guest_login"]
continuous = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot", "num_compromised",
              "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
              "count", "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_count", "srv_serror_rate", "srv_rerror_rate", "srv_diff_host_rate"]
labels = list(set(train_df["label"]))

discrete_values = {key: list(set(train_df[key])) for key in discrete}
continuous_min_max = {key: (train_df[key].min(), train_df[key].max()) for key in continuous}


class Chromosome(BaseModel):
    discrete: dict
    boolean: dict
    continuous: dict
    label: str

    @classmethod
    def generate_random(cls, label: str, full_random: bool = False) -> "Chromosome":
        if full_random:
            discrete_dict = {feature: random.choice(train_df[feature].unique()) for feature in discrete}
            boolean_dict = {feature: random.choice(train_df[feature].unique()) for feature in boolean}
            continuous_dict = {}
            for feature in continuous:
                min_val = train_df[feature].min()
                max_val = train_df[feature].max()

                val_0 = random.uniform(min_val, max_val)
                val_1 = random.uniform(min_val, max_val)
                continuous_dict[feature] = (val_0, val_1) if val_0 < val_1 else (val_1, val_0)

        else:
            label_df = train_df[train_df["label"] == label]

            discrete_dict = {}
            for feature in discrete:
                feature_values = list(set(label_df[feature]))
                discrete_dict[feature] = random.choice(feature_values)

            boolean_dict = {}
            for feature in boolean:
                feature_values = list(set(label_df[feature].astype(int)))
                boolean_dict[feature] = random.choice(feature_values)

            continuous_dict = {}
            for feature in continuous:
                min_val = label_df[feature].min()
                max_val = label_df[feature].max()

                val_0 = random.uniform(min_val, max_val)
                val_1 = random.uniform(min_val, max_val)
                continuous_dict[feature] = (val_0, val_1) if val_0 < val_1 else (val_1, val_0)

        return cls(discrete=discrete_dict, boolean=boolean_dict, continuous=continuous_dict, label=label)

    def get_expression(self):
        expr_discrete = " & ".join([f"{key} == '{value}'" for key, value in self.discrete.items()])
        expr_boolean = " & ".join([f"{key} == {value}" for key, value in self.boolean.items()])
        expr_continuous = " & ".join(
            [f"{key} >= {value[0]} & {key} <= {value[1]}" for key, value in self.continuous.items()])
        expression = " & ".join(
            [
                expr_discrete,
                expr_boolean,
                expr_continuous
            ]
        )
        return expression


def euclidean_distance_vectorized(chromosome, df_subset):
    distances = np.zeros(len(df_subset))

    for feature in discrete:
        not_matching = df_subset[feature] != chromosome.discrete[feature]
        distances += not_matching.astype(float)

    for feature in boolean:
        not_matching = df_subset[feature].astype(int) != chromosome.boolean[feature]
        distances += not_matching.astype(float)

    for feature in continuous:
        min_val, max_val = continuous_min_max[feature]
        range_val = max_val - min_val if max_val > min_val else 1

        chrom_val = sum(chromosome.continuous[feature]) / 2

        norm_chrom_val = (chrom_val - min_val) / range_val
        norm_df_vals = (df_subset[feature] - min_val) / range_val

        distances += (norm_chrom_val - norm_df_vals) ** 2

    return np.sqrt(distances)


def check_chromosome_vectorized(chromosome: Chromosome, df: pd.DataFrame, threshold=2):
    same_label_df = df[df["label"] == chromosome.label]
    other_label_df = df[df["label"] != chromosome.label]

    same_label_distances = euclidean_distance_vectorized(chromosome, same_label_df)
    other_label_distances = euclidean_distance_vectorized(chromosome, other_label_df)

    true_positives = np.sum(same_label_distances <= threshold)
    false_negatives = len(same_label_df) - true_positives

    true_negatives = np.sum(other_label_distances > threshold)
    false_positives = len(other_label_df) - true_negatives

    return {
        "TRUE_POSITIVE": true_positives,
        "FALSE_POSITIVE": false_positives,
        "FALSE_NEGATIVE": false_negatives,
        "TRUE_NEGATIVE": true_negatives
    }


def fitness_fun_vectorized(chromosome: Chromosome, df: pd.DataFrame) -> float:
    measures = check_chromosome_vectorized(chromosome, df)
    try:
        tp_rate = measures["TRUE_POSITIVE"] / (measures["TRUE_POSITIVE"] + measures["FALSE_NEGATIVE"])
    except ZeroDivisionError:
        tp_rate = 0

    try:
        tn_rate = measures["TRUE_NEGATIVE"] / (measures["TRUE_NEGATIVE"] + measures["FALSE_POSITIVE"])
    except ZeroDivisionError:
        tn_rate = 0

    return (tp_rate + tn_rate) / 2 if (tp_rate + tn_rate) > 0 else 0


def mutate(chromosome: Chromosome) -> Chromosome:
    if random.random() < 0.5:
        random_boolean_feature = random.choice(boolean)
        chromosome.boolean[random_boolean_feature] ^= 1

    else:
        random_discrete_feature = random.choice(discrete)
        valid_values = list(discrete_values[random_discrete_feature])
        valid_values.remove(chromosome.discrete[random_discrete_feature])
        new_value = random.choice(valid_values)
        chromosome.discrete[random_discrete_feature] = new_value

    return chromosome


def crossover(chromosome1: Chromosome, chromosome2: Chromosome) -> tuple[Chromosome, Chromosome]:
    offspring1 = chromosome1.model_copy(deep=True)
    offspring2 = chromosome2.model_copy(deep=True)

    for feature_set_name in ['discrete', 'boolean', 'continuous']:
        feature_set1 = getattr(offspring1, feature_set_name)
        feature_set2 = getattr(offspring2, feature_set_name)
        for feature_name, feature_value in list(feature_set1.items())[0:len(feature_set1) // 2]:
            feature_set2[feature_name], feature_set1[feature_name] = feature_set1[feature_name], feature_set2[
                feature_name]

    return offspring1, offspring2


def evolve(df: pd.DataFrame, population_size=100, iterations=100, mutation_rate=0.01, label="normal", stagnation=5, elitism=True, full_random=False):
    population = [Chromosome.generate_random(label, full_random) for _ in range(population_size)]
    best_fitness_scores = []
    stagnation_counter = 0

    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")

        fitness = [fitness_fun_vectorized(chrom, df) for chrom in population]
        # print(fitness)

        most_fit_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)[
                           :int(0.3 * population_size)]
        best_fitness_scores.append(fitness[most_fit_indices[0]])
        best_fitness = fitness[most_fit_indices[0]]
        # print(best_fitness_scores)

        if i > 0 and best_fitness == best_fitness_scores[-2]:
            stagnation_counter += 1
            if stagnation_counter >= stagnation:
                break
        else:
            stagnation_counter = 0

        population = [population[index] for index in most_fit_indices]

        next_generation = []
        while len(next_generation) < int(0.35 * population_size):
            parent1, parent2 = random.choices(population, k=2)
            offspring1, offspring2 = crossover(parent1, parent2)
            next_generation.extend([offspring1, offspring2])

        for i in range(len(next_generation)):
            if random.random() < mutation_rate:
                next_generation[i] = mutate(next_generation[i])

        if elitism:
            next_generation.extend(population[:int(0.05 * population_size)])

        while len(next_generation) < population_size:
            next_generation.append(Chromosome.generate_random(label, full_random))

        population = next_generation

    return population, best_fitness_scores


def find_best_chromosome(population, df):
    fitness_scores = [fitness_fun_vectorized(chromosome, df) for chromosome in population]
    best_index = fitness_scores.index(max(fitness_scores))
    return population[best_index], fitness_scores[best_index]


# iterations = 20
# population_size = 100
# mutation_rate = 0.05
#
# final_population, _ = evolve(train_df,  population_size, iterations, mutation_rate, "smurf", 5, False)
#
# best_chromosome, best_chromosome_fitness = find_best_chromosome(final_population, train_df)
# print(f"Best chromosome fitness: {best_chromosome_fitness}")
#
# test_fitnesses = [fitness_fun_vectorized(chromosome, test_df) for chromosome in final_population]
# test_max_fitness = max(test_fitnesses)
# print(f"Maximum fitness on test set: {test_max_fitness}")
#
# for i in final_population:
#     print(f"Fitness: {fitness_fun_vectorized(i, train_df)}")
#     print(f"label: {i.label}")
#     print(f"chromosome: {i.get_expression()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic algorithm for KDD99 dataset.")

    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations (generations)")
    parser.add_argument("--population_size", type=int, default=10, help="Size of the population")
    parser.add_argument("--mutation_rate", type=float, default=0.05, help="Mutation rate")
    parser.add_argument("--label", type=str, default="smurf", help="Label for which to generate chromosomes")
    parser.add_argument("--stagnation", type=int, default=5, help="Stagnation limit (early stopping)")
    parser.add_argument("--elitism", action="store_true", help="Enable elitism in the evolution")
    parser.add_argument("--full_random", action="store_true", help="Use full random generation instead of label-based")

    args = parser.parse_args()

    final_population, best_fitness_scores = evolve(
        train_df,
        population_size=args.population_size,
        iterations=args.iterations,
        mutation_rate=args.mutation_rate,
        label=args.label,
        stagnation=args.stagnation,
        elitism=args.elitism,
        full_random=args.full_random
    )

    print(f"Best fitness score: {best_fitness_scores}")

    best_chromosome, best_chromosome_fitness = find_best_chromosome(final_population, train_df)
    print(f"Best chromosome fitness: {best_chromosome_fitness}")

    test_fitnesses = [fitness_fun_vectorized(chromosome, test_df) for chromosome in final_population]
    test_max_fitness = max(test_fitnesses)
    print(f"Maximum fitness on test set: {test_max_fitness}")

    # for i in final_population:
    #     print(f"Fitness: {fitness_fun_vectorized(i, train_df)}")
    #     print(f"label: {i.label}")
    #     print(f"chromosome: {i.get_expression()}")

