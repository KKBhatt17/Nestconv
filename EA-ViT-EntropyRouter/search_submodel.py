import csv
import os
import random

from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import get_args_parser
from dataloader.entropy_image_datasets import build_entropy_image_dataset, create_entropy_dataloader
from models.model_stage2 import EAViTStage2, ModifiedBlock
from utils.entropy_conditioning import (
    build_router_input,
    default_entropy_lookup_path,
    entropy_score_from_vectors,
)


random.seed(42)

args = get_args_parser()
router_input_dim = (args.input_size // args.entropy_patch_size) ** 2

dataset_train, dataset_val, nb_classes = build_entropy_image_dataset(args)
trainDataLoader = create_entropy_dataloader(args, dataset_train, shuffle_batches=False)

model = EAViTStage2(
    embed_dim=768,
    depth=12,
    mlp_ratio=4,
    num_heads=12,
    num_classes=nb_classes,
    drop_path_rate=args.drop_path,
    qkv_bias=True,
    block=ModifiedBlock,
    router_input_dim=router_input_dim,
)
device = args.device

checkpoint = torch.load(args.stage1_checkpoint_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

NUM_GENES = 12 + 12 * 8 + 12 * 12 + 12 + 12
GENERATIONS = 301
MUTATION_PROBABILITY = 0.3
CROSSOVER_PROBABILITY = 0.95
POPULATION_SIZE = 5

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-0.5, 1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def resolve_batch_limit(requested_batches, total_batches):
    if requested_batches <= 0 or requested_batches > total_batches:
        return total_batches
    return requested_batches


def collect_representative_batches(data_loader, requested_batches):
    total_batches = len(data_loader)
    if total_batches == 0:
        raise RuntimeError("No batches available for NSGA evaluation.")

    batch_limit = resolve_batch_limit(requested_batches, total_batches)
    if batch_limit == total_batches:
        target_indices = set(range(total_batches))
    else:
        target_indices = set(
            torch.linspace(0, total_batches - 1, steps=batch_limit).round().long().tolist()
        )

    batches = []
    for batch_index, batch in enumerate(data_loader):
        if batch_index in target_indices:
            batches.append(batch)

    return batches


REPRESENTATIVE_BATCHES = collect_representative_batches(trainDataLoader, args.nsga_eval_batches)


def load_population_from_csv(csv_path, pop_size):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    rows = []
    with open(csv_path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("CSV is empty")

    gen_counts = {}
    for row in rows:
        generation = int(row["Generation"])
        gen_counts[generation] = gen_counts.get(generation, 0) + 1

    gens_sorted = sorted(gen_counts.keys())
    last_gen = gens_sorted[-1]

    if gen_counts[last_gen] < pop_size:
        last_gen -= 1
        if last_gen not in gen_counts or gen_counts[last_gen] < pop_size:
            raise ValueError("can not restore")

    target_rows = [row for row in rows if int(row["Generation"]) == last_gen][:pop_size]

    population = []
    for row in target_rows:
        encoding = list(map(int, row["Encoding"]))
        individual = creator.Individual(encoding)
        macs = float(row["MACs"])
        accuracy = float(row["Accuracy"])
        individual.fitness.values = (macs, accuracy)
        population.append(individual)

    start_gen = last_gen + 1
    print(f"restore {len(population)} population from Generation {last_gen}")
    return population, start_gen


def create_individual():
    if random.randint(0, 1) > 0.5:
        return [1 for _ in range(NUM_GENES)]
    return [0 for _ in range(NUM_GENES)]


def create_individual_random():
    return [random.randint(0, 1) for _ in range(NUM_GENES)]


def individual_to_masks(vector):
    vector = list(vector)
    vector[0] = 1

    embed_sum = int(sum(vector[:12]))
    embed_mask = torch.tensor([1] * embed_sum + [0] * (12 - embed_sum), device=device, dtype=torch.float32)

    depth_attn_mask = torch.tensor(vector[12:24], device=device, dtype=torch.float32)
    depth_mlp_mask = torch.tensor(vector[24:36], device=device, dtype=torch.float32)

    mask_attn = []
    mask_mlp = []

    for index in range(12):
        attn_sum = int(sum(vector[36 + index * 12: 36 + (index + 1) * 12]))
        mask_attn.append(
            torch.tensor([1] * attn_sum + [0] * (12 - attn_sum), device=device, dtype=torch.float32)
        )

    for index in range(12):
        mlp_sum = int(sum(vector[180 + index * 8: 180 + (index + 1) * 8]))
        mask_mlp.append(
            torch.tensor([1] * mlp_sum + [0] * (8 - mlp_sum), device=device, dtype=torch.float32)
        )

    return embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask


def evaluate_candidate_on_batch(candidate_encoding, img, label, entropy_vectors):
    embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask = individual_to_masks(candidate_encoding)
    router_input = build_router_input(entropy_vectors, device)
    model.configure_router_input(router_input=router_input, tau=1)
    model.set_mask(embed_mask, mask_attn, mask_mlp, depth_attn_mask, depth_mlp_mask)

    preds, _, _, _, _, _, total_macs = model(img)
    accuracy = (preds.argmax(dim=1) == label).float().mean().item()
    return total_macs.item(), accuracy


def evaluate(vector):
    candidate_encoding = list(vector)
    candidate_encoding[0] = 1

    total_accuracy = 0.0
    total_macs = 0.0

    with torch.no_grad():
        for img, label, entropy_vectors, _ in REPRESENTATIVE_BATCHES:
            img = img.to(device)
            label = label.to(device)
            macs, accuracy = evaluate_candidate_on_batch(candidate_encoding, img, label, entropy_vectors)
            total_accuracy += accuracy
            total_macs += macs

    return total_macs / len(REPRESENTATIVE_BATCHES), total_accuracy / len(REPRESENTATIVE_BATCHES)


def plot_pareto_front(population, generation):
    plt.figure(figsize=(10, 6))

    macs = [individual.fitness.values[0] for individual in population]
    acc = [individual.fitness.values[1] for individual in population]

    front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    f_macs = [individual.fitness.values[0] for individual in front]
    f_acc = [individual.fitness.values[1] for individual in front]

    plt.scatter(macs, acc, c="blue", alpha=0.5, label="Population")
    plt.scatter(f_macs, f_acc, c="red", marker="x", label="Pareto Front")

    plt.xlabel("MACs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Pareto Front @ Generation {generation}", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def save_population(population, generation, file_path="population.csv"):
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Generation", "MACs", "Accuracy", "Encoding", "IsPareto"])

        for individual in population:
            macs, acc = individual.fitness.values
            encoding = "".join(map(str, individual))
            is_pareto = individual in pareto_front
            writer.writerow([generation, macs, acc, encoding, int(is_pareto)])


def assign_macs_global_crowding(population):
    count = len(population)
    macs_vals = np.array([individual.fitness.values[0] for individual in population])

    idx = np.argsort(macs_vals)
    vmin, vmax = macs_vals[idx[0]], macs_vals[idx[-1]]
    crowding_scores = np.zeros(count)

    if vmax > vmin:
        crowding_scores[idx[0]] = (macs_vals[idx[1]] - vmin) / (vmax - vmin)
        crowding_scores[idx[-1]] = (vmax - macs_vals[idx[-2]]) / (vmax - vmin)
    for k in range(1, count - 1):
        i = idx[k]
        crowding_scores[i] = (macs_vals[idx[k + 1]] - macs_vals[idx[k - 1]]) / (vmax - vmin + 1e-12)

    for individual, crowding in zip(population, crowding_scores):
        individual.macs_crowding = crowding


def select_by_partition_incremental(pop, toolbox, cxpb, mutpb, quotas, bins, max_iters, pop_size, min_mac_diff=0.001):
    combined = pop
    for iteration in range(max_iters + 1):
        source = pop if iteration == 0 else combined
        offspring = algorithms.varAnd(source, toolbox, cxpb, mutpb)

        invalid = [individual for individual in offspring if not individual.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid)
        for individual, fit in zip(invalid, fits):
            individual.fitness.values = fit
        combined = combined + offspring if iteration > 0 else pop + offspring

        assign_macs_global_crowding(combined)
        fronts = tools.sortNondominated(combined, len(combined))

        selected = []
        for index, quota in enumerate(quotas):
            low, high = bins[index], bins[index + 1]
            seen_codes = set()
            selected_bin = []

            for front in fronts:
                candidates = [
                    individual
                    for individual in front
                    if (index < len(quotas) - 1 and low <= individual.fitness.values[0] < high)
                    or (index == len(quotas) - 1 and low <= individual.fitness.values[0] <= high)
                ]
                if not candidates:
                    continue

                assign_macs_global_crowding(selected_bin + candidates)
                candidates_sorted = sorted(candidates, key=lambda individual: individual.macs_crowding, reverse=True)

                for individual in candidates_sorted:
                    code = "".join(map(str, individual))
                    mac = individual.fitness.values[0]
                    if code in seen_codes:
                        continue
                    if any(abs(mac - existing.fitness.values[0]) < min_mac_diff for existing in selected_bin):
                        continue
                    seen_codes.add(code)
                    selected_bin.append(individual)
                    if len(selected_bin) >= quota:
                        break
                if len(selected_bin) >= quota:
                    break

            selected.extend(selected_bin)

        if len(selected) >= pop_size:
            return selected[:pop_size]

    final_population = selected[:]
    need = pop_size - len(final_population)
    if need > 0:
        immigrants = [toolbox.individual_random() for _ in range(need)]
        fits = toolbox.map(toolbox.evaluate, immigrants)
        for individual, fit in zip(immigrants, fits):
            individual.fitness.values = fit
        final_population.extend(immigrants)
    return final_population


def get_lookup_candidates(population):
    candidates = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    if not candidates:
        candidates = population

    unique_candidates = {}
    for candidate in candidates:
        encoding = "".join(map(str, candidate))
        if encoding not in unique_candidates:
            unique_candidates[encoding] = candidate

    return sorted(
        unique_candidates.values(),
        key=lambda candidate: (candidate.fitness.values[0], -candidate.fitness.values[1]),
    )


def accuracy_tolerance_for_position(batch_index, lookup_batch_limit):
    if lookup_batch_limit <= 1:
        position = 1.0
    else:
        position = batch_index / (lookup_batch_limit - 1)
    return args.lookup_acc_tolerance_low + position * (args.lookup_acc_tolerance_high - args.lookup_acc_tolerance_low)


def save_entropy_lookup(population, data_loader, file_path):
    candidates = get_lookup_candidates(population)
    lookup_batch_limit = resolve_batch_limit(args.lookup_batches, len(data_loader))

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["BatchIndex", "EntropyMean", "Accuracy", "Encoding"])

        with torch.no_grad():
            for batch_index, (img, label, entropy_vectors, _) in enumerate(data_loader):
                if batch_index >= lookup_batch_limit:
                    break

                img = img.to(device)
                label = label.to(device)
                entropy_mean = entropy_score_from_vectors(entropy_vectors)

                candidate_results = []
                for candidate in candidates:
                    candidate_encoding = list(candidate)
                    candidate_encoding[0] = 1
                    macs, accuracy = evaluate_candidate_on_batch(candidate_encoding, img, label, entropy_vectors)
                    candidate_results.append(
                        {
                            "macs": macs,
                            "accuracy": accuracy,
                            "encoding": "".join(map(str, candidate_encoding)),
                        }
                    )

                best_accuracy = max(result["accuracy"] for result in candidate_results)
                accuracy_tolerance = accuracy_tolerance_for_position(batch_index, lookup_batch_limit)
                eligible_results = [
                    result for result in candidate_results
                    if result["accuracy"] >= best_accuracy - accuracy_tolerance
                ]
                chosen_result = min(eligible_results, key=lambda result: (result["macs"], -result["accuracy"]))

                writer.writerow([batch_index, entropy_mean, chosen_result["accuracy"], chosen_result["encoding"]])


toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("individual_random", tools.initIterate, creator.Individual, create_individual_random)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=MUTATION_PROBABILITY)
toolbox.register("evaluate", evaluate)


def main():
    csv_path = args.nsga_path

    if os.path.isfile(csv_path):
        try:
            population, start_gen = load_population_from_csv(csv_path, POPULATION_SIZE)
        except Exception as exc:
            print("restore failure:", exc)
            population = toolbox.population(n=POPULATION_SIZE)
            start_gen = 0
    else:
        population = toolbox.population(n=POPULATION_SIZE)
        start_gen = 0

    unevaluated = [individual for individual in population if not individual.fitness.valid]
    if unevaluated:
        fits = toolbox.map(toolbox.evaluate, unevaluated)
        for individual, fit in zip(unevaluated, fits):
            individual.fitness.values = fit

    hof = tools.ParetoFront()

    fitnesses = toolbox.map(toolbox.evaluate, population)
    for individual, fit in zip(population, fitnesses):
        individual.fitness.values = fit

    for generation in tqdm(range(start_gen, GENERATIONS)):
        random.shuffle(population)

        population = select_by_partition_incremental(
            population,
            toolbox,
            cxpb=CROSSOVER_PROBABILITY,
            mutpb=MUTATION_PROBABILITY,
            quotas=[2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4],
            bins=np.linspace(0.0, 1.0, 21),
            max_iters=3,
            pop_size=POPULATION_SIZE,
            min_mac_diff=0.002,
        )

        hof.update(population)
        save_population(population, generation, file_path=csv_path)

    entropy_lookup_path = args.entropy_lookup_path or default_entropy_lookup_path(args.nsga_path)
    save_entropy_lookup(list(hof) if len(hof) > 0 else population, trainDataLoader, entropy_lookup_path)
    print(f"Saved entropy lookup to {entropy_lookup_path}")


if __name__ == "__main__":
    main()
