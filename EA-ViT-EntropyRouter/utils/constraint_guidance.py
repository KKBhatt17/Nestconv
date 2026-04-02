import csv
from collections import defaultdict

import torch

from utils.entropy_conditioning import encoding_to_mask_tensors, entropy_score_from_vectors


def load_pareto_data(file_name="pareto_front.csv"):
    data = defaultdict(list)
    with open(file_name, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            generation = int(row["Generation"])
            macs = float(row["MACs"])
            accuracy = float(row["Accuracy"])
            encoding = row["Encoding"]
            data[generation].append((macs, accuracy, encoding))

    return data


def get_preset_mask_nsga(gen_id, constraint, device, data):
    generation = data[gen_id]
    sorted_individuals = sorted(generation, key=lambda item: item[0])

    macs = [individual[0] for individual in sorted_individuals]
    encodings = [individual[2] for individual in sorted_individuals]

    if torch.is_tensor(constraint):
        constraint_value = float(constraint.detach().flatten()[0].item())
    else:
        constraint_value = float(constraint)

    index = min(range(len(macs)), key=lambda i: abs(macs[i] - constraint_value))
    return encoding_to_mask_tensors(encodings[index], device)


def load_constraint_guide(file_name):
    rows = []
    with open(file_name, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "entropy_mean": float(row["EntropyMean"]),
                    "macs": float(row["MACs"]),
                }
            )

    rows.sort(key=lambda item: item["entropy_mean"])
    return rows


def get_target_constraint_from_entropy(entropy_vectors, guide_rows, device):
    entropy_mean = entropy_score_from_vectors(entropy_vectors)
    guide_entry = min(guide_rows, key=lambda row: abs(float(row["entropy_mean"]) - float(entropy_mean)))
    target_constraint = torch.tensor([guide_entry["macs"]], device=device, dtype=torch.float32)
    return target_constraint, entropy_mean
