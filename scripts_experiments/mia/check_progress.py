import os
import csv
from tabulate import tabulate

# Set your base path and save path here
BASE_PATH = "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_sampling"
# BASE_PATH = "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/exp_mia_final_clipping"
SAVE_PATH = "/vol/miltank/users/kaiserj/Clipping_vs_Sampling/completeness.csv"

# Files required in each target subfolder
TARGET_REQUIRED = ["bti.npy", "keep.npy", "logits.npy", "model.pt", "scores.npy"]

# Files required in results folder
RESULTS_REQUIRED = ["adv.npy", "integrals.npy", "num_steps_list.npy", "samplewise_auc_R.npy", "samplewise_auc.npy"]

def check_target_folder(target_path):
    """Check each subfolder in target folder for required files.
    Returns (total_targets, portions_dict, models_count)."""
    subfolders = [
        os.path.join(target_path, d)
        for d in os.listdir(target_path)
        if os.path.isdir(os.path.join(target_path, d))
    ]
    total = len(subfolders)
    counts = {f: 0 for f in TARGET_REQUIRED}

    for sub in subfolders:
        try:
            files = set(os.listdir(sub))
        except FileNotFoundError:
            files = set()
        for f in TARGET_REQUIRED:
            if f in files:
                counts[f] += 1

    portions = {f: (counts[f] / total if total > 0 else 0) for f in TARGET_REQUIRED}
    models_count = counts.get("model.pt", 0)
    return total, portions, models_count

def check_results_folder(results_path):
    """Check if required files exist in results folder."""
    files = os.listdir(results_path) if os.path.exists(results_path) else []
    presence = {f: (f in files) for f in RESULTS_REQUIRED}
    return presence

def main(base_path, save_path):
    datasets = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    )

    all_rows = []
    headers = [
        "Dataset",
        "Experiment",
        "Seed",
        "# Target Subfolders",
        "# Models",
    ] + TARGET_REQUIRED + RESULTS_REQUIRED + ["Tasks"]

    # For bottom summary
    all_tasks_by_dataset = {}

    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        experiments = sorted(
            [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        )

        for exp in experiments:
            print(f"\n=== Dataset: {dataset} | Experiment: {exp} ===")
            exp_path = os.path.join(dataset_path, exp)
            seeds = sorted(
                [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
            )

            table = []

            for seed in seeds:
                seed_path = os.path.join(exp_path, seed)
                target_path = os.path.join(seed_path, "target")
                results_path = os.path.join(seed_path, "results")

                row = [dataset, exp, seed]
                tasks = []
                total = "Missing"
                models_count = 0

                if os.path.exists(target_path):
                    total, portions, models_count = check_target_folder(target_path)
                    row.append(total)
                    row.append(models_count)
                    # Visual check: number of targets
                    print(f"  Seed {seed}: {total} target subfolders")

                    for f in TARGET_REQUIRED:
                        row.append(f"{portions[f]*100:.1f}%")

                    # Task derivation from target completeness
                    if portions.get("logits.npy", 1) < 1:
                        tasks.append("save_logits")
                    if portions.get("scores.npy", 1) < 1:
                        tasks.append("save_scores")
                else:
                    row.append("Missing")
                    row.append(models_count)
                    row.extend(["-"] * len(TARGET_REQUIRED))

                if os.path.exists(results_path):
                    presence = check_results_folder(results_path)
                    for f in RESULTS_REQUIRED:
                        row.append("✓" if presence[f] else "✗")
                    # Task derivation for results
                    if not all(
                        [
                            presence.get(f, False)
                            for f in [
                                "adv.npy",
                                "integrals.npy",
                                "samplewise_auc_R.npy",
                                "samplewise_auc.npy",
                            ]
                        ]
                    ):
                        tasks.append("save_stats")
                else:
                    row.extend(["Missing"] * len(RESULTS_REQUIRED))
                    tasks.append("save_stats")

                # Normalize task order for stable grouping
                tasks = sorted(set(tasks))

                row.append(", ".join(tasks) if tasks else "-")
                table.append(row)
                all_rows.append(row)

                if tasks:
                    all_tasks_by_dataset.setdefault(dataset, {}).setdefault(exp, []).append(
                        {"seed": seed, "tasks": tasks, "models": models_count}
                    )

            print(tabulate(table, headers=headers, tablefmt="grid"))

    # Save all results to CSV
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_rows)

    # Print aggregated task summary at the bottom
    print("\n=== TASK SUMMARY ===")
    if not any(any(v for v in dataset_dict.values()) for dataset_dict in all_tasks_by_dataset.values()):
        print("No tasks required!")
    else:
        for dataset, exp_dict in all_tasks_by_dataset.items():
            print(f"\n## Dataset: {dataset}")
            for exp, tasks_list in exp_dict.items():
                if not tasks_list:
                    continue
                print(f"-- Experiment: {exp} --")
                # Group seeds by identical task sets
                groups = {}
                for item in tasks_list:
                    key = tuple(item["tasks"])  # tasks are sorted
                    groups.setdefault(key, []).append(item)

                for task_key, items in groups.items():
                    seeds = [it["seed"] for it in items]
                    models = [it["models"] for it in items]
                    tasks_str = ", ".join(task_key) if task_key else "-"
                    if len(items) > 1:
                        seeds_str = ", ".join(seeds)
                        print(
                            f"Seeds [{seeds_str}] → Tasks: {tasks_str} | Models: {models}"
                        )
                    else:
                        print(
                            f"Seed {seeds[0]} → Tasks: {tasks_str} | Models: {models[0]}"
                        )

if __name__ == "__main__":
    main(BASE_PATH, SAVE_PATH)
