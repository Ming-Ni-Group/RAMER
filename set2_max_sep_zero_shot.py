import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot EC prediction with max-separation selection.")
    parser.add_argument(
        "--test_name",
        required=True,
        type=str,
        help="Test embedding file name without extension (for example: NEW-392).",
    )
    parser.add_argument(
        "--background_library_h5",
        default="./Background_library/clean100_set.h5",
        type=str,
        help="Path to background library H5 file.",
    )
    parser.add_argument(
        "--background_library_dict",
        default="./Background_library/clean100_set_dict.json",
        type=str,
        help="Path to background library dictionary JSON.",
    )
    parser.add_argument(
        "--embedding_dir",
        default="./RAMER_embedding",
        type=str,
        help="Directory containing test H5 embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Directory to save output CSV.",
    )
    parser.add_argument(
        "--batch_size",
        default=10000,
        type=int,
        help="Batch size for processing test embeddings.",
    )
    return parser.parse_args()


def dynamic_k_selection(similarities, background_ecs):
    """
    Dynamically select K:
    - search max diff while EC types <= 10
    - final selected K should keep EC types <= 5
    """
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_sims = similarities[sorted_indices]

    ec_set = set()
    valid_range = 0
    for idx in sorted_indices:
        ecs = background_ecs[idx]
        ec_set.update(ecs)
        valid_range += 1
        if len(ec_set) > 10:
            break

    if valid_range <= 1:
        return sorted_indices[:1]

    diffs = sorted_sims[: valid_range - 1] - sorted_sims[1:valid_range]
    max_diff_idx = np.argmax(diffs)
    dynamic_k = max_diff_idx + 1

    ec_final_set = set()
    final_k = 0
    for i in range(dynamic_k):
        ecs = background_ecs[sorted_indices[i]]
        ec_final_set.update(ecs)
        final_k += 1
        if len(ec_final_set) > 5:
            final_k -= 1
            break

    return sorted_indices[:final_k]


def process_batch(batch_embeddings, batch_ids, background_norm, background_ecs):
    """Process one test batch and return prediction rows."""
    batch_norm = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
    similarity_matrix = np.dot(batch_norm, background_norm.T)

    batch_results = []
    for i, test_id in enumerate(batch_ids):
        similarities = similarity_matrix[i]
        top_k_idx = dynamic_k_selection(similarities, background_ecs)

        seen_ec = set()
        ordered_ecs = []
        ec_probabilities = {}

        for idx in top_k_idx:
            ecs = background_ecs[idx]
            sim = similarities[idx]
            for ec in ecs:
                if ec not in seen_ec:
                    seen_ec.add(ec)
                    ordered_ecs.append(ec)
                    ec_probabilities.setdefault(ec, []).append(sim)

        ec_probabilities = {ec: np.mean(probs) for ec, probs in ec_probabilities.items()}

        result = {
            "test_id": test_id,
            "predicted_ec": str(ordered_ecs),
            "ec_mean_cosine_similarity": str([round(float(ec_probabilities.get(ec, 0.0)), 4) for ec in ordered_ecs]),
        }
        batch_results.append(result)

    return batch_results


def main():
    args = parse_args()

    test_h5_file = os.path.join(args.embedding_dir, f"{args.test_name}.h5")
    output_csv = os.path.join(args.output_dir, f"{args.test_name}_max-sep.csv")

    if not os.path.exists(test_h5_file):
        raise FileNotFoundError(f"Test embedding file not found: {test_h5_file}")
    if not os.path.exists(args.background_library_h5):
        raise FileNotFoundError(f"Background library H5 not found: {args.background_library_h5}")
    if not os.path.exists(args.background_library_dict):
        raise FileNotFoundError(f"Background library dict not found: {args.background_library_dict}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading background library dict: {args.background_library_dict}")
    with open(args.background_library_dict, "r") as f:
        background_library_dict = json.load(f)

    print("Preloading background library embeddings...")
    with h5py.File(args.background_library_h5, "r") as f:
        background_ids = [tid.decode("utf-8") for tid in f["ids"]]
        background_embeddings = np.array(f["embeddings"])
        background_norm = background_embeddings / np.linalg.norm(background_embeddings, axis=1, keepdims=True)

    background_ecs = [background_library_dict.get(tid, ["unknown"]) for tid in background_ids]
    print(f"Background embedding shape: {background_embeddings.shape}")

    print(f"Loading test metadata from: {test_h5_file}")
    with h5py.File(test_h5_file, "r") as f:
        total_test = len(f["ids"])
        test_ids = [tid.decode("utf-8") for tid in f["ids"]]

    results_list = []

    print("Processing test batches...")
    with h5py.File(test_h5_file, "r") as f:
        test_embeddings = f["embeddings"]
        num_batches = int(np.ceil(total_test / args.batch_size))

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start = batch_idx * args.batch_size
            end = min((batch_idx + 1) * args.batch_size, total_test)
            batch_embeddings = test_embeddings[start:end]
            batch_ids = test_ids[start:end]
            batch_results = process_batch(batch_embeddings, batch_ids, background_norm, background_ecs)
            results_list.extend(batch_results)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()