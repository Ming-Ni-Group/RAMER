import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot EC prediction with top-1 retrieval.")
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
    parser.add_argument(
        "--top_k",
        default=1,
        type=int,
        help="Top-K nearest neighbors for label collection (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    test_h5_file = os.path.join(args.embedding_dir, f"{args.test_name}.h5")
    output_csv = os.path.join(args.output_dir, f"{args.test_name}_top1.csv")

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

    print(
        f"Batch size: {args.batch_size} "
        f"(total samples: {total_test}, total batches: {int(np.ceil(total_test / args.batch_size))})"
    )

    results = []
    print("Processing test batches...")
    with h5py.File(test_h5_file, "r") as f:
        test_embeddings = f["embeddings"]
        num_batches = int(np.ceil(total_test / args.batch_size))

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, total_test)
            batch_embeddings = test_embeddings[start_idx:end_idx]
            batch_ids = test_ids[start_idx:end_idx]

            batch_norm = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_similarity = np.dot(batch_norm, background_norm.T)
            top_k_indices = np.argsort(-batch_similarity, axis=1)[:, : args.top_k]

            for i, test_id in enumerate(batch_ids):
                k_indices = top_k_indices[i]
                top_ecs = set()
                for idx in k_indices:
                    ecs = background_ecs[idx]
                    if isinstance(ecs, list):
                        top_ecs.update(ecs)
                    else:
                        top_ecs.add(ecs)

                top1_similarity = str([round(float(batch_similarity[i, idx]), 4) for idx in k_indices])
                results.append(
                    {
                        "test_id": test_id,
                        "predicted_ec": str(sorted(list(top_ecs))),
                        "top1 cosine_similarity": top1_similarity,
                    }
                )

    df = pd.DataFrame(results)
    df = df[["test_id", "predicted_ec", "top1 cosine_similarity"]]
    df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()
