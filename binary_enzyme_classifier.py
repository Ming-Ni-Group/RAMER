import argparse
import os

import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Binary enzyme/non-enzyme classification from RAMER embeddings.")
    parser.add_argument(
        "--input_name",
        required=True,
        type=str,
        help="Input H5 base name under ./RAMER_embedding (for example: NEW-392).",
    )
    parser.add_argument(
        "--embedding_dir",
        default="./RAMER_embedding",
        type=str,
        help="Directory containing test embedding H5 files.",
    )
    parser.add_argument(
        "--model_path",
        default="./model/RAMER_enzyme_classifier_xgb.model",
        type=str,
        help="Path to XGBoost classifier model.",
    )
    parser.add_argument(
        "--save_dir",
        default="./output",
        type=str,
        help="Directory to save prediction CSV.",
    )
    parser.add_argument(
        "--chunk_size",
        default=100000,
        type=int,
        help="Chunk size for inference.",
    )
    parser.add_argument(
        "--threshold",
        default=0.360,
        type=float,
        help="Classification threshold for enzyme label.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    h5_path = os.path.join(args.embedding_dir, f"{args.input_name}.h5")
    output_csv = os.path.join(args.save_dir, f"{args.input_name}_enzyme_classifier.csv")
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Input H5 file not found: {h5_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    clf = xgb.XGBClassifier()
    clf.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")
    print(f"Input H5 file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        total_samples = f["embeddings"].shape[0]
        has_labels = "labels" in f
        print(f"Total samples: {total_samples:,}")
        if has_labels:
            print("Detected labels in H5 file. Evaluation metrics will be computed.")

    all_labels, all_pred_probs, all_pred_labels = [], [], []
    result_rows = []

    with h5py.File(h5_path, "r") as f:
        embeddings_ds = f["embeddings"]
        ids_ds = f["ids"]
        labels_ds = f["labels"] if has_labels else None

        for start in tqdm(range(0, total_samples, args.chunk_size), desc="Predicting", ncols=100):
            end = min(start + args.chunk_size, total_samples)
            batch_emb = np.array(embeddings_ds[start:end])
            batch_ids = [x.decode("utf-8") for x in ids_ds[start:end]]

            pred_probs = clf.predict_proba(batch_emb)[:, 1]
            pred_labels = (pred_probs > args.threshold).astype(int)

            for pid, pred_label, pred_prob in zip(batch_ids, pred_labels, pred_probs):
                result_rows.append(
                    {
                        "test_id": pid,
                        "is_enzyme": int(pred_label),
                        "enzyme_probability": float(pred_prob),
                    }
                )

            if has_labels:
                batch_labels = np.array(labels_ds[start:end])
                all_labels.extend(batch_labels)
                all_pred_probs.extend(pred_probs)
                all_pred_labels.extend(pred_labels)

    result_df = pd.DataFrame(result_rows, columns=["test_id", "is_enzyme", "enzyme_probability"])
    result_df.to_csv(output_csv, index=False)
    print(f"Saved prediction CSV: {output_csv}")

    if has_labels:
        labels = np.array(all_labels)
        pred_probs = np.array(all_pred_probs)
        pred_labels = np.array(all_pred_labels)

        auc = roc_auc_score(labels, pred_probs)
        acc = accuracy_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels)
        recall = recall_score(labels, pred_labels)
        f1 = f1_score(labels, pred_labels)

        print("\nPrediction metrics:")
        print(f"AUC        = {auc:.4f}")
        print(f"Accuracy   = {acc:.4f}")
        print(f"Precision  = {precision:.4f}")
        print(f"Recall     = {recall:.4f}")
        print(f"F1-score   = {f1:.4f}")
    else:
        print("\nNo labels found in H5 file. Skipping metric computation.")


if __name__ == "__main__":
    main()
